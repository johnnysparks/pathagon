use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    bit, captures_from, column_mask, neighbor_mask_for, row_mask, squares, Action, GameState,
    Player, MAX_CELL_COUNT,
};

const WIN_SCORE: i32 = 1_000_000_000;
const NEG_INF: i32 = i32::MIN / 4;
const POS_INF: i32 = i32::MAX / 4;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluationWeights {
    pub path: i32,
    pub material: i32,
    pub capture: i32,
    pub structure: i32,
    pub threat: i32,
    pub edge: i32,
}

impl Default for EvaluationWeights {
    fn default() -> Self {
        Self {
            path: 240,
            material: 110,
            capture: 700,
            structure: 55,
            threat: 130,
            edge: 80,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchConfig {
    pub depth: u8,
    pub max_nodes: u64,
    pub beam_width: usize,
    pub weights: EvaluationWeights,
    pub tactical_proof_horizon: Option<u8>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            depth: 4,
            max_nodes: 90_000,
            beam_width: 40,
            weights: EvaluationWeights::default(),
            tactical_proof_horizon: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchResult {
    pub action: Option<Action>,
    pub score: i32,
    pub nodes: u64,
    pub exhausted: bool,
    pub completed_depth: u8,
    pub table_hits: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveEvaluation {
    pub action: Action,
    pub before_score: i32,
    pub score: i32,
    pub delta: i32,
    pub nodes: u64,
    pub exhausted: bool,
    pub completed_depth: u8,
    pub table_hits: u64,
}

#[derive(Default)]
struct Budget {
    nodes: u64,
    exhausted: bool,
    table_hits: u64,
}

/// Per-search move-ordering hints used by the hybrid root-limited search.
///
/// Pathfinder's static ordering is still the source of truth. These two
/// quiet cut-off moves are only a cheap way to make alpha-beta encounter a
/// previously successful refutation earlier on sibling nodes and during the
/// next iterative-deepening pass. They never change the legal move set.
#[derive(Default)]
struct SearchHints {
    killers: Vec<[Option<Action>; 2]>,
}

impl SearchHints {
    fn killers_at(&self, ply: usize) -> [Option<Action>; 2] {
        self.killers.get(ply).copied().unwrap_or([None, None])
    }

    fn record_killer(&mut self, ply: usize, action: Action) {
        if self.killers.len() <= ply {
            self.killers.resize(ply + 1, [None, None]);
        }
        let slot = &mut self.killers[ply];
        if slot[0] != Some(action) {
            slot[1] = slot[0];
            slot[0] = Some(action);
        }
    }
}

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TableEntry {
    depth: u8,
    score: i32,
    bound: Bound,
    best_action: Option<Action>,
}

pub fn search_best_action(state: GameState, config: SearchConfig) -> SearchResult {
    search_best_action_with_root_order(state, config, &[])
}

/// Search for the best action while allowing an external policy/sorter to
/// provide a root ordering. The recursive alpha-beta evaluator remains the
/// authority; missing or illegal sorter actions fall back to Pathfinder's
/// deterministic heuristic ordering.
pub fn search_best_action_with_root_order(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
) -> SearchResult {
    search_best_action_with_root_order_and_options(state, config, root_order, false)
}

/// Search with an external root order and optional one-ply tactical extension
/// at the normal depth horizon. The extension is deliberately opt-in so the
/// incumbent Pathfinder remains an unchanged control in strength experiments.
pub fn search_best_action_with_root_order_and_options(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        root_order,
        tactical_extension,
        None,
        false,
    )
}

/// Search with a sorter-provided root order and an optional root candidate
/// limit. A limited root is useful for hybrid Pathfinder agents: the sorter
/// chooses a small candidate set, while the recursive alpha-beta evaluator
/// spends the entire node budget comparing those candidates. The ordinary
/// Pathfinder entry point leaves this unset so its baseline behavior is
/// unchanged.
pub fn search_best_action_with_root_order_and_root_limit(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        true,
    )
}

fn search_best_action_with_root_order_and_root_limit_internal(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    tt_move_order: bool,
) -> SearchResult {
    if state.config.board_size <= 4 {
        if let Some(horizon) = config.tactical_proof_horizon {
            let result = crate::endgame::search_best_action(
                state,
                crate::endgame::TacticalProofConfig {
                    horizon,
                    max_nodes: config.max_nodes,
                },
            );
            let score = match result.outcome {
                1 => WIN_SCORE - i32::from(state.ply),
                -1 => -WIN_SCORE + i32::from(state.ply),
                _ => 0,
            };
            return SearchResult {
                action: result.action,
                score,
                nodes: result.nodes,
                exhausted: result.exhausted,
                completed_depth: result.completed_depth,
                table_hits: result.table_hits,
            };
        }
    }
    let root_player = state.turn;
    let mut initial_actions = root_ordered_actions(state, root_player, config.weights, root_order);
    limit_root_actions(&mut initial_actions, root_limit);
    if initial_actions.is_empty() {
        return SearchResult {
            action: None,
            score: evaluate(state, root_player, config.weights),
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 0,
        };
    }

    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let mut best_action = initial_actions[0];
    let mut best_score = NEG_INF;
    let mut completed_depth = 0;

    for depth in 1..=config.depth {
        let mut actions = root_ordered_actions(state, root_player, config.weights, root_order);
        limit_root_actions(&mut actions, root_limit);
        put_first(&mut actions, best_action);
        let mut iteration_action = actions[0];
        let mut iteration_score = NEG_INF;
        let mut alpha = NEG_INF;
        let mut complete = true;

        for action in actions {
            if budget.nodes >= config.max_nodes {
                budget.exhausted = true;
                complete = false;
                break;
            }
            let next = state.apply_legal(action).state;
            budget.nodes += 1;
            let score = minimax(
                next,
                root_player,
                depth - 1,
                alpha,
                POS_INF,
                config,
                tactical_extension,
                tt_move_order,
                1,
                &mut budget,
                &mut table,
                &mut hints,
            );
            if score > iteration_score
                || (score == iteration_score && action.order() < iteration_action.order())
            {
                iteration_action = action;
                iteration_score = score;
            }
            alpha = alpha.max(iteration_score);
            if budget.exhausted {
                complete = false;
                break;
            }
        }
        if !complete {
            break;
        }
        best_action = iteration_action;
        best_score = iteration_score;
        completed_depth = depth;
    }

    if completed_depth == 0 {
        best_score = evaluate(
            state.apply_legal(best_action).state,
            root_player,
            config.weights,
        );
    }
    SearchResult {
        action: Some(best_action),
        score: best_score,
        nodes: budget.nodes,
        exhausted: budget.exhausted,
        completed_depth,
        table_hits: budget.table_hits,
    }
}

/// Expose Pathfinder's cheap deterministic root ordering to hybrid agents.
pub fn ordered_root_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    ordered_actions(state, root_player, weights)
}

/// Pathfinder's deterministic ordering with a bounded tactical guard.
///
/// The ordinary evaluator already puts captures and wins first, but it can
/// miss the quieter move that removes an opponent's immediate winning reply.
/// When the position is small enough to inspect cheaply, move such forced
/// blocks to the front of the root list. This is intentionally an ordering
/// hint only: the alpha-beta search still evaluates every candidate in its
/// configured beam and remains the authority on the result.
pub fn ordered_root_actions_with_tactical_guard(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    let fallback = ordered_actions(state, root_player, weights);
    if fallback.is_empty() || state.legal_action_count() > 512 {
        return fallback;
    }

    let opponent = root_player.other();
    let opponent_view = if state.turn == opponent {
        state
    } else {
        GameState {
            turn: opponent,
            ..state
        }
    };
    if immediate_winning_actions(opponent_view, opponent).is_empty() {
        return fallback;
    }

    let mut guarded = Vec::with_capacity(fallback.len());
    for action in fallback.iter().copied() {
        let next = state.apply_legal(action).state;
        let next_opponent_view = if next.turn == opponent {
            next
        } else {
            GameState {
                turn: opponent,
                ..next
            }
        };
        if immediate_winning_actions(next_opponent_view, opponent).is_empty() {
            guarded.push(action);
        }
    }
    if guarded.is_empty() {
        return fallback;
    }
    for action in fallback {
        if !guarded.contains(&action) {
            guarded.push(action);
        }
    }
    guarded
}

fn immediate_winning_actions(state: GameState, player: Player) -> Vec<Action> {
    if state.winner.is_some() || state.turn != player {
        return Vec::new();
    }
    state
        .legal_actions()
        .into_iter()
        .filter(|action| state.apply_legal(*action).state.winner == Some(player))
        .collect()
}

pub fn analyze_action(
    state: GameState,
    action: Action,
    config: SearchConfig,
) -> Result<MoveEvaluation, String> {
    if !state.legal_actions().contains(&action) {
        return Err("cannot analyze an illegal Pathagon action".to_owned());
    }
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let next = state.apply_legal(action).state;
    budget.nodes += 1;
    let score = if next.winner.is_some() {
        evaluate(next, root_player, config.weights)
    } else {
        minimax(
            next,
            root_player,
            config.depth.saturating_sub(1),
            NEG_INF,
            POS_INF,
            config,
            false,
            false,
            1,
            &mut budget,
            &mut table,
            &mut hints,
        )
    };
    Ok(MoveEvaluation {
        action,
        before_score,
        score,
        delta: score - before_score,
        nodes: budget.nodes,
        exhausted: budget.exhausted,
        completed_depth: config.depth,
        table_hits: budget.table_hits,
    })
}

pub fn analyze_actions(
    state: GameState,
    config: SearchConfig,
    max_actions: usize,
) -> Vec<MoveEvaluation> {
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let mut alpha = NEG_INF;
    let mut results = Vec::new();
    for action in ordered_actions(state, root_player, config.weights)
        .into_iter()
        .take(max_actions)
    {
        if budget.nodes >= config.max_nodes {
            budget.exhausted = true;
            break;
        }
        let next = state.apply_legal(action).state;
        budget.nodes += 1;
        let score = if next.winner.is_some() {
            evaluate(next, root_player, config.weights)
        } else {
            minimax(
                next,
                root_player,
                config.depth.saturating_sub(1),
                alpha,
                POS_INF,
                config,
                false,
                false,
                1,
                &mut budget,
                &mut table,
                &mut hints,
            )
        };
        results.push(MoveEvaluation {
            action,
            before_score,
            score,
            delta: score - before_score,
            nodes: budget.nodes,
            exhausted: budget.exhausted,
            completed_depth: config.depth,
            table_hits: budget.table_hits,
        });
        alpha = alpha.max(score);
        if budget.exhausted {
            break;
        }
    }
    results.sort_by_key(|result| (-result.score, result.action.order()));
    results
}

/// Choose the same shallow local-pattern move as the browser Lunatic baseline.
/// It deliberately evaluates each legal action once without considering the
/// opponent's reply, making it a cheap breadth opponent for large arenas.
pub fn lunatic_action(state: GameState) -> SearchResult {
    let actions = state.legal_actions();
    if actions.is_empty() {
        return SearchResult {
            action: None,
            score: evaluate(state, state.turn, EvaluationWeights::default()),
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 0,
        };
    }
    let player = state.turn;
    let before_own_distance = connection_distance(state, player);
    let before_opponent_distance = connection_distance(state, player.other());
    let mut best_action = actions[0];
    let mut best_score = NEG_INF;
    for action in actions.iter().copied() {
        let transition = state.apply_legal(action);
        let score = if transition.state.winner == Some(player) {
            WIN_SCORE
        } else {
            let own_distance = connection_distance(transition.state, player);
            let opponent_distance = connection_distance(transition.state, player.other());
            transition.captured.count_ones() as i32 * 10_000
                + (before_own_distance - own_distance) * 500
                + (opponent_distance - before_opponent_distance) * 350
                + if matches!(action, Action::Relocate { .. }) {
                    10
                } else {
                    0
                }
        };
        if score > best_score || (score == best_score && action.order() < best_action.order()) {
            best_action = action;
            best_score = score;
        }
    }
    SearchResult {
        action: Some(best_action),
        score: best_score,
        nodes: actions.len() as u64,
        exhausted: false,
        completed_depth: 1,
        table_hits: 0,
    }
}

pub fn evaluate(state: GameState, player: Player, weights: EvaluationWeights) -> i32 {
    if state.winner == Some(player) {
        return WIN_SCORE - state.ply as i32;
    }
    let opponent = player.other();
    if state.winner == Some(opponent) {
        return -WIN_SCORE + state.ply as i32;
    }
    let path = connection_distance(state, opponent) - connection_distance(state, player);
    let material =
        state.pieces(player).count_ones() as i32 - state.pieces(opponent).count_ones() as i32;
    let capture_direction = if state.last_player == Some(player) {
        1
    } else {
        -1
    };
    let structure = largest_component(state, player) - largest_component(state, opponent);
    let threats = capture_opportunities(state, player) - capture_opportunities(state, opponent);
    let edges = edge_presence(state, player) - edge_presence(state, opponent);
    path * weights.path
        + material * weights.material
        + capture_direction * state.last_capture as i32 * weights.capture
        + structure * weights.structure
        + threats * weights.threat
        + edges * weights.edge
}

fn minimax(
    state: GameState,
    root_player: Player,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    config: SearchConfig,
    tactical_extension: bool,
    tt_move_order: bool,
    ply_from_root: usize,
    budget: &mut Budget,
    table: &mut HashMap<(GameState, Player), TableEntry>,
    hints: &mut SearchHints,
) -> i32 {
    if state.winner.is_some() {
        return evaluate(state, root_player, config.weights);
    }
    if depth == 0 {
        if tactical_extension {
            if let Some(action) = state
                .legal_actions()
                .into_iter()
                .find(|action| state.apply_legal(*action).state.winner == Some(state.turn))
            {
                if budget.nodes < config.max_nodes {
                    budget.nodes += 1;
                    let score =
                        evaluate(state.apply_legal(action).state, root_player, config.weights);
                    if budget.nodes >= config.max_nodes {
                        budget.exhausted = true;
                    }
                    return score;
                }
            }
        }
        return evaluate(state, root_player, config.weights);
    }
    if budget.nodes >= config.max_nodes {
        budget.exhausted = true;
        return evaluate(state, root_player, config.weights);
    }

    let key = (state, root_player);
    let original_alpha = alpha;
    let original_beta = beta;
    let mut preferred_action = None;
    if let Some(entry) = table.get(&key).copied() {
        preferred_action = entry.best_action;
        if entry.depth >= depth {
            budget.table_hits += 1;
            match entry.bound {
                Bound::Exact => return entry.score,
                Bound::Lower => alpha = alpha.max(entry.score),
                Bound::Upper => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    let maximizing = state.turn == root_player;
    let mut actions = ordered_actions(state, root_player, config.weights);
    if tt_move_order {
        if let Some(preferred) = preferred_action {
            put_first(&mut actions, preferred);
        }
        let [killer_one, killer_two] = hints.killers_at(ply_from_root);
        if let Some(killer) = killer_two {
            put_first(&mut actions, killer);
        }
        if let Some(killer) = killer_one {
            put_first(&mut actions, killer);
        }
    }
    actions.truncate(config.beam_width);
    if actions.is_empty() {
        return evaluate(state, root_player, config.weights);
    }
    let mut best = if maximizing { NEG_INF } else { POS_INF };
    let mut best_action = actions[0];
    for action in actions {
        let next = state.apply_legal(action).state;
        budget.nodes += 1;
        let score = minimax(
            next,
            root_player,
            depth - 1,
            alpha,
            beta,
            config,
            tactical_extension,
            tt_move_order,
            ply_from_root + 1,
            budget,
            table,
            hints,
        );
        if maximizing {
            if score > best || (score == best && action.order() < best_action.order()) {
                best = score;
                best_action = action;
            }
            alpha = alpha.max(best);
        } else {
            if score < best || (score == best && action.order() < best_action.order()) {
                best = score;
                best_action = action;
            }
            beta = beta.min(best);
        }
        if beta <= alpha || budget.nodes >= config.max_nodes {
            if beta <= alpha && next.winner.is_none() && next.last_capture == 0 {
                hints.record_killer(ply_from_root, action);
            }
            break;
        }
    }
    if !budget.exhausted {
        let bound = if best <= original_alpha {
            Bound::Upper
        } else if best >= original_beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        table.insert(
            key,
            TableEntry {
                depth,
                score: best,
                bound,
                best_action: Some(best_action),
            },
        );
    }
    if budget.nodes >= config.max_nodes {
        budget.exhausted = true;
    }
    best
}

fn ordered_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    let maximizing = state.turn == root_player;
    let mut scored: Vec<(Action, i32)> = state
        .legal_actions()
        .into_iter()
        .map(|action| {
            let transition = state.apply_legal(action);
            let score = if transition.state.winner == Some(state.turn) {
                2_000_000_000
            } else {
                transition.captured.count_ones() as i32 * 10_000
                    + evaluate(transition.state, root_player, weights)
            };
            (action, score)
        })
        .collect();
    scored.sort_by(|left, right| {
        let score_order = if maximizing {
            right.1.cmp(&left.1)
        } else {
            left.1.cmp(&right.1)
        };
        score_order.then_with(|| left.0.order().cmp(&right.0.order()))
    });
    scored.into_iter().map(|(action, _)| action).collect()
}

fn root_ordered_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
    root_order: &[Action],
) -> Vec<Action> {
    let fallback = ordered_actions(state, root_player, weights);
    if root_order.is_empty() {
        return fallback;
    }
    let mut merged = Vec::with_capacity(fallback.len());
    for action in root_order.iter().copied() {
        if fallback.contains(&action) && !merged.contains(&action) {
            merged.push(action);
        }
    }
    for action in fallback {
        if !merged.contains(&action) {
            merged.push(action);
        }
    }
    merged
}

fn limit_root_actions(actions: &mut Vec<Action>, root_limit: Option<usize>) {
    if let Some(limit) = root_limit {
        actions.truncate(limit.max(1));
    }
}

pub(crate) fn connection_distance(state: GameState, player: Player) -> i32 {
    let opponent = player.other();
    let board_size = state.config.board_size;
    let cell_count = state.config.cells();
    let own_pieces = state.pieces(player);
    let opponent_pieces = state.pieces(opponent);
    let far_edge = if player == Player::Light {
        row_mask(board_size, 0)
    } else {
        column_mask(board_size, board_size - 1)
    };
    let mut distance = [u8::MAX; MAX_CELL_COUNT as usize];
    let mut buckets = [0_u64; MAX_CELL_COUNT as usize + 1];
    for index in 0..board_size {
        let square = if player == Player::Light {
            (board_size - 1) * board_size + index
        } else {
            index * board_size
        };
        let square_bit = bit(square);
        if opponent_pieces & square_bit != 0 {
            continue;
        }
        let cost = u8::from(own_pieces & square_bit == 0);
        distance[square as usize] = cost;
        buckets[cost as usize] |= square_bit;
    }
    let mut current_distance = 0_usize;
    while current_distance <= cell_count as usize {
        while buckets[current_distance] != 0 {
            let square = buckets[current_distance].trailing_zeros() as u8;
            buckets[current_distance] &= !bit(square);
            if bit(square) & far_edge != 0 {
                return current_distance as i32;
            }
            for next in squares(neighbor_mask_for(board_size, square)) {
                let next_bit = bit(next);
                if opponent_pieces & next_bit != 0 {
                    continue;
                }
                let step = u8::from(own_pieces & next_bit == 0);
                let next_distance = current_distance as u8 + step;
                if next_distance >= distance[next as usize] {
                    continue;
                }
                let previous_distance = distance[next as usize];
                if previous_distance != u8::MAX {
                    buckets[previous_distance as usize] &= !next_bit;
                }
                distance[next as usize] = next_distance;
                buckets[next_distance as usize] |= next_bit;
            }
        }
        current_distance += 1;
    }
    cell_count as i32
}

pub(crate) fn largest_component(state: GameState, player: Player) -> i32 {
    let mut remaining = state.pieces(player);
    let mut largest = 0;
    while let Some(first) = squares(remaining).next() {
        let mut stack = [0_u8; MAX_CELL_COUNT as usize];
        let mut stack_len = 1_usize;
        stack[0] = first;
        remaining &= !bit(first);
        let mut size = 0;
        while stack_len != 0 {
            stack_len -= 1;
            let square = stack[stack_len];
            size += 1;
            let adjacent = neighbor_mask_for(state.config.board_size, square) & remaining;
            for next in squares(adjacent) {
                remaining &= !bit(next);
                stack[stack_len] = next;
                stack_len += 1;
            }
        }
        largest = largest.max(size);
    }
    largest
}

pub(crate) fn capture_opportunities(state: GameState, player: Player) -> i32 {
    let occupied = state.light | state.dark | state.forbidden;
    let mut victims = 0;
    for origin in 0..state.config.cells() {
        if occupied & bit(origin) != 0 {
            continue;
        }
        victims |= captures_from(state, origin, player);
    }
    victims.count_ones() as i32
}

pub(crate) fn edge_presence(state: GameState, player: Player) -> i32 {
    let board_size = state.config.board_size;
    let mut near = false;
    let mut far = false;
    for index in 0..board_size {
        let near_square = if player == Player::Light {
            (board_size - 1) * board_size + index
        } else {
            index * board_size
        };
        let far_square = if player == Player::Light {
            index
        } else {
            index * board_size + board_size - 1
        };
        near |= state.board_at(near_square) == Some(player);
        far |= state.board_at(far_square) == Some(player);
    }
    i32::from(near) + i32::from(far)
}

fn put_first(actions: &mut Vec<Action>, preferred: Action) {
    if let Some(index) = actions.iter().position(|action| *action == preferred) {
        actions.swap(0, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn reference_connection_distance(state: GameState, player: Player) -> i32 {
        let opponent = player.other();
        let board_size = state.config.board_size;
        let cell_count = state.config.cells();
        let mut distance = vec![u8::MAX; cell_count as usize];
        let mut queue = VecDeque::new();
        for index in 0..board_size {
            let square = if player == Player::Light {
                (board_size - 1) * board_size + index
            } else {
                index * board_size
            };
            if state.board_at(square) == Some(opponent) {
                continue;
            }
            let cost = u8::from(state.board_at(square) != Some(player));
            distance[square as usize] = cost;
            if cost == 0 {
                queue.push_front(square);
            } else {
                queue.push_back(square);
            }
        }
        while let Some(square) = queue.pop_front() {
            let row = square / board_size;
            let column = square % board_size;
            if (player == Player::Light && row == 0)
                || (player == Player::Dark && column == board_size - 1)
            {
                return distance[square as usize] as i32;
            }
            for next in squares(neighbor_mask_for(board_size, square)) {
                if state.board_at(next) == Some(opponent) {
                    continue;
                }
                let step = u8::from(state.board_at(next) != Some(player));
                let next_distance = distance[square as usize].saturating_add(step);
                if next_distance >= distance[next as usize] {
                    continue;
                }
                distance[next as usize] = next_distance;
                if step == 0 {
                    queue.push_front(next);
                } else {
                    queue.push_back(next);
                }
            }
        }
        cell_count as i32
    }

    #[test]
    fn allocation_free_connection_distance_matches_reference_search() {
        for board_size in crate::MIN_BOARD_SIZE..=crate::MAX_BOARD_SIZE {
            let mut state = GameState::with_board_size(board_size);
            let mut seed = u32::from(board_size).wrapping_mul(0x243f_6a88);
            for _ in 0..96 {
                for player in [Player::Light, Player::Dark] {
                    assert_eq!(
                        connection_distance(state, player),
                        reference_connection_distance(state, player),
                        "connection distance mismatch on {board_size}x{board_size} for {}",
                        player.as_str()
                    );
                }
                let actions = state.legal_actions();
                let Some(action) = actions.get((seed as usize) % actions.len().max(1)).copied()
                else {
                    break;
                };
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state = state.apply_legal(action).state;
                if state.winner.is_some() {
                    break;
                }
            }
        }
    }

    #[test]
    fn iterative_search_respects_budget() {
        let result = search_best_action(
            GameState::new(),
            SearchConfig {
                depth: 5,
                max_nodes: 120,
                beam_width: 49,
                ..SearchConfig::default()
            },
        );
        assert!(result.action.is_some());
        assert!(result.nodes <= 120);
        assert!(result.exhausted);
        assert!((1..5).contains(&result.completed_depth));
    }

    #[test]
    fn external_root_order_is_respected_under_a_tiny_budget() {
        let state = GameState::new();
        let preferred = state
            .legal_actions()
            .last()
            .copied()
            .expect("opening position has legal actions");
        let result = search_best_action_with_root_order(
            state,
            SearchConfig {
                depth: 1,
                max_nodes: 1,
                beam_width: 1,
                ..SearchConfig::default()
            },
            &[preferred],
        );
        assert_eq!(result.action, Some(preferred));
        assert_eq!(result.nodes, 1);
        assert!(result.exhausted);
    }

    #[test]
    fn root_candidate_limit_restricts_hybrid_search_to_sorted_pool() {
        let state = GameState::new();
        let preferred = state
            .legal_actions()
            .last()
            .copied()
            .expect("opening position has legal actions");
        let result = search_best_action_with_root_order_and_root_limit(
            state,
            SearchConfig {
                depth: 1,
                max_nodes: 100,
                beam_width: 8,
                ..SearchConfig::default()
            },
            &[preferred],
            false,
            Some(1),
        );
        assert_eq!(result.action, Some(preferred));
        assert!(!result.exhausted);
    }

    #[test]
    fn tactical_root_guard_preserves_legality_and_fallback_order() {
        let state = GameState::new();
        let guarded = ordered_root_actions_with_tactical_guard(
            state,
            state.turn,
            EvaluationWeights::default(),
        );
        assert_eq!(guarded.len(), state.legal_actions().len());
        assert!(guarded
            .iter()
            .all(|action| state.legal_actions().contains(action)));
        assert_eq!(
            guarded
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            state.legal_actions().into_iter().collect()
        );
    }

    #[test]
    fn lunatic_returns_a_legal_action() {
        let state = GameState::new();
        let result = lunatic_action(state);
        assert!(result.action.is_some());
        assert!(state.legal_actions().contains(&result.action.unwrap()));
        assert_eq!(result.completed_depth, 1);
    }

    #[test]
    fn search_supports_variable_board_sizes() {
        for size in 4..=7 {
            let state = GameState::with_board_size(size);
            let result = search_best_action(
                state,
                SearchConfig {
                    depth: 1,
                    max_nodes: 256,
                    beam_width: 64,
                    ..SearchConfig::default()
                },
            );
            assert!(
                result.action.is_some(),
                "{size}x{size} search returned no action"
            );
            assert!(
                state.legal_actions().contains(&result.action.unwrap()),
                "{size}x{size} search returned an illegal action"
            );
        }
    }

    #[test]
    fn optional_tactical_proof_mode_selects_a_forced_block() {
        let config = crate::BoardConfig::new(4, 5)
            .expect("valid board config")
            .with_max_plies(64)
            .expect("valid ply limit");
        let state = GameState {
            config,
            light: [5_u8, 7, 9, 11, 15]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            dark: [1_u8, 2, 3, 6, 10]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        };
        let result = search_best_action(
            state,
            SearchConfig {
                tactical_proof_horizon: Some(3),
                max_nodes: 100_000,
                ..SearchConfig::default()
            },
        );
        assert!([5_u8, 7, 9, 11, 15]
            .into_iter()
            .map(|from| Action::Relocate { from, to: 0 })
            .any(|action| result.action == Some(action)));
        assert_eq!(result.score, 0);
        assert_eq!(result.completed_depth, 3);
        assert!(!result.exhausted);
    }
}
