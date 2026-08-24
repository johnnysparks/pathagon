use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

use crate::{bit, neighbor_mask_for, squares, Action, GameState, Player};

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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            depth: 4,
            max_nodes: 90_000,
            beam_width: 40,
            weights: EvaluationWeights::default(),
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
}

pub fn search_best_action(state: GameState, config: SearchConfig) -> SearchResult {
    let root_player = state.turn;
    let initial_actions = ordered_actions(state, root_player, config.weights);
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
    let mut best_action = initial_actions[0];
    let mut best_score = NEG_INF;
    let mut completed_depth = 0;

    for depth in 1..=config.depth {
        let mut actions = ordered_actions(state, root_player, config.weights);
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
                &mut budget,
                &mut table,
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
        best_score = evaluate(state.apply_legal(best_action).state, root_player, config.weights);
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

pub fn analyze_action(state: GameState, action: Action, config: SearchConfig) -> Result<MoveEvaluation, String> {
    if !state.legal_actions().contains(&action) {
        return Err("cannot analyze an illegal Pathagon action".to_owned());
    }
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
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
            &mut budget,
            &mut table,
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

pub fn analyze_actions(state: GameState, config: SearchConfig, max_actions: usize) -> Vec<MoveEvaluation> {
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut alpha = NEG_INF;
    let mut results = Vec::new();
    for action in ordered_actions(state, root_player, config.weights).into_iter().take(max_actions) {
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
                &mut budget,
                &mut table,
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
                + if matches!(action, Action::Relocate { .. }) { 10 } else { 0 }
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
    let material = state.pieces(player).count_ones() as i32 - state.pieces(opponent).count_ones() as i32;
    let capture_direction = if state.last_player == Some(player) { 1 } else { -1 };
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
    budget: &mut Budget,
    table: &mut HashMap<(GameState, Player), TableEntry>,
) -> i32 {
    if state.winner.is_some() || depth == 0 {
        return evaluate(state, root_player, config.weights);
    }
    if budget.nodes >= config.max_nodes {
        budget.exhausted = true;
        return evaluate(state, root_player, config.weights);
    }

    let key = (state, root_player);
    let original_alpha = alpha;
    let original_beta = beta;
    if let Some(entry) = table.get(&key).copied().filter(|entry| entry.depth >= depth) {
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

    let maximizing = state.turn == root_player;
    let mut actions = ordered_actions(state, root_player, config.weights);
    actions.truncate(config.beam_width);
    if actions.is_empty() {
        return evaluate(state, root_player, config.weights);
    }
    let mut best = if maximizing { NEG_INF } else { POS_INF };
    for action in actions {
        let next = state.apply_legal(action).state;
        budget.nodes += 1;
        let score = minimax(next, root_player, depth - 1, alpha, beta, config, budget, table);
        if maximizing {
            best = best.max(score);
            alpha = alpha.max(best);
        } else {
            best = best.min(score);
            beta = beta.min(best);
        }
        if beta <= alpha || budget.nodes >= config.max_nodes {
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
        table.insert(key, TableEntry { depth, score: best, bound });
    }
    if budget.nodes >= config.max_nodes {
        budget.exhausted = true;
    }
    best
}

fn ordered_actions(state: GameState, root_player: Player, weights: EvaluationWeights) -> Vec<Action> {
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

fn connection_distance(state: GameState, player: Player) -> i32 {
    let opponent = player.other();
    let board_size = state.config.board_size;
    let cell_count = state.config.cells();
    let mut distance = vec![u8::MAX; cell_count as usize];
    let mut queue = VecDeque::new();
    for index in 0..board_size {
        let square = if player == Player::Light { (board_size - 1) * board_size + index } else { index * board_size };
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
        if (player == Player::Light && row == 0) || (player == Player::Dark && column == board_size - 1) {
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

fn largest_component(state: GameState, player: Player) -> i32 {
    let mut remaining = state.pieces(player);
    let mut largest = 0;
    while let Some(first) = squares(remaining).next() {
        let mut stack = vec![first];
        remaining &= !bit(first);
        let mut size = 0;
        while let Some(square) = stack.pop() {
            size += 1;
            let adjacent = neighbor_mask_for(state.config.board_size, square) & remaining;
            for next in squares(adjacent) {
                remaining &= !bit(next);
                stack.push(next);
            }
        }
        largest = largest.max(size);
    }
    largest
}

fn capture_opportunities(state: GameState, player: Player) -> i32 {
    let occupied = state.light | state.dark | state.forbidden;
    let board_size = state.config.board_size;
    let mut victims = 0;
    for origin in 0..state.config.cells() {
        if occupied & bit(origin) != 0 {
            continue;
        }
        let row = (origin / board_size) as i8;
        let column = (origin % board_size) as i8;
        for (row_delta, column_delta) in [(-1_i8, 0_i8), (1, 0), (0, -1), (0, 1)] {
            let far_row = row + row_delta * 2;
            let far_column = column + column_delta * 2;
            if !(0..board_size as i8).contains(&far_row) || !(0..board_size as i8).contains(&far_column) {
                continue;
            }
            let near = ((row + row_delta) * board_size as i8 + column + column_delta) as u8;
            let far = (far_row * board_size as i8 + far_column) as u8;
            if state.board_at(near) == Some(player.other()) && state.board_at(far) == Some(player) {
                victims |= bit(near);
            }
        }
    }
    victims.count_ones() as i32
}

fn edge_presence(state: GameState, player: Player) -> i32 {
    let board_size = state.config.board_size;
    let mut near = false;
    let mut far = false;
    for index in 0..board_size {
        let near_square = if player == Player::Light { (board_size - 1) * board_size + index } else { index * board_size };
        let far_square = if player == Player::Light { index } else { index * board_size + board_size - 1 };
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

    #[test]
    fn iterative_search_respects_budget() {
        let result = search_best_action(
            GameState::new(),
            SearchConfig { depth: 5, max_nodes: 120, beam_width: 49, ..SearchConfig::default() },
        );
        assert!(result.action.is_some());
        assert!(result.nodes <= 120);
        assert!(result.exhausted);
        assert!((1..5).contains(&result.completed_depth));
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
                SearchConfig { depth: 1, max_nodes: 256, beam_width: 64, ..SearchConfig::default() },
            );
            assert!(result.action.is_some(), "{size}x{size} search returned no action");
            assert!(state.legal_actions().contains(&result.action.unwrap()), "{size}x{size} search returned an illegal action");
        }
    }
}
