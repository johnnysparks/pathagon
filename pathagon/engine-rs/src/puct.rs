//! Policy/value guided PUCT search for the fixed 7x7 deployment model.
//!
//! The rules engine still owns action generation and state transitions. The
//! learned model supplies priors and a leaf value; this module only combines
//! those signals into a deterministic tree policy that can run natively or in
//! the inference-enabled WASM build.

use crate::inference::{PolicyValue, PolicyValueModel};
use crate::search::{evaluate, EvaluationWeights};
use crate::{Action, GameState};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
type Deadline = f64;

#[cfg(not(target_arch = "wasm32"))]
type Deadline = Instant;

#[cfg(target_arch = "wasm32")]
fn deadline_after_ms(milliseconds: u32) -> Deadline {
    js_sys::Date::now() + f64::from(milliseconds)
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_after_ms(milliseconds: u32) -> Deadline {
    Instant::now() + Duration::from_millis(u64::from(milliseconds))
}

#[cfg(target_arch = "wasm32")]
fn deadline_reached(deadline: Deadline) -> bool {
    js_sys::Date::now() >= deadline
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_reached(deadline: Deadline) -> bool {
    Instant::now() >= deadline
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctConfig {
    pub simulations: u32,
    pub cpuct: f32,
    /// Ask models with an action-value head (QAdv) to seed every expanded
    /// node. Disabled by default so ordinary policy/value search keeps its
    /// previous inference cost and behavior.
    pub use_action_value_seeds: bool,
    /// Blend Q/Advantage action-value seeds into tree selection. Zero keeps
    /// the ordinary policy/value behavior; one uses the full Q seed.
    pub qadv_weight: f32,
    /// Hard cap on the number of materialized tree positions, including the
    /// root. `u64::MAX` disables the cap for callers that only use simulations.
    pub max_nodes: u64,
    /// Wall-clock ceiling for one search. Zero disables the ceiling.
    pub max_time_ms: u32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            simulations: 64,
            cpuct: 1.5,
            use_action_value_seeds: false,
            qadv_weight: 1.0,
            max_nodes: u64::MAX,
            max_time_ms: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctActionEvaluation {
    pub action: Action,
    pub prior: f32,
    pub visits: u32,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PuctResult {
    pub action: Option<Action>,
    pub value: f32,
    pub simulations: u32,
    pub nodes: u64,
    pub evaluations: Vec<PuctActionEvaluation>,
}

impl PuctResult {
    /// Export evaluations in the same legal-action order used by the archive
    /// contract. Values are already from the root player's perspective.
    pub fn root_q_targets(&self) -> Result<crate::contract::RootQTargets, String> {
        crate::contract::RootQTargets::new(
            self.evaluations
                .iter()
                .map(|evaluation| evaluation.value)
                .collect(),
            self.evaluations
                .iter()
                .map(|evaluation| evaluation.visits)
                .collect(),
        )
    }
}

struct Node {
    state: GameState,
    actions: Vec<Action>,
    priors: Vec<f32>,
    children: Vec<Option<usize>>,
    child_seeds: Vec<Option<f32>>,
    visits: u32,
    value_sum: f32,
    expanded: bool,
}

impl Node {
    fn new(state: GameState) -> Self {
        Self {
            state,
            actions: Vec::new(),
            priors: Vec::new(),
            children: Vec::new(),
            child_seeds: Vec::new(),
            visits: 0,
            value_sum: 0.0,
            expanded: false,
        }
    }

    fn mean_value(self: &Self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn estimated_value(self: &Self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.mean_value()
        }
    }

    fn expand<M: PolicyValueModel>(
        &mut self,
        model: &M,
        root_output: Option<PolicyValue>,
        root_action_values: Option<Vec<f32>>,
        actions: Option<Vec<Action>>,
        use_action_value_seeds: bool,
        qadv_weight: f32,
    ) -> Result<f32, String> {
        if self.state.winner.is_some() || self.state.ply >= self.state.config.max_plies {
            self.expanded = true;
            return Ok(terminal_value(self.state));
        }
        self.actions = actions.unwrap_or_else(|| self.state.legal_actions());
        if self.actions.is_empty() {
            self.expanded = true;
            return Ok(0.0);
        }
        let (output, action_values) = match root_output {
            Some(output) => (output, root_action_values),
            None if use_action_value_seeds => model
                .evaluate_policy_value_and_action_values_with_actions(self.state, &self.actions)?,
            None => (
                model.evaluate_policy_value_with_actions(self.state, &self.actions)?,
                None,
            ),
        };
        if output.policy_logits.len() < self.actions.len() {
            return Err(format!(
                "policy/value model returned {} logits for {} legal actions",
                output.policy_logits.len(),
                self.actions.len()
            ));
        }
        self.priors = softmax(&output.policy_logits[..self.actions.len()]);
        self.children = (0..self.actions.len()).map(|_| None).collect();
        self.child_seeds = (0..self.actions.len()).map(|_| None).collect();
        if let Some(action_values) = action_values {
            if action_values.len() < self.actions.len() {
                return Err(format!(
                    "action-value model returned {} values for {} legal actions",
                    action_values.len(),
                    self.actions.len()
                ));
            }
            // QAdv values are from the side-to-move perspective at this node.
            // Children store values from their own side-to-move perspective,
            // so negate the parent action value exactly as for root seeds.
            for (index, value) in action_values.iter().take(self.actions.len()).enumerate() {
                if qadv_weight > 0.0 {
                    self.child_seeds[index] =
                        Some(-(value.clamp(-1.0, 1.0) * qadv_weight.clamp(0.0, 1.0)));
                }
            }
        }
        self.expanded = true;
        Ok(output.value.clamp(-1.0, 1.0))
    }

    fn select_action(&self, nodes: &[Node], cpuct: f32) -> usize {
        let parent_scale = (self.visits.max(1) as f32).sqrt();
        let mut best_index = 0;
        let mut best_score = f32::NEG_INFINITY;
        for index in 0..self.actions.len() {
            let (child_visits, child_value) = self.children[index].map_or_else(
                || (0, -self.child_seeds[index].unwrap_or(0.0)),
                |child_index| {
                    let child = &nodes[child_index];
                    (child.visits, -child.estimated_value())
                },
            );
            let score = child_value
                + cpuct * self.priors[index] * parent_scale / (1.0 + child_visits as f32);
            if score > best_score
                || (score == best_score
                    && self.actions[index].order() < self.actions[best_index].order())
            {
                best_index = index;
                best_score = score;
            }
        }
        best_index
    }
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, state: GameState) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node::new(state));
        index
    }
}

pub fn search<M: PolicyValueModel>(
    model: &M,
    state: GameState,
    config: PuctConfig,
) -> Result<PuctResult, String> {
    search_with_root_output(model, state, config, None)
}

/// Run PUCT with an already-computed root policy/value output.
///
/// Q/Advantage self-play already evaluates the root to obtain its action-value
/// head. Reusing that policy/value pair avoids a second full ONNX invocation
/// for the same state. When `PuctConfig::use_action_value_seeds` is enabled,
/// the model's action values also seed unvisited children at deeper nodes.
pub fn search_with_root_output<M: PolicyValueModel>(
    model: &M,
    state: GameState,
    config: PuctConfig,
    root_output: Option<PolicyValue>,
) -> Result<PuctResult, String> {
    search_with_root_output_and_seeds(model, state, config, root_output, None)
}

/// Run PUCT with optional root action-value seeds in canonical action order.
/// A QAdv root can supply its already-computed Q vector here, avoiding the
/// eager heuristic afterstate scan that generic policy/value search uses.
pub fn search_with_root_output_and_seeds<M: PolicyValueModel>(
    model: &M,
    state: GameState,
    config: PuctConfig,
    root_output: Option<PolicyValue>,
    root_seeds: Option<Vec<f32>>,
) -> Result<PuctResult, String> {
    search_with_root_output_and_seeds_and_actions(
        model,
        state,
        config,
        root_output,
        root_seeds,
        None,
    )
}

/// Variant of seeded PUCT for callers that already own the canonical root
/// action list. The list is moved into the root node, avoiding one more legal
/// move generation pass and preserving its exact order.
pub fn search_with_root_output_and_seeds_and_actions<M: PolicyValueModel>(
    model: &M,
    state: GameState,
    config: PuctConfig,
    root_output: Option<PolicyValue>,
    root_seeds: Option<Vec<f32>>,
    root_actions: Option<Vec<Action>>,
) -> Result<PuctResult, String> {
    let deadline = (config.max_time_ms > 0).then(|| deadline_after_ms(config.max_time_ms));
    let root_capacity = config.simulations as usize + crate::MAX_CELL_COUNT as usize + 1;
    let mut tree = Tree::with_capacity(root_capacity);
    let root_index = tree.push(state);
    let root_value = tree.nodes[root_index].expand(
        model,
        root_output,
        None,
        root_actions,
        config.use_action_value_seeds,
        config.qadv_weight,
    )?;
    if tree.nodes[root_index].actions.is_empty() {
        return Ok(PuctResult {
            action: None,
            value: root_value,
            simulations: 0,
            nodes: tree.nodes.len() as u64,
            evaluations: Vec::new(),
        });
    }
    seed_root_afterstates(&mut tree, root_index, root_seeds.as_deref());

    let mut completed_simulations = 0;
    for _ in 0..config.simulations {
        if tree.nodes.len() as u64 >= config.max_nodes || deadline.is_some_and(deadline_reached) {
            break;
        }
        simulate(
            &mut tree,
            root_index,
            model,
            config.cpuct,
            config.use_action_value_seeds,
            config.qadv_weight,
            config.max_nodes,
            deadline,
        )?;
        completed_simulations += 1;
    }

    let root = &tree.nodes[root_index];
    let evaluations = root
        .actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let child = root.children[index].map(|child_index| &tree.nodes[child_index]);
            PuctActionEvaluation {
                action: *action,
                prior: root.priors[index],
                visits: child.map_or(0, |node| node.visits),
                value: child.map_or_else(
                    || -root.child_seeds[index].unwrap_or(0.0),
                    |node| -node.estimated_value(),
                ),
            }
        })
        .collect::<Vec<_>>();
    let action = evaluations
        .iter()
        .max_by(|left, right| {
            left.visits
                .cmp(&right.visits)
                .then_with(|| right.action.order().cmp(&left.action.order()))
        })
        .map(|evaluation| evaluation.action);
    Ok(PuctResult {
        action,
        value: root.mean_value(),
        simulations: completed_simulations,
        nodes: tree.nodes.len() as u64,
        evaluations,
    })
}

fn seed_root_afterstates(tree: &mut Tree, root_index: usize, root_seeds: Option<&[f32]>) {
    if let Some(root_seeds) =
        root_seeds.filter(|seeds| seeds.len() >= tree.nodes[root_index].actions.len())
    {
        for index in 0..tree.nodes[root_index].actions.len() {
            // The supplied values are from the root player's perspective;
            // stored child estimates use the side-to-move perspective.
            tree.nodes[root_index].child_seeds[index] = Some(-root_seeds[index].clamp(-1.0, 1.0));
        }
        return;
    }
    // A QAdv model may have supplied seeds while expanding the root. Preserve
    // those values; only the plain policy/value path needs the legacy
    // heuristic afterstate fallback.
    if tree.nodes[root_index]
        .child_seeds
        .iter()
        .any(Option::is_some)
    {
        return;
    }
    let root_player = tree.nodes[root_index].state.turn;
    let action_count = tree.nodes[root_index].actions.len();
    for index in 0..action_count {
        let action = tree.nodes[root_index].actions[index];
        let next_state = tree.nodes[root_index].state.apply_legal(action).state;
        let root_value = if next_state.winner.is_some() {
            -terminal_value(next_state)
        } else if next_state.ply >= next_state.config.max_plies {
            0.0
        } else {
            let score = evaluate(next_state, root_player, EvaluationWeights::default()) as f32;
            (score / 3_500.0).tanh()
        };
        // Store values from the child side to move; selection and result
        // export consume them with a sign flip. The actual child node is
        // created lazily when PUCT selects this action.
        tree.nodes[root_index].child_seeds[index] = Some(-root_value);
    }
}

fn simulate<M: PolicyValueModel>(
    tree: &mut Tree,
    node_index: usize,
    model: &M,
    cpuct: f32,
    use_action_value_seeds: bool,
    qadv_weight: f32,
    max_nodes: u64,
    deadline: Option<Deadline>,
) -> Result<f32, String> {
    if deadline.is_some_and(deadline_reached) {
        return Ok(0.0);
    }
    if !tree.nodes[node_index].expanded {
        let value = tree.nodes[node_index].expand(
            model,
            None,
            None,
            None,
            use_action_value_seeds,
            qadv_weight,
        )?;
        tree.nodes[node_index].visits += 1;
        tree.nodes[node_index].value_sum += value;
        return Ok(value);
    }
    if tree.nodes[node_index].state.winner.is_some()
        || tree.nodes[node_index].state.ply >= tree.nodes[node_index].state.config.max_plies
        || tree.nodes[node_index].actions.is_empty()
    {
        let value = terminal_value(tree.nodes[node_index].state);
        tree.nodes[node_index].visits += 1;
        tree.nodes[node_index].value_sum += value;
        return Ok(value);
    }

    let index = tree.nodes[node_index].select_action(&tree.nodes, cpuct);
    let child_index = if let Some(child_index) = tree.nodes[node_index].children[index] {
        child_index
    } else {
        let next_state = tree.nodes[node_index]
            .state
            .apply_legal(tree.nodes[node_index].actions[index])
            .state;
        if tree.nodes.len() as u64 >= max_nodes || deadline.is_some_and(deadline_reached) {
            return Ok(0.0);
        }
        let child_index = tree.push(next_state);
        tree.nodes[node_index].children[index] = Some(child_index);
        child_index
    };
    let child_value = simulate(
        tree,
        child_index,
        model,
        cpuct,
        use_action_value_seeds,
        qadv_weight,
        max_nodes,
        deadline,
    )?;
    let value = -child_value;
    tree.nodes[node_index].visits += 1;
    tree.nodes[node_index].value_sum += value;
    Ok(value)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f32>();
    if total.is_finite() && total > 0.0 {
        for probability in &mut probabilities {
            *probability /= total;
        }
    } else {
        let uniform = 1.0 / probabilities.len().max(1) as f32;
        probabilities.fill(uniform);
    }
    probabilities
}

fn terminal_value(state: GameState) -> f32 {
    match state.winner {
        Some(winner) if winner == state.turn => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{PolicyValue, PolicyValueModel};

    struct TestModel;

    impl PolicyValueModel for TestModel {
        fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
            let actions = state.legal_actions();
            let mut policy_logits = vec![0.0; crate::model::MAX_ACTIONS];
            for (index, action) in actions.iter().enumerate() {
                policy_logits[index] = -(action.order() as f32);
            }
            Ok(PolicyValue {
                policy_logits,
                value: 0.0,
            })
        }
    }

    #[test]
    fn puct_returns_a_legal_action_and_accounts_for_simulations() {
        let result = search(
            &TestModel,
            GameState::new(),
            PuctConfig {
                simulations: 12,
                cpuct: 1.5,
                use_action_value_seeds: false,
                qadv_weight: 1.0,
                max_nodes: u64::MAX,
                max_time_ms: 0,
            },
        )
        .expect("run PUCT");
        assert!(result.action.is_some());
        assert_eq!(result.simulations, 12);
        assert_eq!(result.evaluations.len(), 49);
        assert_eq!(
            result
                .evaluations
                .iter()
                .map(|item| item.visits)
                .sum::<u32>(),
            12
        );
        let targets = result.root_q_targets().expect("export root-Q targets");
        assert_eq!(targets.action_values.len(), result.evaluations.len());
        assert_eq!(targets.action_visits.iter().sum::<u32>(), 12);
        assert_eq!(targets.action_values[0], result.evaluations[0].value);
        assert!(GameState::new()
            .legal_actions()
            .contains(&result.action.unwrap()));
    }

    #[test]
    fn puct_stops_after_the_materialized_node_budget() {
        let result = search(
            &TestModel,
            GameState::new(),
            PuctConfig {
                simulations: 12,
                cpuct: 1.5,
                use_action_value_seeds: false,
                qadv_weight: 1.0,
                max_nodes: 2,
                max_time_ms: 0,
            },
        )
        .expect("run node-capped PUCT");
        assert_eq!(result.simulations, 1);
        assert_eq!(
            result
                .evaluations
                .iter()
                .map(|evaluation| evaluation.visits)
                .sum::<u32>(),
            1
        );
    }

    #[test]
    fn root_afterstate_scan_seeds_every_legal_child() {
        let mut tree = Tree::with_capacity(50);
        let root_index = tree.push(GameState::new());
        tree.nodes[root_index]
            .expand(&TestModel, None, None, None, false, 1.0)
            .expect("expand root");
        let action_count = tree.nodes[root_index].actions.len();

        seed_root_afterstates(&mut tree, root_index, None);

        assert_eq!(tree.nodes[root_index].children.len(), action_count);
        assert!(tree.nodes[root_index]
            .children
            .iter()
            .all(|child| child.is_none()));
        assert!(tree.nodes[root_index]
            .child_seeds
            .iter()
            .all(|seed| seed.is_some()));
        assert!(tree.nodes[root_index]
            .children
            .iter()
            .all(|child| child.is_none()));
    }

    #[test]
    fn puct_treats_the_configured_ply_cap_as_a_draw_terminal() {
        let mut state = GameState::new();
        state.ply = state.config.max_plies;
        let result = search(
            &TestModel,
            state,
            PuctConfig {
                simulations: 12,
                cpuct: 1.5,
                use_action_value_seeds: false,
                qadv_weight: 1.0,
                max_nodes: u64::MAX,
                max_time_ms: 0,
            },
        )
        .expect("run capped PUCT");
        assert_eq!(result.action, None);
        assert_eq!(result.value, 0.0);
        assert_eq!(result.simulations, 0);
        assert!(result.evaluations.is_empty());
    }

    #[test]
    fn root_action_value_seeds_preserve_root_perspective_without_children() {
        let state = GameState::new();
        let actions = state.legal_actions();
        let seeds = actions
            .iter()
            .enumerate()
            .map(|(index, _)| index as f32 / actions.len() as f32)
            .collect::<Vec<_>>();
        let result = search_with_root_output_and_seeds(
            &TestModel,
            state,
            PuctConfig {
                simulations: 0,
                cpuct: 1.5,
                use_action_value_seeds: false,
                qadv_weight: 1.0,
                max_nodes: u64::MAX,
                max_time_ms: 0,
            },
            Some(PolicyValue {
                policy_logits: vec![0.0; crate::model::MAX_ACTIONS],
                value: 0.0,
            }),
            Some(seeds.clone()),
        )
        .expect("run seeded PUCT");
        assert_eq!(
            result
                .evaluations
                .iter()
                .map(|evaluation| evaluation.value)
                .collect::<Vec<_>>(),
            seeds
        );
        assert_eq!(
            result
                .evaluations
                .iter()
                .map(|item| item.visits)
                .sum::<u32>(),
            0
        );
    }

    struct QAdvTestModel;

    impl PolicyValueModel for QAdvTestModel {
        fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
            let actions = state.legal_actions();
            Ok(PolicyValue {
                policy_logits: vec![0.0; actions.len()],
                value: 0.0,
            })
        }

        fn evaluate_policy_value_and_action_values_with_actions(
            &self,
            _state: GameState,
            actions: &[Action],
        ) -> Result<(PolicyValue, Option<Vec<f32>>), String> {
            Ok((
                PolicyValue {
                    policy_logits: vec![0.0; actions.len()],
                    value: 0.0,
                },
                Some(
                    actions
                        .iter()
                        .enumerate()
                        .map(|(index, _)| if index == 0 { 1.0 } else { -1.0 })
                        .collect(),
                ),
            ))
        }
    }

    #[test]
    fn qadv_values_seed_unvisited_nodes_beyond_the_root() {
        let model = QAdvTestModel;
        let mut tree = Tree::with_capacity(64);
        let root_index = tree.push(GameState::new());
        tree.nodes[root_index]
            .expand(&model, None, None, None, true, 1.0)
            .expect("expand root");
        seed_root_afterstates(&mut tree, root_index, None);

        // The first root action has the only positive QAdv seed and should be
        // selected before the remaining unvisited actions.
        simulate(
            &mut tree,
            root_index,
            &model,
            1.5,
            true,
            1.0,
            u64::MAX,
            None,
        )
        .expect("simulate");
        assert!(tree.nodes[root_index].children[0].is_some());

        let child_index = tree.nodes[root_index].children[0].expect("first child");
        assert!(tree.nodes[child_index]
            .child_seeds
            .iter()
            .all(Option::is_some));
        assert_eq!(tree.nodes[child_index].child_seeds[0], Some(-1.0));
        assert_eq!(tree.nodes[child_index].select_action(&tree.nodes, 1.5), 0);
    }
}
