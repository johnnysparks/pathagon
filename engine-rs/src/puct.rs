//! Policy/value guided PUCT search for the fixed 7x7 deployment model.
//!
//! The rules engine still owns action generation and state transitions. The
//! learned model supplies priors and a leaf value; this module only combines
//! those signals into a deterministic tree policy that can run natively or in
//! the inference-enabled WASM build.

use crate::inference::PolicyValueModel;
use crate::search::{evaluate, EvaluationWeights};
use crate::{Action, GameState};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctConfig {
    pub simulations: u32,
    pub cpuct: f32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            simulations: 64,
            cpuct: 1.5,
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
    pub evaluations: Vec<PuctActionEvaluation>,
}

struct Node {
    state: GameState,
    actions: Vec<Action>,
    priors: Vec<f32>,
    children: Vec<Option<Box<Node>>>,
    seeded_value: Option<f32>,
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
            seeded_value: None,
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
            self.seeded_value.unwrap_or(0.0)
        } else {
            self.mean_value()
        }
    }

    fn expand<M: PolicyValueModel>(&mut self, model: &M) -> Result<f32, String> {
        if self.state.winner.is_some() || self.state.ply >= self.state.config.max_plies {
            self.expanded = true;
            return Ok(terminal_value(self.state));
        }
        self.actions = self.state.legal_actions();
        if self.actions.is_empty() {
            self.expanded = true;
            return Ok(0.0);
        }
        let output = model.evaluate(self.state)?;
        if output.policy_logits.len() < self.actions.len() {
            return Err(format!(
                "policy/value model returned {} logits for {} legal actions",
                output.policy_logits.len(),
                self.actions.len()
            ));
        }
        self.priors = softmax(&output.policy_logits[..self.actions.len()]);
        self.children = (0..self.actions.len()).map(|_| None).collect();
        self.expanded = true;
        Ok(output.value.clamp(-1.0, 1.0))
    }

    fn select_action(&self, cpuct: f32) -> usize {
        let parent_scale = (self.visits.max(1) as f32).sqrt();
        let mut best_index = 0;
        let mut best_score = f32::NEG_INFINITY;
        for index in 0..self.actions.len() {
            let (child_visits, child_value) = self.children[index]
                .as_deref()
                .map_or((0, 0.0), |child| (child.visits, -child.estimated_value()));
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

pub fn search<M: PolicyValueModel>(
    model: &M,
    state: GameState,
    config: PuctConfig,
) -> Result<PuctResult, String> {
    let mut root = Node::new(state);
    let root_value = root.expand(model)?;
    if root.actions.is_empty() {
        return Ok(PuctResult {
            action: None,
            value: root_value,
            simulations: 0,
            evaluations: Vec::new(),
        });
    }
    seed_root_afterstates(&mut root);

    for _ in 0..config.simulations {
        simulate(&mut root, model, config.cpuct)?;
    }

    let evaluations = root
        .actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let child = root.children[index].as_deref();
            PuctActionEvaluation {
                action: *action,
                prior: root.priors[index],
                visits: child.map_or(0, |node| node.visits),
                value: child.map_or(0.0, |node| -node.estimated_value()),
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
        simulations: config.simulations,
        evaluations,
    })
}

fn seed_root_afterstates(root: &mut Node) {
    let root_player = root.state.turn;
    for (index, action) in root.actions.iter().copied().enumerate() {
        let next_state = root.state.apply_legal(action).state;
        let root_value = if next_state.winner.is_some() {
            -terminal_value(next_state)
        } else if next_state.ply >= next_state.config.max_plies {
            0.0
        } else {
            let score = evaluate(next_state, root_player, EvaluationWeights::default()) as f32;
            (score / 3_500.0).tanh()
        };
        let mut child = Node::new(next_state);
        // Node values are stored for the child side to move; the root scan is
        // evaluated from the parent's perspective.
        child.seeded_value = Some(-root_value);
        root.children[index] = Some(Box::new(child));
    }
}

fn simulate<M: PolicyValueModel>(node: &mut Node, model: &M, cpuct: f32) -> Result<f32, String> {
    if !node.expanded {
        let value = node.expand(model)?;
        node.visits += 1;
        node.value_sum += value;
        return Ok(value);
    }
    if node.state.winner.is_some()
        || node.state.ply >= node.state.config.max_plies
        || node.actions.is_empty()
    {
        let value = terminal_value(node.state);
        node.visits += 1;
        node.value_sum += value;
        return Ok(value);
    }

    let index = node.select_action(cpuct);
    if node.children[index].is_none() {
        node.children[index] = Some(Box::new(Node::new(
            node.state.apply_legal(node.actions[index]).state,
        )));
    }
    let child_value = simulate(
        node.children[index]
            .as_deref_mut()
            .expect("PUCT child created before simulation"),
        model,
        cpuct,
    )?;
    let value = -child_value;
    node.visits += 1;
    node.value_sum += value;
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
        assert!(GameState::new()
            .legal_actions()
            .contains(&result.action.unwrap()));
    }

    #[test]
    fn root_afterstate_scan_seeds_every_legal_child() {
        let mut root = Node::new(GameState::new());
        root.expand(&TestModel).expect("expand root");
        let action_count = root.actions.len();

        seed_root_afterstates(&mut root);

        assert_eq!(root.children.len(), action_count);
        assert!(root
            .children
            .iter()
            .all(|child| child.as_ref().is_some_and(|node| node.seeded_value.is_some())));
        assert!(root.children.iter().all(|child| child.as_ref().unwrap().visits == 0));
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
            },
        )
        .expect("run capped PUCT");
        assert_eq!(result.action, None);
        assert_eq!(result.value, 0.0);
        assert_eq!(result.simulations, 0);
        assert!(result.evaluations.is_empty());
    }
}
