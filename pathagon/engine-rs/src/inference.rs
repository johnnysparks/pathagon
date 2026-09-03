//! ONNX policy/value inference for native Rust and WASM.

use std::sync::Arc;

use crate::model::{
    GnnPolicyValueInputs, GnnQAdvInputs, PolicyValueInputs, ACTION_FEATURE_COUNT,
    BOARD_FEATURE_COUNT, GLOBAL_FEATURE_COUNT, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT,
    MAX_ACTIONS, QADV_TRANSITION_FEATURE_COUNT,
};
use crate::GameState;
use tract::prelude::*;

pub struct PolicyValue {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

pub struct QAdvPolicyValue {
    pub policy_logits: Vec<f32>,
    pub value: f32,
    pub q_values: Vec<f32>,
}

pub struct JepaActionValue {
    pub rank_logits: Vec<f32>,
    pub action_values: Vec<f32>,
}

static ZERO_QADV_TRANSITION_FEATURES: [f32; MAX_ACTIONS * QADV_TRANSITION_FEATURE_COUNT] =
    [0.0; MAX_ACTIONS * QADV_TRANSITION_FEATURE_COUNT];

pub trait PolicyValueModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String>;

    /// Evaluate a state when the caller already has its canonical action list.
    /// Implementations may use it to avoid regenerating legal actions.
    fn evaluate_with_actions(
        &self,
        state: GameState,
        _actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        self.evaluate(state)
    }

    /// Evaluate only the policy/value path when a model has an auxiliary
    /// action head that is not needed by tree expansion. Implementations with
    /// no cheaper path inherit the full evaluation.
    fn evaluate_policy_value(&self, state: GameState) -> Result<PolicyValue, String> {
        self.evaluate(state)
    }

    fn evaluate_policy_value_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        self.evaluate_with_actions(state, actions)
    }

    /// Evaluate the policy/value path and, when available, an action-value
    /// vector aligned with `actions`. QAdv models override this to return the
    /// action head from the same inference call. PUCT uses those values as
    /// seeds for unvisited children; ordinary policy/value models retain the
    /// previous zero/heuristic seed behavior.
    fn evaluate_policy_value_and_action_values_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<(PolicyValue, Option<Vec<f32>>), String> {
        Ok((
            self.evaluate_policy_value_with_actions(state, actions)?,
            None,
        ))
    }
}

pub struct OnnxPolicyValueModel {
    model: Arc<tract::tract_core::model::typed::TypedRunnableModel>,
}

impl OnnxPolicyValueModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            model: load_model(bytes)?,
        })
    }
}

impl PolicyValueModel for OnnxPolicyValueModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
        let inputs = PolicyValueInputs::from_state(state)?;
        let board = tensor(&[1, BOARD_FEATURE_COUNT, 7, 7], &inputs.board_features)?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| {
                [
                    f32::from(action.kind),
                    f32::from(action.from),
                    f32::from(action.to),
                ]
            })
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let outputs = self
            .model
            .run(tvec![
                board.into_tvalue(),
                global.into_tvalue(),
                action_specs.into_tvalue(),
                action_mask.into_tvalue(),
            ])
            .map_err(|error| error.to_string())?;
        if outputs.len() != 2 {
            return Err(format!(
                "policy/value model returned {} outputs, expected 2",
                outputs.len()
            ));
        }
        let policy_logits = f32_values(&outputs[0])?;
        let value = f32_values(&outputs[1])?
            .first()
            .copied()
            .ok_or_else(|| "policy/value model returned an empty value tensor".to_owned())?;
        Ok(PolicyValue {
            policy_logits,
            value,
        })
    }
}

/// ONNX policy/value runner for the fixed 7x7 GNN export.
///
/// This implements the same small trait as the deployed CNN runner, allowing
/// native PUCT to share the Python checkpoint's graph policy/value trunk.
pub struct OnnxGnnPolicyValueModel {
    model: Arc<tract::tract_core::model::typed::TypedRunnableModel>,
}

impl OnnxGnnPolicyValueModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            model: load_model(bytes)?,
        })
    }
}

impl PolicyValueModel for OnnxGnnPolicyValueModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
        let actions = state.legal_actions();
        self.evaluate_with_actions(state, &actions)
    }

    fn evaluate_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        let inputs = GnnPolicyValueInputs::from_state_with_actions(state, actions)?;
        let node_features = tensor(
            &[1, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT],
            &inputs.node_features,
        )?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| {
                [
                    f32::from(action.kind),
                    f32::from(action.from),
                    f32::from(action.to),
                ]
            })
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let outputs = self
            .model
            .run(tvec![
                node_features.into_tvalue(),
                global.into_tvalue(),
                action_specs.into_tvalue(),
                action_mask.into_tvalue(),
            ])
            .map_err(|error| error.to_string())?;
        if outputs.len() != 2 {
            return Err(format!(
                "GNN policy/value model returned {} outputs, expected 2",
                outputs.len()
            ));
        }
        let policy_logits = f32_values(&outputs[0])?;
        let value = f32_values(&outputs[1])?
            .first()
            .copied()
            .ok_or_else(|| "GNN policy/value model returned an empty value tensor".to_owned())?;
        Ok(PolicyValue {
            policy_logits,
            value,
        })
    }
}

/// ONNX runner for the action-conditioned JEPA afterstate heads. The model
/// shares the fixed graph/action ABI with the GNN policy path, but its outputs
/// are explicitly action-ranking logits and bounded afterstate values.
pub struct OnnxJepaModel {
    model: Arc<tract::tract_core::model::typed::TypedRunnableModel>,
}

impl OnnxJepaModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            model: load_model(bytes)?,
        })
    }

    pub fn evaluate_jepa(&self, state: GameState) -> Result<JepaActionValue, String> {
        let actions = state.legal_actions();
        self.evaluate_jepa_with_actions(state, &actions)
    }

    pub fn evaluate_jepa_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<JepaActionValue, String> {
        let inputs = GnnPolicyValueInputs::from_state_with_actions(state, actions)?;
        let node_features = tensor(
            &[1, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT],
            &inputs.node_features,
        )?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| {
                [
                    f32::from(action.kind),
                    f32::from(action.from),
                    f32::from(action.to),
                ]
            })
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let outputs = self
            .model
            .run(tvec![
                node_features.into_tvalue(),
                global.into_tvalue(),
                action_specs.into_tvalue(),
                action_mask.into_tvalue(),
            ])
            .map_err(|error| error.to_string())?;
        if outputs.len() != 2 {
            return Err(format!(
                "JEPA action model returned {} outputs, expected 2",
                outputs.len()
            ));
        }
        Ok(JepaActionValue {
            rank_logits: f32_values(&outputs[0])?,
            action_values: f32_values(&outputs[1])?,
        })
    }
}

/// ONNX runner for the full Q/Advantage export, including deterministic
/// transition features and the per-action Q head.
pub struct OnnxQAdvModel {
    model: Arc<tract::tract_core::model::typed::TypedRunnableModel>,
}

impl OnnxQAdvModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            model: load_model(bytes)?,
        })
    }

    pub fn evaluate_qadv(&self, state: GameState) -> Result<QAdvPolicyValue, String> {
        let actions = state.legal_actions();
        self.evaluate_qadv_with_actions(state, &actions)
    }

    pub fn evaluate_qadv_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<QAdvPolicyValue, String> {
        let inputs = GnnQAdvInputs::from_state_with_actions(state, actions)?;
        let node_features = tensor(
            &[1, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT],
            &inputs.node_features,
        )?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| {
                [
                    f32::from(action.kind),
                    f32::from(action.from),
                    f32::from(action.to),
                ]
            })
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let transition_features = tensor(
            &[1, MAX_ACTIONS, QADV_TRANSITION_FEATURE_COUNT],
            &inputs.transition_features,
        )?;
        let outputs = self
            .model
            .run(tvec![
                node_features.into_tvalue(),
                global.into_tvalue(),
                action_specs.into_tvalue(),
                action_mask.into_tvalue(),
                transition_features.into_tvalue(),
            ])
            .map_err(|error| error.to_string())?;
        if outputs.len() != 3 {
            return Err(format!(
                "QAdv model returned {} outputs, expected 3",
                outputs.len()
            ));
        }
        let policy_logits = f32_values(&outputs[0])?;
        let value = f32_values(&outputs[1])?
            .first()
            .copied()
            .ok_or_else(|| "QAdv model returned an empty value tensor".to_owned())?;
        let q_values = f32_values(&outputs[2])?;
        Ok(QAdvPolicyValue {
            policy_logits,
            value,
            q_values,
        })
    }

    /// The exported QAdv graph shares its policy/value trunk with the Q head;
    /// transition features feed only the third output. PUCT leaf expansion
    /// does not need Q values, so it can skip rebuilding every legal
    /// afterstate's transition vector and pass zero padding to the graph.
    fn evaluate_policy_value_only(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        let inputs = GnnPolicyValueInputs::from_state_with_actions(state, actions)?;
        let node_features = tensor(
            &[1, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT],
            &inputs.node_features,
        )?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| {
                [
                    f32::from(action.kind),
                    f32::from(action.from),
                    f32::from(action.to),
                ]
            })
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let transition_features = tensor(
            &[1, MAX_ACTIONS, QADV_TRANSITION_FEATURE_COUNT],
            &ZERO_QADV_TRANSITION_FEATURES,
        )?;
        let outputs = self
            .model
            .run(tvec![
                node_features.into_tvalue(),
                global.into_tvalue(),
                action_specs.into_tvalue(),
                action_mask.into_tvalue(),
                transition_features.into_tvalue(),
            ])
            .map_err(|error| error.to_string())?;
        if outputs.len() != 3 {
            return Err(format!(
                "QAdv model returned {} outputs, expected 3",
                outputs.len()
            ));
        }
        let policy_logits = f32_values(&outputs[0])?;
        let value = f32_values(&outputs[1])?
            .first()
            .copied()
            .ok_or_else(|| "QAdv model returned an empty value tensor".to_owned())?;
        Ok(PolicyValue {
            policy_logits,
            value,
        })
    }
}

impl PolicyValueModel for OnnxQAdvModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
        let output = self.evaluate_qadv(state)?;
        Ok(PolicyValue {
            policy_logits: output.policy_logits,
            value: output.value,
        })
    }

    fn evaluate_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        let output = self.evaluate_qadv_with_actions(state, actions)?;
        Ok(PolicyValue {
            policy_logits: output.policy_logits,
            value: output.value,
        })
    }

    fn evaluate_policy_value(&self, state: GameState) -> Result<PolicyValue, String> {
        let actions = state.legal_actions();
        self.evaluate_policy_value_with_actions(state, &actions)
    }

    fn evaluate_policy_value_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<PolicyValue, String> {
        self.evaluate_policy_value_only(state, actions)
    }

    fn evaluate_policy_value_and_action_values_with_actions(
        &self,
        state: GameState,
        actions: &[crate::Action],
    ) -> Result<(PolicyValue, Option<Vec<f32>>), String> {
        let output = self.evaluate_qadv_with_actions(state, actions)?;
        Ok((
            PolicyValue {
                policy_logits: output.policy_logits,
                value: output.value,
            },
            Some(output.q_values),
        ))
    }
}

fn load_model(
    bytes: &[u8],
) -> Result<Arc<tract::tract_core::model::typed::TypedRunnableModel>, String> {
    let inference = tract::onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .map_err(|error| error.to_string())?;
    inference
        .into_optimized()
        .map_err(|error| error.to_string())?
        .into_runnable()
        .map_err(|error| error.to_string())
}

fn tensor(shape: &[usize], values: &[f32]) -> Result<tract::tract_core::prelude::Tensor, String> {
    let expected = shape.iter().product::<usize>();
    if values.len() != expected {
        return Err(format!(
            "tensor shape expects {expected} values, received {}",
            values.len()
        ));
    }
    tract::tract_core::prelude::Tensor::from_shape(shape, values).map_err(|error| error.to_string())
}

fn f32_values(tensor: &tract::tract_core::prelude::Tensor) -> Result<Vec<f32>, String> {
    tensor
        .to_plain_array_view::<f32>()
        .map(|values| values.iter().copied().collect())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubModel;

    impl PolicyValueModel for StubModel {
        fn evaluate(&self, _state: GameState) -> Result<PolicyValue, String> {
            Ok(PolicyValue {
                policy_logits: vec![0.25],
                value: -0.5,
            })
        }
    }

    #[test]
    fn policy_value_trait_defaults_delegate_and_preserve_no_action_values() {
        let model = StubModel;
        let state = GameState::new();
        let actions = state.legal_actions();
        let direct = model.evaluate(state).unwrap();
        assert_eq!(
            model.evaluate_with_actions(state, &actions).unwrap().value,
            direct.value
        );
        assert_eq!(
            model.evaluate_policy_value(state).unwrap().policy_logits,
            direct.policy_logits
        );
        assert_eq!(
            model
                .evaluate_policy_value_with_actions(state, &actions)
                .unwrap()
                .value,
            direct.value
        );
        let (policy_value, action_values) = model
            .evaluate_policy_value_and_action_values_with_actions(state, &actions)
            .unwrap();
        assert_eq!(policy_value.value, direct.value);
        assert_eq!(action_values, None);
    }

    #[test]
    fn tensor_and_plain_float_helpers_validate_shape_and_type() {
        assert!(tensor(&[2], &[1.0]).is_err());
        let value_tensor = tensor(&[2], &[1.0, -2.0]).unwrap();
        assert_eq!(f32_values(&value_tensor).unwrap(), vec![1.0, -2.0]);
        let integer_tensor =
            tract::tract_core::prelude::Tensor::from_shape(&[1], &[1_i32]).unwrap();
        assert!(f32_values(&integer_tensor).is_err());
    }

    #[test]
    fn malformed_onnx_is_rejected_by_all_model_constructors() {
        assert!(load_model(&[]).is_err());
        assert!(OnnxPolicyValueModel::from_bytes(&[]).is_err());
        assert!(OnnxGnnPolicyValueModel::from_bytes(&[]).is_err());
        assert!(OnnxQAdvModel::from_bytes(&[]).is_err());
    }
}
