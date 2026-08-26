//! ONNX policy/value inference for native Rust and WASM.

use crate::model::{
    GnnPolicyValueInputs, PolicyValueInputs, ACTION_FEATURE_COUNT, BOARD_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT, MAX_ACTIONS,
};
use crate::GameState;
use tract::prelude::*;

pub struct PolicyValue {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

pub trait PolicyValueModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String>;
}

pub struct OnnxPolicyValueModel {
    model: tract::Runnable,
}

impl OnnxPolicyValueModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut inference = tract::onnx()
            .map_err(|error| error.to_string())?
            .load_buffer(bytes)
            .map_err(|error| error.to_string())?;
        inference.analyse().map_err(|error| error.to_string())?;
        let model = inference
            .into_model()
            .map_err(|error| error.to_string())?
            .into_runnable()
            .map_err(|error| error.to_string())?;
        Ok(Self { model })
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
            .run([board, global, action_specs, action_mask])
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
/// This deliberately implements the same small trait as the deployed CNN
/// runner, allowing the existing native PUCT implementation to benchmark a
/// QAdv checkpoint's shared trunk before the Q/A action head is ported.
pub struct OnnxGnnPolicyValueModel {
    model: tract::Runnable,
}

impl OnnxGnnPolicyValueModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut inference = tract::onnx()
            .map_err(|error| error.to_string())?
            .load_buffer(bytes)
            .map_err(|error| error.to_string())?;
        inference.analyse().map_err(|error| error.to_string())?;
        let model = inference
            .into_model()
            .map_err(|error| error.to_string())?
            .into_runnable()
            .map_err(|error| error.to_string())?;
        Ok(Self { model })
    }
}

impl PolicyValueModel for OnnxGnnPolicyValueModel {
    fn evaluate(&self, state: GameState) -> Result<PolicyValue, String> {
        let inputs = GnnPolicyValueInputs::from_state(state)?;
        let node_features = tensor(
            &[1, GNN_GRAPH_NODE_COUNT, GNN_NODE_FEATURE_COUNT],
            &inputs.node_features,
        )?;
        let global = tensor(&[1, GLOBAL_FEATURE_COUNT], &inputs.global_features)?;
        let action_specs = inputs
            .action_specs
            .iter()
            .flat_map(|action| [f32::from(action.kind), f32::from(action.from), f32::from(action.to)])
            .collect::<Vec<_>>();
        let action_specs = tensor(&[1, MAX_ACTIONS, ACTION_FEATURE_COUNT], &action_specs)?;
        let action_mask = tensor(&[1, MAX_ACTIONS], &inputs.action_mask)?;
        let outputs = self
            .model
            .run([node_features, global, action_specs, action_mask])
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
        Ok(PolicyValue { policy_logits, value })
    }
}

fn tensor(shape: &[usize], values: &[f32]) -> Result<tract::Tensor, String> {
    let expected = shape.iter().product::<usize>();
    if values.len() != expected {
        return Err(format!(
            "tensor shape expects {expected} values, received {}",
            values.len()
        ));
    }
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    tract::Tensor::from_bytes(DatumType::F32, shape, &bytes).map_err(|error| error.to_string())
}

fn f32_values(tensor: &tract::Tensor) -> Result<Vec<f32>, String> {
    let (datum_type, _shape, bytes) = tensor.as_bytes().map_err(|error| error.to_string())?;
    if datum_type != DatumType::F32 {
        return Err(format!(
            "expected f32 model output, received {datum_type:?}"
        ));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32-sized chunk")))
        .collect())
}
