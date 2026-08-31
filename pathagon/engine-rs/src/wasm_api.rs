//! Minimal wasm-bindgen surface for the browser adapter.
//!
//! JSON is intentional at this boundary. It keeps the JavaScript contract
//! inspectable and lets the native Rust, WASM Rust, and parity test paths call
//! the exact same conversion and rules functions before we optimize the hot
//! path with typed views.

use wasm_bindgen::prelude::*;

use crate::runtime::{
    analyze_action_json, analyze_actions_json, apply_action_json, apply_action_transition_json,
    legal_actions_json, lunatic_action_json, position_json, rank_transition_policy_json,
    search_best_action_json, search_best_action_with_tactical_filter_deadline_json,
    search_best_action_with_tactical_filter_deadline_progress_json,
    search_best_action_with_tactical_filter_json, search_transition_policy_json,
    search_transition_policy_with_progress_json,
};
use crate::transition_policy::TransitionPolicyModel;
use crate::{BoardConfig, GameState};

#[cfg(feature = "inference")]
use crate::contract::ContractAction;
#[cfg(feature = "inference")]
use crate::inference::{OnnxPolicyValueModel, PolicyValueModel};
#[cfg(feature = "inference")]
use crate::puct::{search as puct_search, PuctConfig};
#[cfg(feature = "inference")]
use crate::runtime::parse_position;
#[cfg(feature = "inference")]
use serde::Serialize;

fn js_error(error: String) -> JsValue {
    JsValue::from_str(&error)
}

fn search_progress_callback(callback: &js_sys::Function) -> crate::search::SearchProgressCallback {
    let callback = callback.clone();
    Box::new(move |nodes, table_hits| {
        let _ = callback.call2(
            &JsValue::NULL,
            &JsValue::from_f64(nodes as f64),
            &JsValue::from_f64(table_hits as f64),
        );
    })
}

#[wasm_bindgen]
pub fn pathagon_engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen]
pub fn pathagon_initial_position(
    board_size: u8,
    reserve_per_player: u8,
) -> Result<String, JsValue> {
    let config = BoardConfig::new(board_size, reserve_per_player).map_err(js_error)?;
    position_json(GameState::with_config(config)).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_legal_actions(position: &str) -> Result<String, JsValue> {
    legal_actions_json(position).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_apply_action(position: &str, action: &str) -> Result<String, JsValue> {
    apply_action_json(position, action).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_apply_action_transition(position: &str, action: &str) -> Result<String, JsValue> {
    apply_action_transition_json(position, action).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_search_best_action(position: &str, config: &str) -> Result<String, JsValue> {
    search_best_action_json(position, config).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_search_best_action_with_tactical_filter(
    position: &str,
    config: &str,
) -> Result<String, JsValue> {
    search_best_action_with_tactical_filter_json(position, config).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_search_best_action_with_tactical_filter_deadline(
    position: &str,
    config: &str,
    deadline_ms: u32,
) -> Result<String, JsValue> {
    search_best_action_with_tactical_filter_deadline_json(position, config, deadline_ms)
        .map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_search_best_action_with_tactical_filter_deadline_progress(
    position: &str,
    config: &str,
    deadline_ms: u32,
    callback: &js_sys::Function,
) -> Result<String, JsValue> {
    search_best_action_with_tactical_filter_deadline_progress_json(
        position,
        config,
        deadline_ms,
        search_progress_callback(callback),
    )
    .map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_lunatic_action(position: &str) -> Result<String, JsValue> {
    lunatic_action_json(position).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_analyze_action(
    position: &str,
    action: &str,
    config: &str,
) -> Result<String, JsValue> {
    analyze_action_json(position, action, config).map_err(js_error)
}

#[wasm_bindgen]
pub fn pathagon_analyze_actions(
    position: &str,
    config: &str,
    max_actions: u32,
) -> Result<String, JsValue> {
    analyze_actions_json(position, config, max_actions as usize).map_err(js_error)
}

/// Packaged explicit action-transition policy. The model is loaded from the
/// versioned JSON artifact by JavaScript, while all state transitions and
/// legality checks remain in Rust.
#[wasm_bindgen]
pub struct PathagonTransitionPolicyModel {
    model: TransitionPolicyModel,
}

#[wasm_bindgen]
impl PathagonTransitionPolicyModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<PathagonTransitionPolicyModel, JsValue> {
        let model = TransitionPolicyModel::from_bytes(bytes).map_err(js_error)?;
        Ok(Self { model })
    }

    #[wasm_bindgen(js_name = modelName)]
    pub fn model_name(&self) -> String {
        self.model.model.clone()
    }

    #[wasm_bindgen(js_name = encoding)]
    pub fn encoding(&self) -> String {
        self.model.encoding.clone()
    }

    #[wasm_bindgen(js_name = score)]
    pub fn score(&self, position: &str, action: &str, safe: bool) -> Result<f32, JsValue> {
        let state = crate::runtime::parse_position(position).map_err(js_error)?;
        let contract_action: crate::contract::ContractAction =
            serde_json::from_str(action).map_err(|error| js_error(error.to_string()))?;
        let action = contract_action.into();
        if !state.legal_actions().contains(&action) {
            return Err(js_error(
                "transition-policy score received an illegal action".to_owned(),
            ));
        }
        Ok(self.model.score(state, action, safe))
    }

    #[wasm_bindgen(js_name = rankActions)]
    pub fn rank_actions(&self, position: &str, max_actions: u32) -> Result<String, JsValue> {
        rank_transition_policy_json(position, &self.model, max_actions as usize).map_err(js_error)
    }

    #[wasm_bindgen(js_name = searchBestAction)]
    pub fn search_best_action(
        &self,
        position: &str,
        config: &str,
        deadline_ms: u32,
    ) -> Result<String, JsValue> {
        search_transition_policy_json(position, config, &self.model, deadline_ms).map_err(js_error)
    }

    #[wasm_bindgen(js_name = searchBestActionWithProgress)]
    pub fn search_best_action_with_progress(
        &self,
        position: &str,
        config: &str,
        deadline_ms: u32,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        search_transition_policy_with_progress_json(
            position,
            config,
            &self.model,
            deadline_ms,
            search_progress_callback(callback),
        )
        .map_err(js_error)
    }
}

#[cfg(feature = "inference")]
#[wasm_bindgen]
pub struct PathagonCnnModel {
    model: OnnxPolicyValueModel,
}

#[cfg(feature = "inference")]
#[derive(Serialize)]
struct RuntimePolicyValue {
    actions: Vec<ContractAction>,
    #[serde(rename = "policyLogits")]
    policy_logits: Vec<f32>,
    value: f32,
}

#[cfg(feature = "inference")]
#[derive(Serialize)]
struct RuntimePuctActionEvaluation {
    action: ContractAction,
    prior: f32,
    visits: u32,
    value: f32,
}

#[cfg(feature = "inference")]
#[derive(Serialize)]
struct RuntimePuctResult {
    action: Option<ContractAction>,
    value: f32,
    simulations: u32,
    evaluations: Vec<RuntimePuctActionEvaluation>,
}

#[cfg(feature = "inference")]
#[wasm_bindgen]
impl PathagonCnnModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<PathagonCnnModel, JsValue> {
        let model = OnnxPolicyValueModel::from_bytes(bytes).map_err(js_error)?;
        Ok(Self { model })
    }

    #[wasm_bindgen(js_name = evaluate)]
    pub fn evaluate_position(&self, position: &str) -> Result<String, JsValue> {
        let state = parse_position(position).map_err(js_error)?;
        let actions = state.legal_actions();
        let output = self.model.evaluate(state).map_err(js_error)?;
        let policy_logits = output
            .policy_logits
            .into_iter()
            .take(actions.len())
            .collect();
        let response = RuntimePolicyValue {
            actions: actions.into_iter().map(Into::into).collect(),
            policy_logits,
            value: output.value,
        };
        serde_json::to_string(&response).map_err(|error| js_error(error.to_string()))
    }

    #[wasm_bindgen(js_name = selectAction)]
    pub fn select_action(
        &self,
        position: &str,
        simulations: u32,
        cpuct: f32,
    ) -> Result<String, JsValue> {
        let state = parse_position(position).map_err(js_error)?;
        let result = puct_search(
            &self.model,
            state,
            PuctConfig {
                simulations,
                cpuct,
                use_action_value_seeds: false,
            },
        )
        .map_err(js_error)?;
        let response = RuntimePuctResult {
            action: result.action.map(Into::into),
            value: result.value,
            simulations: result.simulations,
            evaluations: result
                .evaluations
                .into_iter()
                .map(|evaluation| RuntimePuctActionEvaluation {
                    action: evaluation.action.into(),
                    prior: evaluation.prior,
                    visits: evaluation.visits,
                    value: evaluation.value,
                })
                .collect(),
        };
        serde_json::to_string(&response).map_err(|error| js_error(error.to_string()))
    }
}
