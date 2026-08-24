//! Minimal wasm-bindgen surface for the browser adapter.
//!
//! JSON is intentional at this boundary. It keeps the JavaScript contract
//! inspectable and lets the native Rust, WASM Rust, and parity test paths call
//! the exact same conversion and rules functions before we optimize the hot
//! path with typed views.

use wasm_bindgen::prelude::*;

use crate::runtime::{
    analyze_action_json, analyze_actions_json, apply_action_json, legal_actions_json,
    lunatic_action_json, position_json, search_best_action_json,
};
use crate::{BoardConfig, GameState};

fn js_error(error: String) -> JsValue {
    JsValue::from_str(&error)
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
pub fn pathagon_search_best_action(position: &str, config: &str) -> Result<String, JsValue> {
    search_best_action_json(position, config).map_err(js_error)
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
