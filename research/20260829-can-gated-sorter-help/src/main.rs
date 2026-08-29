use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use pathagon_engine::inference::{OnnxGnnPolicyValueModel, PolicyValueModel};
use pathagon_engine::search::{
    analyze_action, evaluate, search_best_action_with_root_order_and_root_limit,
    search_best_action_with_tactical_filter, tactical_root_safe_actions, EvaluationWeights,
    SearchConfig,
};
use pathagon_engine::{Action, BoardConfig, GameState};
use serde_json::{json, Map, Value};

const HISTORICAL_CONFIG: SearchConfig = SearchConfig {
    depth: 4,
    max_nodes: 2_000,
    beam_width: 8,
    weights: EvaluationWeights {
        path: 240,
        material: 110,
        capture: 700,
        structure: 55,
        threat: 130,
        edge: 80,
    },
    tactical_proof_horizon: None,
};

const TEACHER_CONFIG: SearchConfig = SearchConfig {
    depth: 5,
    max_nodes: 6_000,
    beam_width: 16,
    weights: EvaluationWeights {
        path: 240,
        material: 110,
        capture: 700,
        structure: 55,
        threat: 130,
        edge: 80,
    },
    tactical_proof_horizon: None,
};

const ROOT_PLIES: [u16; 6] = [8, 16, 24, 32, 48, 64];
const SORT_POOL: usize = 8;
#[derive(Clone, Copy)]
struct Gate {
    id: &'static str,
    minimum_confidence: f32,
    maximum_native_gap: i32,
}

// These thresholds were fixed before running the audit. Confidence is the
// model-logit advantage over the native first action; native gap is the
// successor heuristic gap between the first two safe native actions.
const GATES: [Gate; 7] = [
    Gate {
        id: "strict",
        minimum_confidence: 0.40,
        maximum_native_gap: 100,
    },
    Gate {
        id: "high-confidence",
        minimum_confidence: 0.20,
        maximum_native_gap: 250,
    },
    Gate {
        id: "balanced",
        minimum_confidence: 0.10,
        maximum_native_gap: 250,
    },
    Gate {
        id: "low-confidence",
        minimum_confidence: 0.05,
        maximum_native_gap: 500,
    },
    Gate {
        id: "permissive",
        minimum_confidence: 0.00,
        maximum_native_gap: 500,
    },
    Gate {
        id: "ambiguous-only",
        minimum_confidence: 0.00,
        maximum_native_gap: 100,
    },
    Gate {
        id: "always-on",
        minimum_confidence: f32::NEG_INFINITY,
        maximum_native_gap: i32::MAX,
    },
];

#[derive(Clone, Copy)]
struct Rng(u32);

impl Rng {
    const fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let mut value = self.0;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        value ^ (value >> 14)
    }

    fn index(&mut self, length: usize) -> usize {
        if length == 0 {
            0
        } else {
            ((self.next() as u64 * length as u64) >> 32) as usize
        }
    }
}

#[derive(Clone)]
struct Root {
    source_family: String,
    split: &'static str,
    phase: &'static str,
    state: GameState,
}

fn main() {
    let args = parse_args();
    let model_path = PathBuf::from(
        args.get("model").and_then(Value::as_str).unwrap_or(
            "research/20260827-pathfinder-rust-sorter/artifacts/compact-gnn-policy.onnx",
        ),
    );
    let output_path = PathBuf::from(
        args.get("output")
            .and_then(Value::as_str)
            .unwrap_or("research/20260829-can-gated-sorter-help/workspace/calibration.json"),
    );
    let family_count = number(&args, "families", 24_u32).max(2);
    let holdout_families = number(&args, "holdout-families", family_count / 2).clamp(1, family_count - 1);
    let roots = generate_roots(family_count, holdout_families);
    let model_bytes = fs::read(&model_path)
        .unwrap_or_else(|error| fail(&format!("cannot read model {}: {error}", model_path.display())));
    let model = OnnxGnnPolicyValueModel::from_bytes(&model_bytes)
        .unwrap_or_else(|error| fail(&format!("cannot load model {}: {error}", model_path.display())));

    eprintln!(
        "gated-sorter-audit: {} roots, {} calibration families, {} holdout families",
        roots.len(),
        family_count - holdout_families,
        holdout_families,
    );
    let records = roots
        .iter()
        .enumerate()
        .map(|(index, root)| audit_root(index, root, &model))
        .collect::<Vec<_>>();
    let report = build_report(&model_path, model_bytes.len(), family_count, holdout_families, &records);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", parent.display())));
    }
    fs::write(&output_path, serde_json::to_vec_pretty(&report).expect("serialize audit report"))
        .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output_path.display())));
    println!("{}", serde_json::to_string(&report).expect("serialize audit summary"));
}

fn generate_roots(family_count: u32, holdout_families: u32) -> Vec<Root> {
    let config = BoardConfig::new(7, 14)
        .and_then(|config| config.with_max_plies(180))
        .expect("valid audit board");
    let calibration_families = family_count - holdout_families;
    let mut roots = Vec::new();
    for family in 0..family_count {
        let split = if family < calibration_families {
            "calibration"
        } else {
            "holdout"
        };
        let mut state = GameState::with_config(config);
        let mut rng = Rng::new(0x2026_0829_u32.wrapping_add(family.wrapping_mul(0x9e37_79b9)));
        let mut next_root = 0;
        for ply in 0..=ROOT_PLIES[ROOT_PLIES.len() - 1] {
            if next_root < ROOT_PLIES.len() && ply == ROOT_PLIES[next_root] && state.winner.is_none() {
                roots.push(Root {
                    source_family: format!("random-walk-{family:02}"),
                    split,
                    phase: phase(state),
                    state,
                });
                next_root += 1;
            }
            if state.winner.is_some() {
                break;
            }
            let actions = state.legal_actions();
            let Some(action) = actions.get(rng.index(actions.len())).copied() else {
                break;
            };
            state = state.apply_legal(action).state;
        }
    }
    roots
}

fn phase(state: GameState) -> &'static str {
    if state.reserve[state.turn.index()] > 0 {
        "placement"
    } else if state.ply >= 48 {
        "late-movement"
    } else {
        "movement"
    }
}

fn audit_root(index: usize, root: &Root, model: &OnnxGnnPolicyValueModel) -> Value {
    let state = root.state;
    let legal = state.legal_actions();
    let safe = tactical_root_safe_actions(state, state.turn, HISTORICAL_CONFIG.weights);
    let native_order = safe.clone();
    let pool_len = SORT_POOL.min(safe.len());
    let pool = &safe[..pool_len];
    let inference_started = Instant::now();
    let output = model
        .evaluate_with_actions(state, pool)
        .unwrap_or_else(|error| panic!("model inference failed for root {index}: {error}"));
    let inference_micros = inference_started.elapsed().as_micros() as u64;
    if output.policy_logits.len() < pool_len {
        fail(&format!(
            "model returned {} logits for {} actions at root {index}",
            output.policy_logits.len(), pool_len
        ));
    }
    let mut ranked = pool
        .iter()
        .copied()
        .zip(output.policy_logits.into_iter().take(pool_len))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.order().cmp(&right.0.order()))
    });
    let model_order = ranked
        .iter()
        .map(|(action, _)| *action)
        .chain(safe.iter().copied().skip(pool_len))
        .collect::<Vec<_>>();
    let native_top = native_order[0];
    let model_top = ranked[0].0;
    let native_scores = native_order
        .iter()
        .map(|action| heuristic_root_score(state, *action))
        .collect::<Vec<_>>();
    let native_gap = native_scores
        .first()
        .zip(native_scores.get(1))
        .map_or(0, |(first, second)| first.saturating_sub(*second));
    let native_top_score = native_scores[0];
    let model_native_index = native_order
        .iter()
        .position(|action| *action == model_top)
        .expect("model action must be in native safe order");
    let model_native_gap = native_top_score.saturating_sub(native_scores[model_native_index]);
    let model_top_logit = ranked[0].1;
    let native_top_logit = ranked
        .iter()
        .find(|(action, _)| *action == native_top)
        .map_or(output_logit_fallback(), |(_, logit)| *logit);
    let confidence = model_top_logit - native_top_logit;
    let top_two_margin = ranked
        .first()
        .zip(ranked.get(1))
        .map_or(0.0, |(first, second)| first.1 - second.1);
    let tactical = tactical_categories(state, &legal, &safe);
    let teacher_search = search_best_action_with_root_order_and_root_limit(
        state,
        TEACHER_CONFIG,
        &safe,
        false,
        Some(safe.len()),
    );
    let teacher_best_action = teacher_search
        .action
        .expect("teacher must return a legal action");
    let teacher_actions = [teacher_best_action, native_top, model_top];
    let mut teacher_scores = Vec::with_capacity(teacher_actions.len());
    for action in teacher_actions.into_iter().collect::<Vec<_>>() {
        if teacher_scores.iter().any(|(known, _)| *known == action) {
            continue;
        }
        let result = analyze_action(state, action, TEACHER_CONFIG)
            .unwrap_or_else(|error| panic!("teacher failed for root {index} action {action}: {error}"));
        teacher_scores.push((action, result.score));
    }
    let teacher_best_score = teacher_scores
        .iter()
        .find(|(action, _)| *action == teacher_best_action)
        .map(|(_, score)| *score)
        .expect("teacher action is scored");
    let native_teacher_score = teacher_scores
        .iter()
        .find(|(action, _)| *action == native_top)
        .map(|(_, score)| *score)
        .expect("native action is in teacher pool");
    let model_teacher_score = teacher_scores
        .iter()
        .find(|(action, _)| *action == model_top)
        .map(|(_, score)| *score)
        .expect("model action is in teacher pool");
    let native_regret_raw = i64::from(teacher_best_score) - i64::from(native_teacher_score);
    let model_regret_raw = i64::from(teacher_best_score) - i64::from(model_teacher_score);
    let native_search_started = Instant::now();
    let native_search = search_best_action_with_tactical_filter(state, HISTORICAL_CONFIG);
    let native_search_micros = native_search_started.elapsed().as_micros() as u64;
    let model_search_started = Instant::now();
    let model_search = search_best_action_with_root_order_and_root_limit(
        state,
        HISTORICAL_CONFIG,
        &model_order,
        false,
        Some(safe.len()),
    );
    let model_search_micros = model_search_started.elapsed().as_micros() as u64;
    let mut result = Map::new();
    result.insert("index".to_owned(), json!(index));
    result.insert("sourceFamily".to_owned(), json!(root.source_family));
    result.insert("split".to_owned(), json!(root.split));
    result.insert("phase".to_owned(), json!(root.phase));
    result.insert("ply".to_owned(), json!(state.ply));
    result.insert("legalActions".to_owned(), json!(legal.len()));
    result.insert("safeActions".to_owned(), json!(safe.len()));
    result.insert("category".to_owned(), json!(tactical.category));
    result.insert("immediateWinCount".to_owned(), json!(tactical.immediate_win_count));
    result.insert("opponentThreatCount".to_owned(), json!(tactical.opponent_threat_count));
    result.insert("nativeTop".to_owned(), json!(native_top.to_string()));
    result.insert("modelTop".to_owned(), json!(model_top.to_string()));
    result.insert("modelTopRankInPool".to_owned(), json!(1));
    result.insert("teacherBest".to_owned(), json!(teacher_best_action.to_string()));
    result.insert("teacherPool".to_owned(), json!(safe.len()));
    result.insert("teacherExhausted".to_owned(), json!(u32::from(teacher_search.exhausted)));
    result.insert("teacherNodes".to_owned(), json!(teacher_search.nodes));
    result.insert("teacherDepth".to_owned(), json!(teacher_search.completed_depth));
    result.insert("nativeGap".to_owned(), json!(native_gap));
    result.insert("modelNativeGap".to_owned(), json!(model_native_gap));
    result.insert("confidence".to_owned(), json!(confidence));
    result.insert("topTwoMargin".to_owned(), json!(top_two_margin));
    result.insert("nativeTopMatchesTeacher".to_owned(), json!(native_top == teacher_best_action));
    result.insert("modelTopMatchesTeacher".to_owned(), json!(model_top == teacher_best_action));
    result.insert("teacherInNativeTop8".to_owned(), json!(native_order.iter().take(8).any(|action| *action == teacher_best_action)));
    result.insert("teacherInModelTop4".to_owned(), json!(ranked.iter().take(4).any(|(action, _)| *action == teacher_best_action)));
    result.insert("nativeRegret".to_owned(), json!(native_regret_raw.max(0)));
    result.insert("modelRegret".to_owned(), json!(model_regret_raw.max(0)));
    result.insert("nativeRegretRaw".to_owned(), json!(native_regret_raw));
    result.insert("modelRegretRaw".to_owned(), json!(model_regret_raw));
    result.insert("nativeSearch".to_owned(), json!(native_search.action.map(|action| action.to_string())));
    result.insert("modelSearch".to_owned(), json!(model_search.action.map(|action| action.to_string())));
    result.insert("searchChanged".to_owned(), json!(native_search.action != model_search.action));
    result.insert("modelOrderChanged".to_owned(), json!(native_top != model_top));
    result.insert("nativeSearchNodes".to_owned(), json!(native_search.nodes));
    result.insert("modelSearchNodes".to_owned(), json!(model_search.nodes));
    result.insert("modelSearchDepth".to_owned(), json!(model_search.completed_depth));
    result.insert("modelInferenceMicros".to_owned(), json!(inference_micros));
    result.insert("nativeSearchMicros".to_owned(), json!(native_search_micros));
    result.insert("modelSearchMicros".to_owned(), json!(model_search_micros));
    result.insert("immediateWinActions".to_owned(), json!(tactical.immediate_wins.iter().map(ToString::to_string).collect::<Vec<_>>()));
    result.insert("safeActionsPreview".to_owned(), json!(safe.iter().take(8).map(ToString::to_string).collect::<Vec<_>>()));
    Value::Object(result)
}

fn output_logit_fallback() -> f32 {
    // The compact model has a legal-action logit for every action in the pool.
    // This finite sentinel is only reachable for a malformed output and keeps
    // the audit report serializable while the length check above remains fatal.
    -1.0e30
}

fn heuristic_root_score(state: GameState, action: Action) -> i32 {
    let transition = state.apply_legal(action);
    if transition.state.winner == Some(state.turn) {
        2_000_000_000
    } else {
        transition.captured.count_ones() as i32 * 10_000
            + evaluate(transition.state, state.turn, HISTORICAL_CONFIG.weights)
    }
}

struct TacticalCategories {
    category: &'static str,
    immediate_win_count: usize,
    opponent_threat_count: usize,
    immediate_wins: Vec<Action>,
}

fn tactical_categories(state: GameState, legal: &[Action], safe: &[Action]) -> TacticalCategories {
    let immediate_wins = legal
        .iter()
        .copied()
        .filter(|action| state.apply_legal(*action).state.winner == Some(state.turn))
        .collect::<Vec<_>>();
    let mut opponent_view = state;
    opponent_view.turn = state.turn.other();
    let opponent_threats = opponent_view
        .legal_actions()
        .into_iter()
        .filter(|action| opponent_view.apply_legal(*action).state.winner == Some(opponent_view.turn))
        .collect::<Vec<_>>();
    let forced_block = !opponent_threats.is_empty() && safe.len() < legal.len();
    let multi_capture = legal.iter().any(|action| state.apply_legal(*action).captured.count_ones() >= 2);
    let category = if !immediate_wins.is_empty() {
        "immediate-win"
    } else if forced_block {
        "forced-block"
    } else if multi_capture {
        "multi-capture"
    } else if state.reserve[state.turn.index()] > 0 {
        "placement"
    } else if state.ply >= 48 {
        "late-movement"
    } else {
        "movement"
    };
    TacticalCategories {
        category,
        immediate_win_count: immediate_wins.len(),
        opponent_threat_count: opponent_threats.len(),
        immediate_wins,
    }
}

fn build_report(
    model_path: &PathBuf,
    model_bytes: usize,
    family_count: u32,
    holdout_families: u32,
    records: &[Value],
) -> Value {
    let gates = GATES
        .iter()
        .map(|gate| gate_summary(*gate, records, "calibration"))
        .chain(GATES.iter().map(|gate| gate_summary(*gate, records, "holdout")))
        .collect::<Vec<_>>();
    json!({
        "schema": "pathagon-gated-sorter-calibration-v1",
        "protocol": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": 7,
            "reservePerPlayer": 14,
            "sourceFamilies": family_count,
            "calibrationFamilies": family_count - holdout_families,
            "holdoutFamilies": holdout_families,
            "rootPlies": ROOT_PLIES,
            "sortPool": SORT_POOL,
            "historicalSearch": search_json(HISTORICAL_CONFIG),
            "teacherSearch": search_json(TEACHER_CONFIG),
            "gatesPredeclared": GATES.iter().map(|gate| json!({"id": gate.id, "minimumConfidence": gate.minimum_confidence, "maximumNativeGap": gate.maximum_native_gap})).collect::<Vec<_>>(),
            "splitRule": "random-walk source families are disjoint; first calibrationFamilies are calibration and the remainder are holdout"
        },
        "model": {
            "path": model_path.to_string_lossy(),
            "bytes": model_bytes
        },
        "roots": records.len(),
        "gates": gates,
        "records": records
    })
}

fn search_json(config: SearchConfig) -> Value {
    json!({
        "depth": config.depth,
        "nodes": config.max_nodes,
        "beam": config.beam_width,
        "weights": {
            "path": config.weights.path,
            "material": config.weights.material,
            "capture": config.weights.capture,
            "structure": config.weights.structure,
            "threat": config.weights.threat,
            "edge": config.weights.edge
        }
    })
}

fn gate_summary(gate: Gate, records: &[Value], split: &str) -> Value {
    let selected = records
        .iter()
        .filter(|record| record["split"] == split)
        .collect::<Vec<_>>();
    let roots = selected.len() as u64;
    let mut opened = 0_u64;
    let mut changed = 0_u64;
    let mut native_matches = 0_u64;
    let mut selected_matches = 0_u64;
    let mut native_regret = 0_i64;
    let mut selected_regret = 0_i64;
    let mut search_changed = 0_u64;
    let mut immediate_regressions = 0_u64;
    let mut forced_block_regressions = 0_u64;
    let mut teacher_in_model_top4 = 0_u64;
    let mut teacher_in_native_top8 = 0_u64;
    let mut by_category = Map::new();
    for record in selected {
        let confidence = record["confidence"].as_f64().unwrap_or(f64::NEG_INFINITY) as f32;
        let native_gap = record["nativeGap"].as_i64().unwrap_or(i64::MAX) as i32;
        let order_changed = record["modelOrderChanged"].as_bool().unwrap_or(false);
        let is_open = order_changed
            && confidence >= gate.minimum_confidence
            && native_gap <= gate.maximum_native_gap;
        let native = record["nativeTop"].as_str().unwrap_or("");
        let model = record["modelTop"].as_str().unwrap_or("");
        let selected_action = if is_open { model } else { native };
        if is_open {
            opened += 1;
        }
        if selected_action != native {
            changed += 1;
        }
        if record["nativeTopMatchesTeacher"].as_bool().unwrap_or(false) {
            native_matches += 1;
        }
        if selected_action == record["teacherBest"].as_str().unwrap_or("") {
            selected_matches += 1;
        }
        native_regret += record["nativeRegret"].as_i64().unwrap_or(0);
        selected_regret += if is_open {
            record["modelRegret"].as_i64().unwrap_or(0)
        } else {
            record["nativeRegret"].as_i64().unwrap_or(0)
        };
        if is_open && record["searchChanged"].as_bool().unwrap_or(false) {
            search_changed += 1;
        }
        if record["teacherInModelTop4"].as_bool().unwrap_or(false) {
            teacher_in_model_top4 += 1;
        }
        if record["teacherInNativeTop8"].as_bool().unwrap_or(false) {
            teacher_in_native_top8 += 1;
        }
        let category = record["category"].as_str().unwrap_or("unknown");
        let entry = by_category
            .entry(category.to_owned())
            .or_insert_with(|| json!({"roots": 0, "opened": 0, "regressions": 0}));
        entry["roots"] = json!(entry["roots"].as_u64().unwrap_or(0) + 1);
        if is_open {
            entry["opened"] = json!(entry["opened"].as_u64().unwrap_or(0) + 1);
        }
        let selected_is_immediate_win = record["immediateWinActions"].as_array().map_or(false, |actions| {
            actions.iter().any(|action| action.as_str() == Some(selected_action))
        });
        let immediate_regression = category == "immediate-win" && !selected_is_immediate_win;
        if immediate_regression {
            immediate_regressions += 1;
            entry["regressions"] = json!(entry["regressions"].as_u64().unwrap_or(0) + 1);
        }
        let safe_preview_contains = record["safeActionsPreview"].as_array().map_or(false, |actions| actions.iter().any(|action| action.as_str() == Some(selected_action)));
        if category == "forced-block" && !safe_preview_contains {
            forced_block_regressions += 1;
            entry["regressions"] = json!(entry["regressions"].as_u64().unwrap_or(0) + 1);
        }
    }
    json!({
        "split": split,
        "id": gate.id,
        "minimumConfidence": gate.minimum_confidence,
        "maximumNativeGap": gate.maximum_native_gap,
        "roots": roots,
        "opened": opened,
        "activationRate": ratio(opened, roots),
        "decisionsChanged": changed,
        "decisionChangeRate": ratio(changed, roots),
        "nativeTeacherTop1": native_matches,
        "selectedTeacherTop1": selected_matches,
        "nativeTop1Rate": ratio(native_matches, roots),
        "selectedTop1Rate": ratio(selected_matches, roots),
        "nativeMeanRegret": mean(native_regret, roots),
        "selectedMeanRegret": mean(selected_regret, roots),
        "regretImprovement": mean(native_regret - selected_regret, roots),
        "activatedSearchChanges": search_changed,
        "activatedSearchChangeRate": ratio(search_changed, opened),
        "teacherInNativeTop8": teacher_in_native_top8,
        "teacherInModelTop4": teacher_in_model_top4,
        "immediateWinRegressions": immediate_regressions,
        "forcedBlockRegressions": forced_block_regressions,
        "byCategory": by_category
    })
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(total: i64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn parse_args() -> Map<String, Value> {
    let mut parsed = Map::new();
    let values = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < values.len() {
        if let Some(option) = values[index].strip_prefix("--") {
            let (key, value) = option.split_once('=').map_or_else(
                || {
                    if values.get(index + 1).is_some_and(|next| !next.starts_with("--")) {
                        index += 1;
                        (option, values[index].clone())
                    } else {
                        (option, "true".to_owned())
                    }
                },
                |(key, inline)| (key, inline.to_owned()),
            );
            parsed.insert(key.to_owned(), Value::String(value));
        }
        index += 1;
    }
    parsed
}

fn number<T>(args: &Map<String, Value>, key: &str, fallback: T) -> T
where
    T: std::str::FromStr,
{
    args.get(key)
        .and_then(Value::as_str)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn fail(message: &str) -> ! {
    eprintln!("gated-sorter-audit: {message}");
    std::process::exit(2)
}
