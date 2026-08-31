//! Disk-facing retrograde propagation for exact endgame graphs.
//!
//! State generation is intentionally a separate concern. A generator supplies
//! D4-canonical keys and every legal child edge; this module then performs
//! monotonic W/D/L propagation. Missing edges or incomplete enumeration never
//! become draws. A closed unresolved region is a draw only when every node in
//! that region is marked complete.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::ground_truth::{GroundTruthOutcome, GroundTruthValue};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeNode {
    pub key: String,
    #[serde(default)]
    pub children: Vec<String>,
    /// Set true only after the generator has enumerated every legal action.
    #[serde(default)]
    pub complete: bool,
    /// Terminal result from this node's side-to-move perspective.
    #[serde(default)]
    pub terminal: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeValue {
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeStats {
    pub nodes: usize,
    pub edges: usize,
    pub rounds: usize,
    pub solved: usize,
    pub draws: usize,
    pub unknown: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeOutput {
    pub schema_version: u8,
    pub table_family: String,
    pub values: BTreeMap<String, RetrogradeValue>,
    pub stats: RetrogradeStats,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrontierCheckpoint {
    pub schema_version: u8,
    pub table_family: String,
    pub input_nodes: usize,
    pub input_edges: usize,
    pub rounds: usize,
    pub solved: usize,
    pub draws: usize,
    pub unknown: usize,
    pub complete_graph: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RetrogradeGraph {
    nodes: BTreeMap<String, RetrogradeNode>,
}

impl RetrogradeGraph {
    pub fn insert(&mut self, node: RetrogradeNode) -> Result<(), String> {
        if node.key.is_empty() {
            return Err("retrograde node key cannot be empty".to_owned());
        }
        if node.children.iter().any(|child| child.is_empty()) {
            return Err(format!(
                "retrograde node {} has an empty child key",
                node.key
            ));
        }
        if let Some(terminal) = node.terminal.as_deref() {
            if parse_outcome(terminal).is_none() {
                return Err(format!(
                    "retrograde node {} has an invalid terminal outcome",
                    node.key
                ));
            }
        }
        if let Some(previous) = self.nodes.insert(node.key.clone(), node.clone()) {
            if previous != node {
                return Err(format!("contradictory retrograde node {}", node.key));
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn edge_count(&self) -> usize {
        self.nodes.values().map(|node| node.children.len()).sum()
    }

    pub fn solve(&self) -> (BTreeMap<String, RetrogradeValue>, RetrogradeStats) {
        let mut values = self
            .nodes
            .iter()
            .filter_map(|(key, node)| {
                node.terminal
                    .as_deref()
                    .and_then(parse_outcome)
                    .filter(|outcome| outcome.is_known())
                    .map(|outcome| {
                        (
                            key.clone(),
                            RetrogradeValue {
                                outcome,
                                distance: Some(0),
                            },
                        )
                    })
            })
            .collect::<BTreeMap<_, _>>();
        let mut rounds = 0;
        loop {
            rounds += 1;
            let mut updates = Vec::new();
            for (key, node) in &self.nodes {
                if values.contains_key(key) {
                    continue;
                }
                let Some(value) = self.resolve_node(node, &values) else {
                    continue;
                };
                updates.push((key.clone(), value));
            }
            if updates.is_empty() {
                break;
            }
            for (key, value) in updates {
                values.insert(key, value);
            }
        }

        for key in self.nodes.keys().cloned().collect::<Vec<_>>() {
            if !values.contains_key(&key) && self.closed_unresolved_region(&key, &values) {
                values.insert(
                    key,
                    RetrogradeValue {
                        outcome: GroundTruthOutcome::Draw,
                        distance: None,
                    },
                );
            }
        }
        let solved = values
            .values()
            .filter(|value| value.outcome.is_known())
            .count();
        let draws = values
            .values()
            .filter(|value| value.outcome == GroundTruthOutcome::Draw)
            .count();
        let unknown = self.nodes.len().saturating_sub(values.len());
        (
            values,
            RetrogradeStats {
                nodes: self.nodes.len(),
                edges: self.edge_count(),
                rounds,
                solved,
                draws,
                unknown,
            },
        )
    }

    fn closed_unresolved_region(
        &self,
        start: &str,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> bool {
        let mut stack = vec![start.to_owned()];
        let mut visited = std::collections::BTreeSet::new();
        while let Some(key) = stack.pop() {
            if !visited.insert(key.clone()) {
                continue;
            }
            let Some(node) = self.nodes.get(&key) else {
                return false;
            };
            if !node.complete {
                return false;
            }
            for child in &node.children {
                if !self.nodes.contains_key(child) {
                    return false;
                }
                if !values.contains_key(child) {
                    stack.push(child.clone());
                }
            }
        }
        true
    }

    fn resolve_node(
        &self,
        node: &RetrogradeNode,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> Option<RetrogradeValue> {
        if !node.complete || node.children.is_empty() && !node.complete {
            return None;
        }
        if node.children.is_empty() {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            });
        }
        let mut child_values = Vec::with_capacity(node.children.len());
        for child in &node.children {
            child_values.push(values.get(child)?);
        }
        let parent_values = child_values
            .iter()
            .map(|value| GroundTruthValue {
                outcome: value.outcome.negate(),
                distance: value.distance.map(|distance| distance.saturating_add(1)),
            })
            .collect::<Vec<_>>();
        if let Some(distance) = parent_values
            .iter()
            .filter(|value| value.outcome == GroundTruthOutcome::Win)
            .filter_map(|value| value.distance)
            .min()
        {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(distance),
            });
        }
        if parent_values
            .iter()
            .all(|value| value.outcome == GroundTruthOutcome::Loss)
        {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Loss,
                distance: parent_values
                    .iter()
                    .filter_map(|value| value.distance)
                    .max(),
            });
        }
        if parent_values.iter().all(|value| value.outcome.is_known()) {
            Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            })
        } else {
            None
        }
    }

    pub fn write_checkpoint(
        &self,
        path: impl AsRef<Path>,
        rounds: usize,
        stats: RetrogradeStats,
    ) -> io::Result<()> {
        let checkpoint = FrontierCheckpoint {
            schema_version: 1,
            table_family: "pathagon-retrograde-frontier-v1".to_owned(),
            input_nodes: self.len(),
            input_edges: self.edge_count(),
            rounds,
            solved: stats.solved,
            draws: stats.draws,
            unknown: stats.unknown,
            complete_graph: self.nodes.values().all(|node| node.complete),
        };
        atomic_json_write(path.as_ref(), &checkpoint)
    }

    pub fn write_values(
        &self,
        path: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
        stats: RetrogradeStats,
    ) -> io::Result<()> {
        let output = RetrogradeOutput {
            schema_version: 1,
            table_family: "pathagon-retrograde-wdl-v1".to_owned(),
            values: values.clone(),
            stats,
        };
        atomic_json_write(path.as_ref(), &output)
    }
}

pub fn read_nodes(path: impl AsRef<Path>) -> io::Result<RetrogradeGraph> {
    let source = fs::read_to_string(path.as_ref())?;
    let mut graph = RetrogradeGraph::default();
    for (line_number, line) in source.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let node: RetrogradeNode = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.as_ref().display(), line_number + 1),
            )
        })?;
        graph.insert(node).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.as_ref().display(), line_number + 1),
            )
        })?;
    }
    Ok(graph)
}

fn parse_outcome(value: &str) -> Option<GroundTruthOutcome> {
    match value {
        "loss" => Some(GroundTruthOutcome::Loss),
        "draw" => Some(GroundTruthOutcome::Draw),
        "win" => Some(GroundTruthOutcome::Win),
        "unknown" => Some(GroundTruthOutcome::Unknown),
        _ => None,
    }
}

fn atomic_json_write<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(key: &str, children: &[&str]) -> RetrogradeNode {
        RetrogradeNode {
            key: key.to_owned(),
            children: children.iter().map(|child| (*child).to_owned()).collect(),
            complete: true,
            terminal: None,
        }
    }

    #[test]
    fn forced_win_distance_propagates_from_terminal_loss() {
        let mut graph = RetrogradeGraph::default();
        graph.insert(node("root", &["terminal"])).unwrap();
        graph
            .insert(RetrogradeNode {
                key: "terminal".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
            })
            .unwrap();
        let (values, _) = graph.solve();
        assert_eq!(values["root"].outcome, GroundTruthOutcome::Win);
        assert_eq!(values["root"].distance, Some(1));
    }

    #[test]
    fn incomplete_cycle_remains_unknown_but_closed_cycle_is_draw() {
        let mut incomplete = RetrogradeGraph::default();
        incomplete
            .insert(RetrogradeNode {
                complete: false,
                ..node("a", &["a"])
            })
            .unwrap();
        assert!(!incomplete.solve().0.contains_key("a"));

        let mut closed = RetrogradeGraph::default();
        closed.insert(node("a", &["b"])).unwrap();
        closed.insert(node("b", &["a"])).unwrap();
        assert_eq!(closed.solve().0["a"].outcome, GroundTruthOutcome::Draw);

        closed
            .insert(RetrogradeNode {
                complete: false,
                ..node("incomplete", &["incomplete"])
            })
            .unwrap();
        assert_eq!(closed.solve().0["a"].outcome, GroundTruthOutcome::Draw);
        assert!(!closed.solve().0.contains_key("incomplete"));
    }
}
