//! Variable-length traverse operator — multi-hop BFS relationship expansion.
//!
//! Implements Cypher patterns like `(a)-[*2..5]->(b)`. For each incoming row,
//! performs a breadth-first search from the source node up to `max_hops` away,
//! yielding result rows for destinations reached at or beyond `min_hops`.
//!
//! ```text
//!  child iter ──► env (with bound start node)
//!                     │
//!       ┌─────────────┴─────────────┐
//!       │  BFS from start node      │
//!       │  hop 1: frontier_0 edges  │──► results if hop >= min_hops
//!       │  hop 2: frontier_1 edges  │──► results if hop >= min_hops
//!       │  ...                      │
//!       │  hop N: stop at max_hops  │
//!       └─────────────┬─────────────┘
//!                     │
//!     for each (start, dest) in range:
//!       env += {from: start, to: dest}
//!                     │
//!                 yield Env ──► parent
//! ```
//!
//! Cycle avoidance is handled via a `visited` set; each node is visited
//! at most once per starting node.

use std::collections::HashSet;
use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx};

pub struct CondVarLenTraverseOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CondVarLenTraverseOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            current: None,
            relationship_pattern,
            idx,
        }
    }
}

impl Iterator for CondVarLenTraverseOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current {
                if let Some(item) = current.next() {
                    self.runtime.inspect_result(self.idx, &item);
                    return Some(item);
                }
                self.current = None;
            }
            let vars = match self.iter.next()? {
                Ok(vars) => vars,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let runtime = self.runtime;
            let relationship_pattern = self.relationship_pattern;

            let from_id = vars
                .get(&relationship_pattern.from.alias)
                .and_then(|v| match v {
                    Value::Node(id) => Some(id),
                    _ => None,
                });
            if from_id.is_none() && vars.is_bound(&relationship_pattern.from.alias) {
                self.current = Some(Box::new(std::iter::empty()));
                continue;
            }
            let to_id = vars
                .get(&relationship_pattern.to.alias)
                .and_then(|v| match v {
                    Value::Node(id) => Some(*id),
                    _ => None,
                });
            if to_id.is_none() && vars.is_bound(&relationship_pattern.to.alias) {
                self.current = Some(Box::new(std::iter::empty()));
                continue;
            }

            let min_hops = relationship_pattern.min_hops.unwrap_or(1);
            let max_hops = relationship_pattern.max_hops.unwrap_or(u32::MAX);
            let bidirectional = relationship_pattern.bidirectional;

            // Get starting nodes
            let start_nodes: Vec<NodeId> = from_id.map_or_else(
                || {
                    runtime
                        .g
                        .borrow()
                        .get_nodes(&relationship_pattern.from.labels, 0)
                        .collect()
                },
                |id| vec![*id],
            );

            let mut results: Vec<Env> = Vec::new();

            for start_node in start_nodes {
                // BFS with visited tracking to avoid cycles
                let mut frontier: Vec<NodeId> = vec![start_node];
                let mut visited: HashSet<NodeId> = HashSet::new();
                visited.insert(start_node);

                for hop in 1..=max_hops {
                    let mut next_frontier_set: HashSet<NodeId> = HashSet::new();
                    let mut next_frontier: Vec<NodeId> = Vec::new();
                    for &current in &frontier {
                        let g = runtime.g.borrow();
                        for (edge_src, edge_dst, _) in g.get_node_relationships(current) {
                            // For directed: only follow outgoing (edge_src == current)
                            // For bidirectional: follow both directions
                            let neighbor = if edge_src == current {
                                Some(edge_dst)
                            } else if bidirectional && edge_dst == current {
                                Some(edge_src)
                            } else {
                                None
                            };
                            if let Some(dest) = neighbor
                                && !visited.contains(&dest)
                            {
                                // Generate result for this edge if within hop range
                                if hop >= min_hops && (to_id.is_none() || to_id == Some(dest)) {
                                    let mut env = vars.clone();
                                    env.insert(
                                        &relationship_pattern.from.alias,
                                        Value::Node(start_node),
                                    );
                                    env.insert(&relationship_pattern.to.alias, Value::Node(dest));
                                    results.push(env);
                                }
                                // Add to frontier (deduplicated for BFS efficiency)
                                if next_frontier_set.insert(dest) {
                                    next_frontier.push(dest);
                                }
                            }
                        }
                    }
                    if next_frontier.is_empty() {
                        break;
                    }
                    // Add frontier nodes to visited AFTER processing all edges at this level
                    for &node in &next_frontier {
                        visited.insert(node);
                    }
                    frontier = next_frontier;
                }
            }

            self.current = Some(Box::new(results.into_iter().map(Ok)));
        }
    }
}
