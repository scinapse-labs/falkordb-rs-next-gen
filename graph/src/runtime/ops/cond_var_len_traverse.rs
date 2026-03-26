//! Batch-mode variable-length traverse operator — multi-hop relationship expansion.
//!
//! Implements Cypher patterns like `(a)-[*2..5]->(b)`. For each active row
//! in each input batch, enumerates all simple paths (no repeated nodes within
//! a single path) from the source node up to `max_hops` away, yielding result
//! rows for destinations reached at or beyond `min_hops`. Output rows are
//! accumulated into batches of up to `BATCH_SIZE`.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};
use roaring::RoaringTreemap;
use thin_vec::ThinVec;

pub struct CondVarLenTraverseOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<Env<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CondVarLenTraverseOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            pending: VecDeque::new(),
            relationship_pattern,
            idx,
        }
    }

    fn expand_row(
        &self,
        vars: &Env<'a>,
        out: &mut Vec<Env<'a>>,
    ) {
        let relationship_pattern = self.relationship_pattern;

        let from_id = vars
            .get(&relationship_pattern.from.alias)
            .and_then(|v| match v {
                Value::Node(id) => Some(*id),
                _ => None,
            });
        if from_id.is_none() && vars.is_bound(&relationship_pattern.from.alias) {
            return;
        }
        let to_id = vars
            .get(&relationship_pattern.to.alias)
            .and_then(|v| match v {
                Value::Node(id) => Some(*id),
                _ => None,
            });
        if to_id.is_none() && vars.is_bound(&relationship_pattern.to.alias) {
            return;
        }

        let min_hops = relationship_pattern.min_hops.unwrap_or(1);
        let max_hops = relationship_pattern.max_hops.unwrap_or(u32::MAX);
        let bidirectional = relationship_pattern.bidirectional;

        // When `to` is bound but `from` is unbound (e.g. `(:L1)<-[:R1*]-()`)
        // we reverse the traversal: start from the bound `to` node and follow
        // edges in the opposite direction, emitting destinations as `from`.
        let reversed = from_id.is_none() && to_id.is_some() && !bidirectional;

        // Get starting nodes
        let start_nodes: Vec<NodeId> = if reversed {
            vec![to_id.unwrap()]
        } else {
            from_id.map_or_else(
                || {
                    self.runtime
                        .g
                        .borrow()
                        .get_nodes(&relationship_pattern.from.labels, 0)
                        .collect()
                },
                |id| vec![id],
            )
        };

        let dest_labels = if reversed {
            &relationship_pattern.from.labels
        } else {
            &relationship_pattern.to.labels
        };
        let dest_id = if reversed { from_id } else { to_id };

        let g = self.runtime.g.borrow();

        for start_node in start_nodes {
            // Handle 0-hop case: start node itself is a valid result.
            if min_hops == 0 && (dest_id.is_none() || dest_id == Some(start_node)) {
                let mut env = vars.clone_pooled(self.runtime.env_pool);
                env.insert(&relationship_pattern.from.alias, Value::Node(start_node));
                env.insert(&relationship_pattern.to.alias, Value::Node(start_node));
                env.insert(
                    &relationship_pattern.alias,
                    Value::List(Arc::new(ThinVec::new())),
                );
                out.push(env);
            }

            // DFS to enumerate all simple paths (no repeated nodes within a
            // single path). Each stack frame: (node, path, visited_set, hop).
            let mut stack: Vec<(NodeId, ThinVec<Value>, RoaringTreemap)> = Vec::new();
            let mut initial_visited = RoaringTreemap::new();
            initial_visited.insert(u64::from(start_node));
            stack.push((start_node, ThinVec::new(), initial_visited));

            while let Some((current, path, visited)) = stack.pop() {
                let hop = path.len() as u32 + 1;
                if hop > max_hops {
                    continue;
                }

                for (edge_src, edge_dst, edge_id) in
                    g.get_node_relationships_by_type(current, &relationship_pattern.types)
                {
                    let neighbor = if reversed {
                        if edge_dst == current {
                            Some(edge_src)
                        } else {
                            None
                        }
                    } else if edge_src == current {
                        Some(edge_dst)
                    } else if bidirectional && edge_dst == current {
                        Some(edge_src)
                    } else {
                        None
                    };
                    if let Some(dest) = neighbor
                        && !visited.contains(u64::from(dest))
                    {
                        let mut new_path = path.clone();
                        new_path.push(Value::Relationship(Box::new((edge_id, current, dest))));

                        // Emit result if within hop range and destination matches
                        if hop >= min_hops
                            && (dest_id.is_none() || dest_id == Some(dest))
                            && (dest_labels.is_empty()
                                || dest_labels
                                    .iter()
                                    .all(|l| g.get_node_labels(dest).any(|nl| nl == *l)))
                        {
                            let mut env = vars.clone_pooled(self.runtime.env_pool);
                            if reversed {
                                env.insert(&relationship_pattern.from.alias, Value::Node(dest));
                                env.insert(&relationship_pattern.to.alias, Value::Node(start_node));
                            } else {
                                env.insert(
                                    &relationship_pattern.from.alias,
                                    Value::Node(start_node),
                                );
                                env.insert(&relationship_pattern.to.alias, Value::Node(dest));
                            }
                            env.insert(
                                &relationship_pattern.alias,
                                Value::List(Arc::new(new_path.clone())),
                            );
                            out.push(env);
                        }

                        // Continue DFS from dest if we haven't reached max_hops
                        if hop < max_hops {
                            let mut next_visited = visited.clone();
                            next_visited.insert(u64::from(dest));
                            stack.push((dest, new_path, next_visited));
                        }
                    }
                }
            }
        }
    }
}

impl<'a> Iterator for CondVarLenTraverseOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut envs);

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for vars in batch.active_env_iter() {
                let mut expanded = Vec::new();
                self.expand_row(vars, &mut expanded);
                self.pending.extend(expanded);

                super::drain_pending(&mut self.pending, &mut envs);

                if envs.len() >= BATCH_SIZE {
                    break;
                }
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
