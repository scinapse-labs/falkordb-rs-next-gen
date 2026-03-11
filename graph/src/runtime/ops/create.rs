//! Create operator — creates new nodes and relationships in the graph.
//!
//! Implements Cypher `CREATE (n:Label {props})-[:REL]->(m)`. For each
//! incoming row, reserves new node/relationship IDs and records the
//! mutations in the pending batch (applied later by `CommitOp`).
//!
//! ```text
//!  child iter ──► env
//!                  │
//!    ┌─────────────┴─────────────┐
//!    │  for each node in pattern │──► reserve ID, set labels & attrs
//!    │  for each rel in pattern  │──► reserve ID, set type & attrs
//!    │  env += new bindings      │
//!    └─────────────┬─────────────┘
//!                  │
//!              yield Env ──► parent
//! ```
//!
//! Label strings are resolved to numeric `LabelId`s lazily on the first
//! row (cached via `OnceCell`). When the direct parent is a `Commit` node
//! at the plan root, result rows are suppressed (write-only optimization).

use std::cell::OnceCell;
use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::LabelId;
use crate::parser::ast::{QueryGraph, QueryNode, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CreateOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    pattern: QueryGraph<Arc<String>, Arc<String>, Variable>,
    resolved_pattern: OnceCell<QueryGraph<Arc<String>, LabelId, Variable>>,
    parent_commit: bool,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CreateOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let parent_commit = if let Some(parent) = runtime.plan.node(idx).parent()
            && matches!(parent.data(), IR::Commit)
            && parent.parent().is_none()
        {
            true
        } else {
            false
        };

        Self {
            runtime,
            iter,
            pattern: pattern.clone(),
            resolved_pattern: OnceCell::new(),
            parent_commit,
            idx,
        }
    }
}

impl Iterator for CreateOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let result = match self.iter.next()? {
                Ok(mut vars) => {
                    let resolved_pattern = self.resolved_pattern.get_or_init(|| {
                        let resolved = self.runtime.resolve_pattern(&self.pattern);
                        self.runtime.pending.borrow_mut().resize(
                            self.runtime.g.borrow().node_cap(),
                            self.runtime.g.borrow().labels_count(),
                        );
                        resolved
                    });
                    match self.runtime.create(resolved_pattern, &mut vars) {
                        Ok(()) => {
                            if self.parent_commit {
                                continue;
                            }
                            Ok(vars)
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            };
            self.runtime.inspect_result(self.idx, &result);
            return Some(result);
        }
    }
}

impl Runtime {
    pub fn resolve_pattern(
        &self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    ) -> QueryGraph<Arc<String>, LabelId, Variable> {
        let mut resolved_pattern = QueryGraph::default();
        for node in pattern.nodes() {
            resolved_pattern.add_node(Arc::new(QueryNode::new(
                node.alias.clone(),
                node.labels
                    .iter()
                    .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                    .collect(),
                node.attrs.clone(),
            )));
        }
        for rel in pattern.relationships() {
            resolved_pattern.add_relationship(Arc::new(QueryRelationship::new(
                rel.alias.clone(),
                rel.types.clone(),
                rel.attrs.clone(),
                Arc::new(QueryNode::new(
                    rel.from.alias.clone(),
                    rel.from
                        .labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                    rel.from.attrs.clone(),
                )),
                Arc::new(QueryNode::new(
                    rel.to.alias.clone(),
                    rel.to
                        .labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                    rel.to.attrs.clone(),
                )),
                rel.bidirectional,
                rel.min_hops,
                rel.max_hops,
            )));
        }
        for path in pattern.paths() {
            resolved_pattern.add_path(path.clone());
        }
        resolved_pattern
    }

    pub fn create(
        &self,
        pattern: &QueryGraph<Arc<String>, LabelId, Variable>,
        vars: &mut Env,
    ) -> Result<(), String> {
        for node in pattern.nodes() {
            let id = self.g.borrow_mut().reserve_node();
            {
                let mut pending = self.pending.borrow_mut();
                pending.created_node(id);
                pending.set_node_labels(id, &node.labels);
            }

            let attrs = self.run_expr(&node.attrs, node.attrs.root().idx(), vars, None)?;
            match attrs {
                Value::Map(attrs) => {
                    self.pending
                        .borrow_mut()
                        .set_node_attributes(id, Arc::unwrap_or_clone(attrs))?;
                }
                _ => unreachable!(),
            }
            vars.insert(&node.alias, Value::Node(id));
        }
        for rel in pattern.relationships() {
            let (from_id, to_id) = {
                let Value::Node(from_id) = vars
                    .get(&rel.from.alias)
                    .ok_or_else(|| format!("Variable {} not found", rel.from.alias.as_str()))?
                    .clone()
                else {
                    return Err(String::from("Invalid node id"));
                };
                let Value::Node(to_id) = vars
                    .get(&rel.to.alias)
                    .ok_or_else(|| format!("Variable {} not found", rel.to.alias.as_str()))?
                    .clone()
                else {
                    return Err(String::from("Invalid node id"));
                };
                (from_id, to_id)
            };

            {
                let g = self.g.borrow();
                let pending = self.pending.borrow();
                if (g.is_node_deleted(from_id) && !pending.is_node_created(from_id))
                    || pending.is_node_deleted(from_id)
                    || (g.is_node_deleted(to_id) && !pending.is_node_created(to_id))
                    || pending.is_node_deleted(to_id)
                {
                    return Err(String::from(
                        "Failed to create relationship; endpoint was not found.",
                    ));
                }
            }
            let id = self.g.borrow_mut().reserve_relationship();
            self.pending.borrow_mut().created_relationship(
                id,
                from_id,
                to_id,
                rel.types.first().unwrap().clone(),
            );
            let attrs = self.run_expr(&rel.attrs, rel.attrs.root().idx(), vars, None)?;
            match attrs {
                Value::Map(attrs) => {
                    self.pending
                        .borrow_mut()
                        .set_relationship_attributes(id, Arc::unwrap_or_clone(attrs))?;
                }
                _ => {
                    return Err(String::from("Invalid relationship properties"));
                }
            }
            vars.insert(
                &rel.alias,
                Value::Relationship(Box::new((id, from_id, to_id))),
            );
        }
        Ok(())
    }
}
