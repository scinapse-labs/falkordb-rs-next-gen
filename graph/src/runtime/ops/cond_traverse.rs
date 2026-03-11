//! Conditional traverse operator — expands single-hop relationships.
//!
//! For each incoming row, scans relationships matching the given type/label
//! constraints and property filters. Supports both directed and bidirectional
//! traversal patterns.
//!
//! ```text
//!  child iter ──► env (with bound src node)
//!                     │
//!       ┌─────────────┴─────────────┐
//!       │  scan matching relations  │  (type filter, label filter, attr filter)
//!       └─────────────┬─────────────┘
//!                     │
//!        for each (src, rel, dst):
//!          env += {rel_alias: rel, from: src, to: dst}
//!                     │
//!                 yield Env ──► parent
//! ```
//!
//! When `bidirectional` is set, the reverse direction is also scanned
//! (skipping self-loops to avoid duplicates).

use std::iter::empty;
use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CondTraverseOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CondTraverseOp<'a> {
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

impl Iterator for CondTraverseOp<'_> {
    type Item = Result<Env, String>;

    #[allow(clippy::too_many_lines)]
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

            let filter_attrs = match runtime.run_expr(
                &relationship_pattern.attrs,
                relationship_pattern.attrs.root().idx(),
                &vars,
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let from_node_attrs = match runtime.run_expr(
                &relationship_pattern.from.attrs,
                relationship_pattern.from.attrs.root().idx(),
                &vars,
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let to_node_attrs = match runtime.run_expr(
                &relationship_pattern.to.attrs,
                relationship_pattern.to.attrs.root().idx(),
                &vars,
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let from_id = vars
                .get(&relationship_pattern.from.alias)
                .and_then(|v| match v {
                    Value::Node(id) => Some(*id),
                    _ => None,
                });
            if from_id.is_none() && vars.is_bound(&relationship_pattern.from.alias) {
                self.current = Some(Box::new(empty()));
                continue;
            }
            let to_id = vars
                .get(&relationship_pattern.to.alias)
                .and_then(|v| match v {
                    Value::Node(id) => Some(*id),
                    _ => None,
                });
            if to_id.is_none() && vars.is_bound(&relationship_pattern.to.alias) {
                self.current = Some(Box::new(empty()));
                continue;
            }

            self.current = Some(Box::new(
                Box::new(
                    runtime
                        .g
                        .borrow()
                        .get_relationships(
                            &relationship_pattern.types,
                            &relationship_pattern.from.labels,
                            &relationship_pattern.to.labels,
                        )
                        .map(|(src, dst)| (src, dst, false)),
                )
                .chain(if relationship_pattern.bidirectional {
                    // For bidirectional traversals, also scan the reverse direction
                    // Skip self-loops (src == dst) as they are already included in the forward scan
                    Box::new(
                        runtime
                            .g
                            .borrow()
                            .get_relationships(
                                &relationship_pattern.types,
                                &relationship_pattern.to.labels,
                                &relationship_pattern.from.labels,
                            )
                            .filter(|(src, dst)| src != dst)
                            .map(|(src, dst)| (src, dst, true)),
                    ) as Box<dyn Iterator<Item = (NodeId, NodeId, bool)>>
                } else {
                    Box::new(empty())
                })
                .flat_map(move |(src, dst, is_reverse)| {
                    let (from_node, to_node) = if is_reverse { (dst, src) } else { (src, dst) };
                    if from_id.is_some() && from_id.unwrap() != from_node {
                        return Box::new(empty()) as Box<dyn Iterator<Item = Result<Env, String>>>;
                    }
                    if to_id.is_some() && to_id.unwrap() != to_node {
                        return Box::new(empty()) as Box<dyn Iterator<Item = Result<Env, String>>>;
                    }
                    // Check from node property attributes
                    if let Value::Map(ref attrs) = from_node_attrs
                        && !attrs.is_empty()
                    {
                        let g = runtime.g.borrow();
                        for (attr, avalue) in attrs.iter() {
                            match g.get_node_attribute(from_node, attr) {
                                Some(pvalue) if pvalue == *avalue => {}
                                _ => {
                                    return Box::new(empty())
                                        as Box<dyn Iterator<Item = Result<Env, String>>>;
                                }
                            }
                        }
                    }
                    // Check to node property attributes
                    if let Value::Map(ref attrs) = to_node_attrs
                        && !attrs.is_empty()
                    {
                        let g = runtime.g.borrow();
                        for (attr, avalue) in attrs.iter() {
                            match g.get_node_attribute(to_node, attr) {
                                Some(pvalue) if pvalue == *avalue => {}
                                _ => {
                                    return Box::new(empty())
                                        as Box<dyn Iterator<Item = Result<Env, String>>>;
                                }
                            }
                        }
                    }
                    let vars = vars.clone();
                    let filter_attrs = filter_attrs.clone();
                    Box::new(
                        runtime
                            .g
                            .borrow()
                            .get_src_dest_relationships(src, dst, &relationship_pattern.types)
                            .filter(move |v| {
                                if let Value::Map(filter_attrs) = &filter_attrs
                                    && !filter_attrs.is_empty()
                                {
                                    let g = runtime.g.borrow();
                                    for (attr, avalue) in filter_attrs.iter() {
                                        if let Some(pvalue) = g.get_relationship_attribute(*v, attr)
                                        {
                                            if *avalue == pvalue {
                                                continue;
                                            }
                                            return false;
                                        }
                                        return false;
                                    }
                                }
                                true
                            })
                            .map(move |id| {
                                let mut vars = vars.clone();
                                vars.insert(
                                    &relationship_pattern.alias,
                                    Value::Relationship(Box::new((id, src, dst))),
                                );
                                vars.insert(
                                    &relationship_pattern.from.alias,
                                    Value::Node(from_node),
                                );
                                vars.insert(&relationship_pattern.to.alias, Value::Node(to_node));
                                Ok(vars)
                            }),
                    ) as Box<dyn Iterator<Item = Result<Env, String>>>
                }),
            ));
        }
    }
}
