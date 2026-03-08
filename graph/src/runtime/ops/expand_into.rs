//! Expand-into operator — checks for relationships between two already-bound nodes.
//!
//! Unlike `CondTraverse` which discovers new destination nodes, `ExpandInto`
//! verifies that a relationship exists between two nodes already in the
//! environment (both `from` and `to` are bound). It scans edges between the
//! known endpoints and filters by relationship type and attributes.
//!
//! ```text
//!  child iter ──► env (with bound src AND dst nodes)
//!                     │
//!       ┌─────────────┴─────────────┐
//!       │  scan edges (src -> dst)  │  (+ reverse if bidirectional)
//!       │  filter by type & attrs   │
//!       └─────────────┬─────────────┘
//!                     │
//!        for each matching edge:
//!          env += {rel_alias: relationship}
//!                     │
//!                 yield Env ──► parent
//! ```

use std::iter::empty;
use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ExpandIntoOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ExpandIntoOp<'a> {
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

impl Iterator for ExpandIntoOp<'_> {
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

            let src = match vars.get(&relationship_pattern.from.alias) {
                Some(Value::Node(id)) => *id,
                Some(Value::Null) | None => {
                    self.current = Some(Box::new(empty()));
                    continue;
                }
                _ => {
                    let result = Err(String::from(
                        "Invalid node id for 'from' in relationship pattern",
                    ));
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let dst = match vars.get(&relationship_pattern.to.alias) {
                Some(Value::Node(id)) => *id,
                Some(Value::Null) | None => {
                    self.current = Some(Box::new(empty()));
                    continue;
                }
                _ => {
                    let result = Err(String::from(
                        "Invalid node id for 'to' in relationship pattern",
                    ));
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
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
            // Collect (edge_src, edge_dst) tuples for edge scanning
            // Forward direction: check edges from src to dst
            let mut edge_pairs: Vec<(NodeId, NodeId)> = vec![(src, dst)];
            // For bidirectional traversals, also check reverse direction (dst -> src)
            if relationship_pattern.bidirectional && src != dst {
                edge_pairs.push((dst, src));
            }
            self.current = Some(Box::new(edge_pairs.into_iter().flat_map(
                move |(edge_src, edge_dst)| {
                    let vars = vars.clone();
                    let filter_attrs = filter_attrs.clone();
                    runtime
                        .g
                        .borrow()
                        .get_src_dest_relationships(edge_src, edge_dst, &relationship_pattern.types)
                        .filter(move |id| {
                            // Filter out pending-deleted relationships
                            if runtime
                                .pending
                                .borrow()
                                .is_relationship_deleted(*id, edge_src, edge_dst)
                            {
                                return false;
                            }
                            // Check relationship attributes
                            if let Value::Map(ref filter_attrs) = filter_attrs
                                && !filter_attrs.is_empty()
                            {
                                let g = runtime.g.borrow();
                                for (attr, avalue) in filter_attrs.iter() {
                                    if let Some(pvalue) = g.get_relationship_attribute(*id, attr) {
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
                                Value::Relationship(Box::new((id, edge_src, edge_dst))),
                            );
                            // Keep from/to bindings matching the outer env (src, dst)
                            vars.insert(&relationship_pattern.from.alias, Value::Node(src));
                            vars.insert(&relationship_pattern.to.alias, Value::Node(dst));
                            Ok(vars)
                        })
                },
            )));
        }
    }
}
