//! Remove operator — removes properties and labels from nodes/relationships.
//!
//! Implements Cypher `REMOVE n.prop` and `REMOVE n:Label`. For properties,
//! sets the attribute value to `NULL` in the pending batch. For labels,
//! computes the set difference between current and removed labels and
//! records the removal.

use super::OpIter;
use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, orderset::OrderSet, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct RemoveOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    items: &'a Vec<QueryExpr<Variable>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> RemoveOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        items: &'a Vec<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        runtime.pending.borrow_mut().resize(
            runtime.g.borrow().node_cap(),
            runtime.g.borrow().labels_count(),
        );
        Self {
            runtime,
            iter,
            items,
            idx,
        }
    }
}

impl Iterator for RemoveOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.iter.next()? {
            Ok(vars) => self.runtime.remove(self.items, &vars).map(|()| vars),
            Err(e) => Err(e),
        };
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}

impl Runtime {
    pub fn remove(
        &self,
        items: &Vec<QueryExpr<Variable>>,
        vars: &Env,
    ) -> Result<(), String> {
        for item in items {
            let (entity, property, labels) = match item.root().data() {
                ExprIR::Property(property) => (
                    self.run_expr(item, item.root().child(0).idx(), vars, None)?,
                    Some(property),
                    None,
                ),
                ExprIR::FuncInvocation(func) if func.name == "hasLabels" => {
                    let labels = item
                        .root()
                        .child(1)
                        .children()
                        .filter_map(|c| match c.data() {
                            ExprIR::String(label) => Some(label.clone()),
                            _ => None,
                        })
                        .collect::<OrderSet<_>>();

                    (
                        self.run_expr(item, item.root().child(0).idx(), vars, None)?,
                        None,
                        Some(labels),
                    )
                }
                _ => {
                    unreachable!();
                }
            };
            match entity {
                Value::Node(node) => {
                    if (self.g.borrow().is_node_deleted(node)
                        && !self.pending.borrow().is_node_created(node))
                        || self.pending.borrow().is_node_deleted(node)
                    {
                        continue;
                    }
                    if let Some(property) = property {
                        self.pending.borrow_mut().set_node_attribute(
                            node,
                            property.clone(),
                            Value::Null,
                        )?;
                    }
                    if let Some(labels) = labels {
                        let mut current_labels = self
                            .g
                            .borrow()
                            .get_node_label_ids(node)
                            .collect::<OrderSet<_>>();
                        self.pending
                            .borrow()
                            .update_node_labels(node, &mut current_labels);
                        let labels = labels
                            .iter()
                            .filter_map(|l| self.g.borrow_mut().get_label_id(l.as_str()))
                            .filter(|l| current_labels.contains(l))
                            .collect::<Vec<_>>();
                        self.pending.borrow_mut().remove_node_labels(node, &labels);
                    }
                }
                Value::Relationship(rel) => {
                    if let Some(property) = property {
                        self.pending.borrow_mut().set_relationship_attribute(
                            rel.0,
                            property.clone(),
                            Value::Null,
                        )?;
                    }
                    if labels.is_some() {
                        return Err(String::from(
                            "Type mismatch: expected Node but was Relationship",
                        ));
                    }
                }
                Value::Null => {}
                _ => {
                    return Err(format!(
                        "Type mismatch: expected Node or Relationship but was {}",
                        entity.name()
                    ));
                }
            }
        }
        Ok(())
    }
}
