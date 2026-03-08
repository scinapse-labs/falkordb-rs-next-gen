//! Delete operator — marks nodes and relationships for deletion.
//!
//! Implements Cypher `DELETE n` / `DETACH DELETE n`. For each incoming row,
//! evaluates the delete expressions and records deletions in the pending
//! batch. Node deletions cascade: all connected relationships are also
//! marked for deletion. Deleted entity metadata (labels, attributes) is
//! captured for statistics reporting.
//!
//! ```text
//!  child iter ──► env
//!                  │
//!    ┌─────────────┴──────────────┐
//!    │  for each delete expr:     │
//!    │    Node(id)  ──► delete    │──► also delete connected relationships
//!    │    Rel(id)   ──► delete    │
//!    │    Path(vs)  ──► recurse   │
//!    └─────────────┬──────────────┘
//!                  │
//!              yield Env ──► parent
//! ```

use super::OpIter;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    env::Env,
    runtime::Runtime,
    value::{DeletedNode, DeletedRelationship, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct DeleteOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    trees: &'a Vec<QueryExpr<Variable>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> DeleteOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        trees: &'a Vec<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            trees,
            idx,
        }
    }
}

impl Iterator for DeleteOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.iter.next()? {
            Ok(vars) => self.runtime.delete(self.trees, &vars).map(|()| vars),
            Err(e) => Err(e),
        };
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}

impl Runtime {
    pub fn delete(
        &self,
        trees: &Vec<QueryExpr<Variable>>,
        vars: &Env,
    ) -> Result<(), String> {
        for tree in trees {
            let value = self.run_expr(tree, tree.root().idx(), vars, None)?;
            self.delete_entity(value)?;
        }
        Ok(())
    }

    pub fn delete_entity(
        &self,
        value: Value,
    ) -> Result<(), String> {
        match value {
            Value::Node(id) => {
                if !self.g.borrow().is_node_deleted(id) {
                    for (src, dest, id) in self.g.borrow().get_node_relationships(id) {
                        self.pending
                            .borrow_mut()
                            .deleted_relationship(id, src, dest);
                    }
                    self.pending.borrow_mut().deleted_node(id);
                    let labels = self.g.borrow().get_node_label_ids(id).collect();
                    let attrs = self.get_node_attrs(id).collect();
                    self.deleted_nodes
                        .borrow_mut()
                        .insert(id, DeletedNode::new(labels, attrs));
                }
            }
            Value::Relationship(rel) => {
                if !self.g.borrow().is_relationship_deleted(rel.0) {
                    self.pending
                        .borrow_mut()
                        .deleted_relationship(rel.0, rel.1, rel.2);
                    let type_id = self.g.borrow().get_relationship_type_id(rel.0);
                    let attrs = self.get_relationship_attrs(rel.0).collect();
                    self.deleted_relationships
                        .borrow_mut()
                        .insert(rel.0, DeletedRelationship::new(type_id, attrs));
                }
            }
            Value::Path(values) => {
                for value in values {
                    self.delete_entity(value)?;
                }
            }
            Value::Null => {}
            _ => {
                return Err(String::from(
                    "Delete type mismatch, expecting either Node or Relationship.",
                ));
            }
        }
        Ok(())
    }
}
