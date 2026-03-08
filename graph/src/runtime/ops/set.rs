//! Set operator — updates properties and labels on nodes/relationships.
//!
//! Implements Cypher `SET n.prop = value`, `SET n = {map}`, `SET n:Label`,
//! and `SET n += {map}`. Supports property-level updates, full entity
//! replacement, and label assignment. Label strings are lazily resolved
//! to `LabelId`s on the first row. Changes are recorded in the pending
//! batch for later commit.

use std::sync::Arc;

use once_cell::unsync::OnceCell;

use super::OpIter;
use crate::graph::graph::LabelId;
use crate::parser::ast::{ExprIR, SetItem, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct SetOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    items: &'a [SetItem<Arc<String>, Variable>],
    resolved_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SetOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        items: &'a [SetItem<Arc<String>, Variable>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            items,
            resolved_items: OnceCell::new(),
            idx,
        }
    }
}

impl Iterator for SetOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.iter.next()? {
            Ok(vars) => {
                let resolved = self.resolved_items.get_or_init(|| {
                    let items = self.runtime.resolve_set_items(self.items);
                    self.runtime.pending.borrow_mut().resize(
                        self.runtime.g.borrow().node_cap(),
                        self.runtime.g.borrow().labels_count(),
                    );
                    items
                });
                self.runtime.set(resolved, &vars).map(|()| vars)
            }
            Err(e) => Err(e),
        };
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}

impl Runtime {
    pub fn resolve_set_items(
        &self,
        items: &[SetItem<Arc<String>, Variable>],
    ) -> Vec<SetItem<LabelId, Variable>> {
        items
            .iter()
            .map(|item| match item {
                SetItem::Label(var, labels) => SetItem::Label(
                    var.clone(),
                    labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                ),
                SetItem::Attribute(entity, value, replace) => {
                    SetItem::Attribute(entity.clone(), value.clone(), *replace)
                }
            })
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    pub fn set(
        &self,
        items: &Vec<SetItem<LabelId, Variable>>,
        vars: &Env,
    ) -> Result<(), String> {
        for item in items {
            match item {
                SetItem::Attribute(entity, value, replace) => {
                    let run_expr = self.run_expr(value, value.root().idx(), vars, None)?;
                    let (entity, attr) = match entity.root().data() {
                        ExprIR::Variable(name) => {
                            let entity = vars
                                .get(name)
                                .ok_or_else(|| format!("Variable {} not found", name.as_str()))?
                                .clone();
                            (entity, None)
                        }
                        ExprIR::Property(property) => (
                            self.run_expr(entity, entity.root().child(0).idx(), vars, None)?,
                            Some(property),
                        ),
                        _ => {
                            unreachable!();
                        }
                    };
                    match entity {
                        Value::Node(id) => {
                            if (self.g.borrow().is_node_deleted(id)
                                && !self.pending.borrow().is_node_created(id))
                                || self.pending.borrow().is_node_deleted(id)
                            {
                                continue;
                            }
                            if let Some(attr) = attr {
                                if let Some(v) = self.get_node_attribute(id, attr)
                                    && v == run_expr
                                {
                                    continue;
                                }

                                self.pending.borrow_mut().set_node_attribute(
                                    id,
                                    attr.clone(),
                                    run_expr,
                                )?;
                            } else {
                                match run_expr {
                                    Value::Map(map) => {
                                        if *replace {
                                            self.pending.borrow_mut().clear_node_attributes(id);
                                            for key in self.g.borrow().get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in map.iter() {
                                            self.pending.borrow_mut().set_node_attribute(
                                                id,
                                                key.clone(),
                                                value.clone(),
                                            )?;
                                        }
                                    }
                                    Value::Node(tid) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_node_attrs(tid);
                                        if *replace {
                                            for key in g.get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            self.pending
                                                .borrow_mut()
                                                .set_node_attribute(id, key, value)?;
                                        }
                                    }
                                    Value::Relationship(rel) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_relationship_attrs(rel.0);
                                        if *replace {
                                            for key in g.get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            self.pending
                                                .borrow_mut()
                                                .set_node_attribute(id, key, value)?;
                                        }
                                    }
                                    _ => {
                                        return Err("Property values can only be of primitive types or arrays of primitive types".to_string());
                                    }
                                }
                            }
                        }
                        Value::Relationship(target_rel) => {
                            if self.g.borrow().is_relationship_deleted(target_rel.0)
                                || self.pending.borrow().is_relationship_deleted(
                                    target_rel.0,
                                    target_rel.1,
                                    target_rel.2,
                                )
                            {
                                continue;
                            }
                            if let Some(attr) = attr {
                                if let Some(v) = self.get_relationship_attribute(target_rel.0, attr)
                                    && v == run_expr
                                {
                                    continue;
                                }

                                self.pending.borrow_mut().set_relationship_attribute(
                                    target_rel.0,
                                    attr.clone(),
                                    run_expr,
                                )?;
                            } else {
                                match run_expr {
                                    Value::Map(map) => {
                                        let g = self.g.borrow();
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in map.iter() {
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key.clone(),
                                                value.clone(),
                                            )?;
                                        }
                                    }
                                    Value::Node(sid) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_node_attrs(sid);
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key,
                                                value,
                                            )?;
                                        }
                                    }
                                    Value::Relationship(source_rel) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_relationship_attrs(source_rel.0);
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key,
                                                value,
                                            )?;
                                        }
                                    }
                                    _ => {
                                        return Err("Property values can only be of primitive types or arrays of primitive types".to_string());
                                    }
                                }
                            }
                        }
                        // Silently ignore SET on Null and non-entity types
                        // (e.g. Path), matching C FalkorDB behavior.
                        _ => {}
                    }
                }
                SetItem::Label(entity, labels) => {
                    let run_expr = vars.get(entity);
                    match run_expr {
                        Some(Value::Node(id)) => {
                            if (self.g.borrow().is_node_deleted(*id)
                                && !self.pending.borrow().is_node_created(*id))
                                || self.pending.borrow().is_node_deleted(*id)
                            {
                                continue;
                            }
                            self.pending.borrow_mut().set_node_labels(*id, labels);
                        }
                        Some(Value::Null) => {}
                        _ => {
                            return Err(format!(
                                "Type mismatch: expected Node but was {}",
                                run_expr.map_or_else(|| "undefined".to_string(), Value::name)
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
