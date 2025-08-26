use std::{cell::RefCell, collections::HashMap, sync::Arc};

use ordermap::{OrderMap, OrderSet};
use roaring::RoaringTreemap;

use crate::{
    graph::graph::{Graph, NodeId, RelationshipId},
    runtime::{
        functions::Type,
        runtime::QueryStatistics,
        value::{Value, ValueTypeOf},
    },
};

pub struct PendingRelationship {
    pub from: NodeId,
    pub to: NodeId,
    pub type_name: Arc<String>,
}

impl PendingRelationship {
    #[must_use]
    pub const fn new(
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) -> Self {
        Self {
            from,
            to,
            type_name,
        }
    }
}

#[derive(Default)]
pub struct Pending {
    created_nodes: RoaringTreemap,
    created_relationships: OrderMap<RelationshipId, PendingRelationship>,
    deleted_nodes: RoaringTreemap,
    deleted_relationships: OrderSet<(RelationshipId, NodeId, NodeId)>,
    set_nodes_attrs: OrderMap<NodeId, OrderMap<Arc<String>, Value>>,
    set_relationships_attrs: OrderMap<RelationshipId, OrderMap<Arc<String>, Value>>,
    set_node_labels: OrderMap<NodeId, OrderSet<Arc<String>>>,
    remove_node_labels: OrderMap<NodeId, OrderSet<Arc<String>>>,
    index_add_docs: HashMap<Arc<String>, RoaringTreemap>,
    index_remove_docs: HashMap<Arc<String>, RoaringTreemap>,
}

impl Pending {
    pub fn created_node(
        &mut self,
        id: NodeId,
    ) {
        self.created_nodes.insert(id.into());
    }

    pub fn set_node_attributes(
        &mut self,
        id: NodeId,
        attrs: OrderMap<Arc<String>, Value>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            if value
                .value_of_type(&Type::Union(vec![
                    Type::Bool,
                    Type::Int,
                    Type::Float,
                    Type::String,
                    Type::Null,
                    Type::List(Box::new(Type::Union(vec![
                        Type::Bool,
                        Type::Int,
                        Type::Float,
                        Type::String,
                    ]))),
                ]))
                .is_some()
            {
                return Err(
                    "Property values can only be of primitive types or arrays of primitive types",
                )?;
            }
        }
        self.set_nodes_attrs.insert(id, attrs);
        Ok(())
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        if value
            .value_of_type(&Type::Union(vec![
                Type::Bool,
                Type::Int,
                Type::Float,
                Type::String,
                Type::Null,
                Type::List(Box::new(Type::Union(vec![
                    Type::Bool,
                    Type::Int,
                    Type::Float,
                    Type::String,
                ]))),
            ]))
            .is_some()
        {
            return Err(
                "Property values can only be of primitive types or arrays of primitive types",
            )?;
        }
        self.set_nodes_attrs
            .entry(id)
            .or_default()
            .insert(key, value);
        Ok(())
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.set_nodes_attrs
            .get(&id)
            .and_then(|attrs| attrs.get(key))
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        if let Some(added) = self.set_nodes_attrs.get(&id) {
            for (key, value) in added {
                if *value == Value::Null {
                    attrs.remove(key);
                } else {
                    attrs.insert(key.clone(), value.clone());
                }
            }
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: OrderSet<Arc<String>>,
    ) {
        self.set_node_labels.insert(id, labels);
    }

    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: OrderSet<Arc<String>>,
    ) {
        self.remove_node_labels.insert(id, labels);
    }

    pub fn update_node_labels(
        &self,
        id: NodeId,
        labels: &mut OrderSet<Arc<String>>,
    ) {
        if let Some(added) = self.set_node_labels.get(&id) {
            labels.extend(added.iter().cloned());
        }
        if let Some(removed) = self.remove_node_labels.get(&id) {
            for label in removed {
                labels.remove(label);
            }
        }
    }

    pub fn deleted_node(
        &mut self,
        id: NodeId,
    ) {
        self.deleted_nodes.insert(id.into());
    }

    pub fn created_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) {
        self.created_relationships
            .insert(id, PendingRelationship::new(from, to, type_name));
    }

    pub fn set_relationship_attributes(
        &mut self,
        id: RelationshipId,
        attrs: OrderMap<Arc<String>, Value>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            if value
                .value_of_type(&Type::Union(vec![
                    Type::Bool,
                    Type::Int,
                    Type::Float,
                    Type::String,
                    Type::Null,
                    Type::List(Box::new(Type::Union(vec![
                        Type::Bool,
                        Type::Int,
                        Type::Float,
                        Type::String,
                    ]))),
                ]))
                .is_some()
            {
                return Err(
                    "Property values can only be of primitive types or arrays of primitive types",
                )?;
            }
        }
        self.set_relationships_attrs.insert(id, attrs);
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        if value
            .value_of_type(&Type::Union(vec![
                Type::Bool,
                Type::Int,
                Type::Float,
                Type::String,
                Type::Null,
                Type::List(Box::new(Type::Union(vec![
                    Type::Bool,
                    Type::Int,
                    Type::Float,
                    Type::String,
                ]))),
            ]))
            .is_some()
        {
            return Err(
                "Property values can only be of primitive types or arrays of primitive types",
            )?;
        }
        self.set_relationships_attrs
            .entry(id)
            .or_default()
            .insert(key, value);
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.set_relationships_attrs
            .get(&id)
            .and_then(|attrs| attrs.get(key))
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        if let Some(added) = self.set_relationships_attrs.get(&id) {
            for (key, value) in added {
                if *value == Value::Null {
                    attrs.remove(key);
                } else {
                    attrs.insert(key.clone(), value.clone());
                }
            }
        }
    }

    pub fn deleted_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
    ) {
        self.deleted_relationships.insert((id, from, to));
    }

    #[must_use]
    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        self.created_relationships
            .get(&id)
            .map(|r| r.type_name.clone())
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.into())
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
    ) -> bool {
        self.deleted_relationships.contains(&(id, from, to))
    }

    pub fn commit(
        &mut self,
        g: &RefCell<Graph>,
        stats: &RefCell<QueryStatistics>,
    ) {
        if !self.created_nodes.is_empty() {
            stats.borrow_mut().nodes_created += self.created_nodes.len();
            g.borrow_mut().create_nodes(&self.created_nodes);
            self.created_nodes.clear();
        }
        if !self.created_relationships.is_empty() {
            stats.borrow_mut().relationships_created += self.created_relationships.len();
            g.borrow_mut()
                .create_relationships(&self.created_relationships);
            self.created_relationships.clear();
        }
        if !self.deleted_relationships.is_empty() {
            stats.borrow_mut().relationships_deleted += self.deleted_relationships.len();
            g.borrow_mut()
                .delete_relationships(self.deleted_relationships.clone());
            self.deleted_relationships.clear();
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            for id in &self.deleted_nodes {
                g.borrow_mut()
                    .delete_node(NodeId::from(id), &mut self.index_remove_docs);
            }
            self.deleted_nodes.clear();
        }
        if !self.set_node_labels.is_empty() {
            for (id, labels) in &self.set_node_labels {
                g.borrow_mut()
                    .set_node_labels(*id, labels, &mut self.index_add_docs);
            }
            self.set_node_labels.clear();
        }
        if !self.remove_node_labels.is_empty() {
            for (id, labels) in &self.remove_node_labels {
                g.borrow_mut()
                    .remove_node_labels(*id, labels, &mut self.index_remove_docs);
            }
            self.remove_node_labels.clear();
        }
        if !self.set_nodes_attrs.is_empty() {
            stats.borrow_mut().properties_set += self
                .set_nodes_attrs
                .values()
                .flat_map(|v| v.values())
                .map(|v| match *v {
                    Value::Null => 0,
                    _ => 1,
                })
                .sum::<usize>();
            for (id, attrs) in &self.set_nodes_attrs {
                for (key, value) in attrs {
                    let attr_id = g.borrow_mut().get_or_add_node_attribute_id(key);
                    if g.borrow_mut().set_node_attribute(
                        *id,
                        attr_id,
                        value.clone(),
                        &mut self.index_add_docs,
                        &mut self.index_remove_docs,
                    ) {
                        stats.borrow_mut().properties_removed += 1;
                    }
                }
            }
            self.set_nodes_attrs.clear();
        }
        if !self.set_relationships_attrs.is_empty() {
            stats.borrow_mut().properties_set += self
                .set_relationships_attrs
                .values()
                .flat_map(|v| v.values())
                .map(|v| match *v {
                    Value::Null => 0,
                    _ => 1,
                })
                .sum::<usize>();
            for (id, attrs) in &self.set_relationships_attrs {
                for (key, value) in attrs {
                    let attr_id = g.borrow_mut().get_or_add_relationship_attribute_id(key);
                    if g.borrow_mut()
                        .set_relationship_attribute(*id, attr_id, value.clone())
                    {
                        stats.borrow_mut().properties_removed += 1;
                    }
                }
            }
            self.set_relationships_attrs.clear();
        }
        g.borrow_mut()
            .commit_index(&mut self.index_add_docs, &mut self.index_remove_docs);
    }
}
