//! Property storage for graph entities.
//!
//! This module provides [`AttributeStore`], a columnar store for node and
//! relationship properties. Each property name maps to a separate [`BlockVec`]
//! that stores values indexed by entity ID.
//!
//! ## Design
//!
//! ```text
//! AttributeStore
//!    ├── attrs_name: ["name", "age", "email"]  (property name → index)
//!    └── attributes: [BlockVec, BlockVec, BlockVec]  (one per property)
//!                         │
//!                    [None, "Alice", "Bob", None, "Carol"]
//!                           ↑         ↑           ↑
//!                        node 1    node 2      node 4
//! ```
//!
//! ## Columnar vs Row Storage
//!
//! Columnar storage is chosen because:
//! - Queries often access few properties across many nodes
//! - Sparse storage is efficient when many entities lack certain properties
//! - MVCC versioning can be done per-column

use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::{
    graph::block_vec::BlockVec,
    runtime::{orderset::OrderSet, value::Value},
};

/// Columnar property storage for graph entities.
///
/// Stores properties in a column-oriented layout where each property name
/// maps to a sparse vector of values indexed by entity ID.
#[derive(Clone, Default)]
pub struct AttributeStore {
    /// Column data: one BlockVec per property, indexed by attr position
    attributes: Arc<AtomicRefCell<Vec<BlockVec<Value>>>>,
    /// Property names in insertion order (name → column index)
    pub attrs_name: OrderSet<Arc<String>>,
}

impl AttributeStore {
    pub fn remove(
        &mut self,
        key: u64,
    ) {
        for attr in self.attributes.borrow_mut().iter_mut() {
            attr.remove(key);
        }
    }

    #[must_use]
    pub fn get_attr(
        &self,
        key: u64,
        attr: &Arc<String>,
    ) -> Option<Value> {
        if let Some(idx) = self.attrs_name.get_index_of(attr) {
            return self.attributes.borrow()[idx].get(key);
        }
        None
    }

    #[must_use]
    pub fn has_attributes(
        &self,
        key: u64,
    ) -> bool {
        for attr in self.attributes.borrow().iter() {
            if attr.exists(key) {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn get_attrs(
        &self,
        key: u64,
    ) -> Vec<Arc<String>> {
        let mut ids = vec![];
        for (i, attr) in self.attributes.borrow().iter().enumerate() {
            if attr.exists(key) {
                ids.push(self.attrs_name[i].clone());
            }
        }
        ids
    }

    pub fn remove_attr(
        &mut self,
        key: u64,
        attr: &Arc<String>,
    ) -> bool {
        if let Some(idx) = self.attrs_name.get_index_of(attr)
            && self.attributes.borrow_mut()[idx].remove(key).is_some()
        {
            true
        } else {
            false
        }
    }

    pub fn insert_attr(
        &mut self,
        key: u64,
        attr: &Arc<String>,
        value: Value,
    ) -> bool {
        let mut attributes = self.attributes.borrow_mut();
        let idx = self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
            attributes.push(BlockVec::new(1024));
            self.attrs_name.insert(attr.clone());
            attributes.len() - 1
        });
        attributes[idx].insert(key, value)
    }

    #[must_use]
    pub fn get_attr_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.attrs_name.get_index_of(attr)
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        Self {
            attributes: Arc::new(AtomicRefCell::new(
                self.attributes
                    .borrow()
                    .iter()
                    .map(BlockVec::new_version)
                    .collect(),
            )),
            attrs_name: self.attrs_name.clone(),
        }
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.attributes
            .borrow()
            .iter()
            .map(BlockVec::memory_usage)
            .sum()
    }
}

unsafe impl Send for AttributeStore {}
unsafe impl Sync for AttributeStore {}
