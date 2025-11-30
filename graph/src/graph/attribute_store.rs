use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::{
    graph::block_vec::BlockVec,
    runtime::{orderset::OrderSet, value::Value},
};

#[derive(Clone, Default)]
pub struct AttributeStore {
    attributes: Arc<AtomicRefCell<Vec<BlockVec<Value>>>>,
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
    ) -> Option<Vec<Arc<String>>> {
        let mut ids = vec![];
        for (i, attr) in self.attributes.borrow().iter().enumerate() {
            if attr.exists(key) {
                ids.push(self.attrs_name[i].clone());
            }
        }
        if ids.is_empty() { None } else { Some(ids) }
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
        let idx = self.attrs_name.get_index_of(attr).map_or_else(
            || {
                attributes.push(BlockVec::new(1024));
                self.attrs_name.insert(attr.clone());
                attributes.len() - 1
            },
            |idx| idx,
        );
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
}

unsafe impl Send for AttributeStore {}
unsafe impl Sync for AttributeStore {}
