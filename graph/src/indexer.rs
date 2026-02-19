//! Index management for property-based lookups.
//!
//! This module provides indexing capabilities for graph properties using RediSearch
//! as the underlying index engine. Supports:
//!
//! ## Index Types
//!
//! - **Range**: B-tree index for numeric comparisons (=, <, >, range queries)
//! - **Fulltext**: Text search with tokenization and stemming
//! - **Vector**: Vector similarity search for embeddings
//!
//! ## Architecture
//!
//! ```text
//! Indexer
//!    │
//!    ├── label_fields: Map<Label, Map<Attr, Field>>
//!    │      (Defines which properties are indexed)
//!    │
//!    └── rs_index: Map<Label, RSIndex>
//!           (RediSearch index handles)
//! ```
//!
//! ## Index Queries
//!
//! The [`IndexQuery`] enum represents different query types:
//! - `Equal(attr, value)`: Exact match
//! - `Range(attr, min, max)`: Range query
//! - `Prefix(attr, prefix)`: Prefix search
//! - `Contains(attr, substring)`: Substring search
//! - `Fulltext(query)`: Full-text search
//! - `VectorRange`: Vector similarity search

use std::{
    collections::HashMap,
    ffi::CString,
    sync::{Arc, RwLock},
};

use roaring::RoaringTreemap;

pub use crate::index::{
    Document, EntityType, Field, IndexInfo, IndexQuery, IndexStatus, IndexType, TextIndexOptions,
};
use crate::{index::Index, runtime::value::Value};

pub enum IndexOptions {
    Text(TextIndexOptions),
}

impl IndexOptions {
    /// Extract language from the options (only applicable for Text index options).
    pub fn language(&self) -> &Option<Arc<String>> {
        match self {
            IndexOptions::Text(opts) => opts.language(),
        }
    }

    /// Extract stopwords from the options (only applicable for Text index options).
    pub fn stopwords(&self) -> &Option<Vec<Arc<String>>> {
        match self {
            IndexOptions::Text(opts) => opts.stopwords(),
        }
    }

    /// Extract per-field text options (weight, nostem, phonetic).
    pub fn field_options(&self) -> Option<TextIndexOptions> {
        match self {
            IndexOptions::Text(opts) => {
                if opts.weight().is_some() || opts.nostem().is_some() || opts.phonetic().is_some() {
                    Some(TextIndexOptions::new(
                        opts.weight(),
                        opts.nostem(),
                        opts.phonetic(),
                        None,
                        None,
                    ))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct Indexer {
    index: Arc<RwLock<HashMap<Arc<String>, Index>>>,
}

unsafe impl Send for Indexer {}
unsafe impl Sync for Indexer {}

impl Indexer {
    #[must_use]
    pub fn has_indices(&self) -> bool {
        !self.index.read().unwrap().is_empty()
    }

    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        label: Arc<String>,
        attrs: &Vec<Arc<String>>,
        total: u64,
        options: Option<IndexOptions>,
    ) -> Result<(), String> {
        let mut index = self.index.write().unwrap();
        let label_indexes = index.entry(label.clone()).or_default();

        let (language, stopwords, field_options) = match options {
            Some(IndexOptions::Text(text_opts)) => {
                let language = text_opts.language().clone();
                let stopwords = text_opts.stopwords().clone();
                (language, stopwords, Some(text_opts))
            }
            None => (None, None, None),
        };

        // Validate language/stopwords are not already set for existing fulltext indexes
        let has_fulltext = label_indexes.has_fulltext_field();

        if has_fulltext {
            if language.is_some() {
                return Err(format!(
                    "Can not override index configuration: Language is already set for label '{label}'"
                ));
            }

            if stopwords.is_some() {
                return Err(format!(
                    "Can not override index configuration: Stopwords are already set for label '{label}'"
                ));
            }
        }

        // For now, field_options is match against full text indexes only
        if field_options.is_some() && *index_type != IndexType::Fulltext {
            return Err("Text index options are only valid for fulltext indexes".into());
        }

        for attr in attrs {
            let field_name = match index_type {
                IndexType::Range => Arc::new(format!("range:{attr}")),
                IndexType::Fulltext => attr.clone(),
                IndexType::Vector => Arc::new(format!("vector:{attr}")),
                IndexType::Point => Arc::new(format!("point:{attr}")),
            };

            if label_indexes.has_field_with_type(attr, index_type) {
                return Err(format!("Attribute '{attr}' is already indexed"));
            }

            let field = Arc::new(Field::new(
                CString::new(field_name.as_str()).map_err(|e| e.to_string())?,
                index_type.clone(),
                field_options.clone(),
            ));

            if label_indexes.contains_field(attr) {
                label_indexes.add_field_to_existing(attr, field);
            } else {
                label_indexes.insert_field(attr.clone(), field);
            }
        }
        let fields = label_indexes.clone_fields();

        if !label_indexes.has_rs_index() {
            let effective_stopwords = stopwords
                .clone()
                .or_else(|| label_indexes.stopwords().cloned());
            let effective_language = language
                .clone()
                .or_else(|| label_indexes.language().cloned());
            label_indexes.create_rs_index(
                label.clone(),
                effective_stopwords.as_ref(),
                effective_language.as_ref(),
            )?;
        }

        label_indexes.register_fields(&fields, field_options.as_ref());

        // Update the label indexes with global settings
        if language.is_some() && label_indexes.language().is_none() {
            label_indexes.set_language(language);
        }
        if stopwords.is_some() && label_indexes.stopwords().is_none() {
            label_indexes.set_stopwords(stopwords);
        }

        label_indexes.set_under_construction(0, total);
        label_indexes.increment_pending();
        Ok(())
    }

    pub fn drop_index(
        &mut self,
        label: Arc<String>,
        attrs: &Vec<Arc<String>>,
        index_type: &IndexType,
        total: u64,
    ) -> Option<(usize, usize)> {
        let mut index = self.index.write().unwrap();
        if let Some(index) = index.get_mut(&label) {
            let number_of_indexes = index.index_count();
            let mut removed = false;
            for attr in attrs {
                let (has_type, field_count) = if let Some(fields) = index.get_fields(attr) {
                    (fields.iter().any(|f| f.ty == *index_type), fields.len())
                } else {
                    continue;
                };
                if has_type {
                    if field_count == 1 {
                        index.remove_field(attr);
                        removed = true;
                    } else {
                        index.retain_fields(attr, index_type);
                    }
                }
            }
            // All labels were removed
            if index.is_empty() {
                index.set_under_construction(0, 0);
                index.increment_pending();
            }
            if removed {
                index.set_under_construction(0, total);
            }
            // Return the number of indexes before and after the operation
            return Some((number_of_indexes, index.index_count()));
        }
        None
    }

    pub fn remove(
        &mut self,
        label: Arc<String>,
    ) {
        self.index.write().unwrap().remove(&label);
    }

    #[must_use]
    pub fn is_label_indexed(
        &self,
        label: Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().unwrap().get(&label)
            && index.is_operational()
        {
            return true;
        }
        false
    }

    #[must_use]
    pub fn is_attr_indexed(
        &self,
        label: Arc<String>,
        field: Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().unwrap().get(&label)
            && index.is_operational()
        {
            return index.contains_field(&field);
        }
        false
    }

    #[must_use]
    pub fn query(
        &self,
        label: Arc<String>,
        query: IndexQuery<Value>,
    ) -> Vec<u64> {
        if let Some(index) = self.index.read().unwrap().get(&label) {
            return index.query(query);
        }
        vec![]
    }

    pub fn enable(
        &mut self,
        label: Arc<String>,
    ) -> bool {
        let mut index = self.index.write().unwrap();
        if let Some(index) = index.get_mut(&label) {
            let res = index.decrement_pending();
            debug_assert!(res > 0);
            return res == 1;
        }
        false
    }

    pub fn disable(
        &mut self,
        label: Arc<String>,
    ) {
        let mut index = self.index.write().unwrap();
        if let Some(index) = index.get_mut(&label) {
            index.increment_pending();
        }
    }

    #[must_use]
    pub fn enabled(
        &self,
        label: Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.read().unwrap().get(&label) {
            return index.pending_count() == 0;
        }
        false
    }

    #[must_use]
    pub fn pending_changes(
        &self,
        label: Arc<String>,
    ) -> i32 {
        if let Some(index) = self.index.read().unwrap().get(&label) {
            return index.pending_count();
        }
        0
    }

    pub fn commit(
        &mut self,
        add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let mut index = self.index.write().unwrap();
        for (label, add_docs) in add_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for doc in add_docs.drain(..) {
                index.add_document(&doc);
            }
            index.set_operational();
        }
        for (label, remove_docs) in remove_docs {
            let Some(index) = index.get_mut(label) else {
                continue;
            };
            for id in remove_docs.iter() {
                index.delete_document(id);
            }
        }
    }

    #[must_use]
    pub fn get_fields(
        &self,
        label: Arc<String>,
    ) -> HashMap<Arc<String>, Vec<Arc<Field>>> {
        self.index
            .read()
            .unwrap()
            .get(&label)
            .map(|index| index.clone_fields())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        self.index
            .read()
            .unwrap()
            .iter()
            .map(|(label, index)| IndexInfo {
                label: label.clone(),
                status: index.status(),
                fields: index.clone_fields(),
            })
            .collect()
    }

    pub fn recreate_index(
        &mut self,
        label: Arc<String>,
    ) -> Result<(), String> {
        let mut index = self.index.write().unwrap();
        if let Some(index) = index.get_mut(&label) {
            index.recreate_index(label)?;
        }
        Ok(())
    }
}
