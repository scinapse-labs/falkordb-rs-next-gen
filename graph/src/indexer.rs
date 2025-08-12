use std::{
    collections::HashMap,
    ffi::CString,
    hash::Hash,
    os::raw::{c_char, c_int, c_void},
    ptr::null_mut,
    sync::Arc,
};

use crate::{
    redisearch::{
        GC_POLICY_FORK, REDISEARCH_ADD_REPLACE, RSDoc, RSFLDOPT_NONE, RSFLDTYPE_FULLTEXT,
        RSFLDTYPE_GEO, RSFLDTYPE_NUMERIC, RSFLDTYPE_TAG, RSFLDTYPE_VECTOR, RSIndex, RSRANGE_INF,
        RSRANGE_NEG_INF, RediSearch_CreateDocument2, RediSearch_CreateField,
        RediSearch_CreateIndex, RediSearch_CreateIndexOptions, RediSearch_CreateNumericNode,
        RediSearch_CreateTagNode, RediSearch_CreateTagTokenNode, RediSearch_DeleteDocument,
        RediSearch_DocumentAddFieldNumber, RediSearch_DocumentAddFieldString, RediSearch_DropIndex,
        RediSearch_FreeIndexOptions, RediSearch_GetResultsIterator, RediSearch_IndexAddDocument,
        RediSearch_IndexOptionsSetGCPolicy, RediSearch_IndexOptionsSetStopwords,
        RediSearch_QueryNodeAddChild, RediSearch_ResultsIteratorFree,
        RediSearch_ResultsIteratorNext, RediSearch_TagFieldSetCaseSensitive,
        RediSearch_TagFieldSetSeparator, RediSearch_TextFieldSetWeight,
    },
    runtime::value::Value,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexType {
    Range,
    Fulltext,
    Vector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityType {
    Node,
    Relationship,
}

#[derive(Clone)]
pub struct Document {
    rs_doc: *mut RSDoc,
}

unsafe impl Send for Document {}
unsafe impl Sync for Document {}

impl Document {
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            rs_doc: unsafe {
                RediSearch_CreateDocument2(
                    (&raw const id).cast::<c_void>(),
                    8,
                    null_mut(),
                    1.0,
                    null_mut(),
                )
            },
        }
    }

    pub fn set(
        &mut self,
        field: Arc<Field>,
        value: Value,
    ) {
        unsafe {
            match value {
                Value::Bool(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        f64::from(i),
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::Int(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        i as f64,
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::Float(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        i,
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::String(s) => {
                    RediSearch_DocumentAddFieldString(
                        self.rs_doc,
                        field.name.as_ptr(),
                        s.as_ptr().cast::<i8>(),
                        s.len(),
                        if field.ty == IndexType::Fulltext {
                            RSFLDTYPE_FULLTEXT
                        } else {
                            RSFLDTYPE_TAG
                        },
                    );
                }
                Value::List(values) => todo!(),
                Value::VecF32(items) => todo!(),
                Value::Null
                | Value::Map(_)
                | Value::Node(_)
                | Value::Relationship(_, _, _)
                | Value::Path(_)
                | Value::Arc(_) => unreachable!(),
            }
        }
    }
}

#[derive(Debug)]
pub enum IndexQuery<T> {
    Equal(Arc<String>, T),
    Range(Arc<String>, Option<T>, Option<T>),
    And(Vec<IndexQuery<T>>),
    Or(Vec<IndexQuery<T>>),
}

pub struct Field {
    pub name: CString,
    pub ty: IndexType,
}

impl PartialEq for Field {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.name == other.name && self.ty == other.ty
    }
}

impl Eq for Field {}

impl Hash for Field {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        self.name.hash(state);
    }
}

#[derive(Clone)]
pub enum IndexStatus {
    Operational,
    UnderConstruction(u64, u64),
}

struct Index {
    rs_idx: *mut RSIndex,
    fields: HashMap<Arc<String>, Vec<Arc<Field>>>,
    status: IndexStatus,
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            RediSearch_DropIndex(self.rs_idx);
        }
    }
}

#[derive(Default)]
pub struct Indexer {
    index: HashMap<Arc<String>, Index>,
}

unsafe impl Send for Indexer {}
unsafe impl Sync for Indexer {}

impl Indexer {
    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        label: Arc<String>,
        attrs: &Vec<Arc<String>>,
        total: u64,
    ) -> Result<(), String> {
        let mut fields = HashMap::new();
        if let Some(a) = self.index.get_mut(&label) {
            for attr in attrs {
                if let Some(f) = a.fields.get_mut(attr) {
                    if f.iter().any(|f| f.ty == *index_type) {
                        return Err(format!("Attribute '{attr}' is already indexed"));
                    }
                    let field_name = match index_type {
                        IndexType::Range => Arc::new(format!("range:{attr}")),
                        IndexType::Fulltext => attr.clone(),
                        IndexType::Vector => Arc::new(format!("vector:{attr}")),
                    };
                    let field = Arc::new(Field {
                        name: CString::new(field_name.as_str()).unwrap(),
                        ty: index_type.clone(),
                    });
                    f.push(field);
                } else {
                    let field_name = match index_type {
                        IndexType::Range => Arc::new(format!("range:{attr}")),
                        IndexType::Fulltext => attr.clone(),
                        IndexType::Vector => Arc::new(format!("vector:{attr}")),
                    };
                    let field = Arc::new(Field {
                        name: CString::new(field_name.as_str()).unwrap(),
                        ty: index_type.clone(),
                    });
                    a.fields.insert(attr.clone(), vec![field]);
                }
            }
            fields.clone_from(&a.fields);
        } else {
            for attr in attrs {
                let field_name = match index_type {
                    IndexType::Range => Arc::new(format!("range:{attr}")),
                    IndexType::Fulltext => attr.clone(),
                    IndexType::Vector => Arc::new(format!("vector:{attr}")),
                };
                if fields.contains_key(attr) {
                    return Err(format!("Attribute '{attr}' is already indexed"));
                }
                let field = Arc::new(Field {
                    name: CString::new(field_name.as_str()).unwrap(),
                    ty: index_type.clone(),
                });
                fields.insert(attr.clone(), vec![field]);
            }
        }
        unsafe {
            let options = RediSearch_CreateIndexOptions();
            // RediSearch_IndexOptionsSetLanguage(options, idx->language);
            RediSearch_IndexOptionsSetGCPolicy(options, GC_POLICY_FORK as _);
            RediSearch_IndexOptionsSetStopwords(options, null_mut(), 0);

            let index = RediSearch_CreateIndex(c"index".as_ptr().cast::<i8>(), options);
            RediSearch_FreeIndexOptions(options);

            for field in fields.values().flat_map(|f| f.iter()) {
                match field.ty {
                    IndexType::Range => {
                        let types = RSFLDTYPE_NUMERIC | RSFLDTYPE_GEO | RSFLDTYPE_TAG;
                        let field_id = RediSearch_CreateField(
                            index,
                            field.name.as_ptr(),
                            types,
                            RSFLDOPT_NONE,
                        );

                        RediSearch_TagFieldSetSeparator(index, field_id, 1 as c_char);
                        RediSearch_TagFieldSetCaseSensitive(index, field_id, 1);
                    }
                    IndexType::Fulltext => {
                        let field_id = RediSearch_CreateField(
                            index,
                            field.name.as_ptr(),
                            RSFLDTYPE_FULLTEXT,
                            RSFLDOPT_NONE,
                        );

                        RediSearch_TextFieldSetWeight(index, field_id, 1.0);
                    }
                    IndexType::Vector => {
                        let field_id = RediSearch_CreateField(
                            index,
                            field.name.as_ptr(),
                            RSFLDTYPE_VECTOR,
                            RSFLDOPT_NONE,
                        );
                        //    RediSearch_VectorFieldSetDim(index, field_id, field->hnsw_options.dimension);
                        // RediSearch_VectorFieldSetHNSWParams(index, field_id, IndexField_OptionsGetM(field), IndexField_OptionsGetEfConstruction(field), IndexField_OptionsGetEfRuntime(field), IndexField_OptionsGetSimFunc(field));
                    }
                }
            }

            self.index.insert(
                label,
                Index {
                    rs_idx: index,
                    fields,
                    status: IndexStatus::UnderConstruction(0, total),
                },
            );
        }
        Ok(())
    }

    pub fn drop_index(
        &mut self,
        label: Arc<String>,
        attrs: &Vec<Arc<String>>,
        index_type: &IndexType,
        total: u64,
    ) -> Option<Vec<Arc<String>>> {
        if let Some(index) = self.index.get_mut(&label) {
            let mut removed = false;
            for attr in attrs {
                if let Some(field) = index.fields.get(attr)
                    && field.iter().any(|f| f.ty == *index_type)
                {
                    if field.len() == 1 {
                        index.fields.remove(attr);
                        removed = true;
                    } else {
                        index
                            .fields
                            .get_mut(attr)
                            .unwrap()
                            .retain(|f| f.ty != *index_type);
                    }
                }
            }
            if index.fields.is_empty() {
                self.index.remove(&label).unwrap();
                return None;
            }
            if removed {
                index.status = IndexStatus::UnderConstruction(0, total);
                return Some(index.fields.keys().cloned().collect());
            }
        }
        None
    }

    #[must_use]
    pub fn is_label_indexed(
        &self,
        label: Arc<String>,
    ) -> bool {
        if let Some(index) = self.index.get(&label)
            && matches!(index.status, IndexStatus::Operational)
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
        if let Some(index) = self.index.get(&label)
            && matches!(index.status, IndexStatus::Operational)
        {
            return index.fields.contains_key(&field);
        }
        false
    }

    #[must_use]
    pub fn query(
        &self,
        label: Arc<String>,
        query: IndexQuery<Value>,
    ) -> Vec<u64> {
        if let Some(index) = self.index.get(&label) {
            match query {
                IndexQuery::Equal(key, Value::Int(value)) => unsafe {
                    let field = &index.fields.get(&key).unwrap()[0];
                    let query = RediSearch_CreateNumericNode(
                        index.rs_idx,
                        field.name.as_ptr(),
                        value as f64,
                        value as f64,
                        1,
                        1,
                    );
                    let iter = RediSearch_GetResultsIterator(query, index.rs_idx);

                    let mut res = vec![];
                    loop {
                        let node_id =
                            RediSearch_ResultsIteratorNext(iter, index.rs_idx, null_mut())
                                .cast::<u64>();
                        if node_id.is_null() {
                            break;
                        }
                        res.push(node_id.read_unaligned());
                    }
                    RediSearch_ResultsIteratorFree(iter);
                    return res;
                },
                IndexQuery::Equal(key, Value::String(value)) => unsafe {
                    let field = &index.fields.get(&key).unwrap()[0];
                    let query = RediSearch_CreateTagNode(index.rs_idx, field.name.as_ptr());
                    let msg = CString::new(value.as_str()).unwrap();
                    let child =
                        RediSearch_CreateTagTokenNode(index.rs_idx, msg.as_ptr().cast::<c_char>());
                    RediSearch_QueryNodeAddChild(query, child);
                    let iter = RediSearch_GetResultsIterator(query, index.rs_idx);

                    let mut res = vec![];
                    loop {
                        let node_id =
                            RediSearch_ResultsIteratorNext(iter, index.rs_idx, null_mut())
                                .cast::<u64>();
                        if node_id.is_null() {
                            break;
                        }
                        res.push(node_id.read_unaligned());
                    }
                    RediSearch_ResultsIteratorFree(iter);
                    return res;
                },
                IndexQuery::Range(key, min, max) => {
                    let (min, max) = match (min, max) {
                        (Some(Value::Float(min)), None) => (min, RSRANGE_INF),
                        (None, Some(Value::Float(max))) => (RSRANGE_NEG_INF, max),
                        (Some(Value::Float(min)), Some(Value::Float(max))) => (min, max),
                        (Some(Value::Int(min)), None) => (min as f64, RSRANGE_INF),
                        (None, Some(Value::Int(max))) => (RSRANGE_NEG_INF, max as f64),
                        (Some(Value::Int(min)), Some(Value::Int(max))) => (min as f64, max as f64),
                        _ => todo!(),
                    };
                    unsafe {
                        let field = &index.fields.get(&key).unwrap()[0];
                        let query = RediSearch_CreateNumericNode(
                            index.rs_idx,
                            field.name.as_ptr(),
                            max,
                            min,
                            0,
                            0,
                        );
                        let iter = RediSearch_GetResultsIterator(query, index.rs_idx);

                        let mut res = vec![];
                        loop {
                            let node_id =
                                RediSearch_ResultsIteratorNext(iter, index.rs_idx, null_mut())
                                    .cast::<u64>();
                            if node_id.is_null() {
                                break;
                            }
                            res.push(node_id.read_unaligned());
                        }
                        RediSearch_ResultsIteratorFree(iter);
                        return res;
                    }
                }
                _ => todo!(),
            }
        }

        vec![]
    }

    pub fn commit(
        &mut self,
        index_add_docs: &mut HashMap<Arc<String>, Vec<Document>>,
        remove_docs: &mut HashMap<Arc<String>, Vec<u64>>,
    ) {
        for (label, add_docs) in index_add_docs {
            let Some(index) = self.index.get_mut(label) else {
                continue;
            };
            for doc in add_docs.drain(..) {
                unsafe {
                    let res = RediSearch_IndexAddDocument(
                        index.rs_idx,
                        doc.rs_doc,
                        REDISEARCH_ADD_REPLACE as c_int,
                        null_mut(),
                    );
                    debug_assert_eq!(res, 0);
                }
            }
            index.status = IndexStatus::Operational;
        }
        for (label, remove_docs) in remove_docs {
            let Some(index) = self.index.get_mut(label) else {
                continue;
            };
            for id in remove_docs.drain(..) {
                unsafe {
                    RediSearch_DeleteDocument(index.rs_idx, (&raw const id).cast::<c_void>(), 8)
                };
            }
        }
    }

    #[must_use]
    pub fn get_fields(
        &self,
        label: Arc<String>,
    ) -> HashMap<Arc<String>, Vec<Arc<Field>>> {
        self.index
            .get(&label)
            .map(|index| index.fields.clone())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn index_info(
        &self
    ) -> Vec<(
        Arc<String>,
        IndexStatus,
        HashMap<Arc<String>, Vec<Arc<Field>>>,
    )> {
        self.index
            .iter()
            .map(|(label, index)| {
                let attrs = index
                    .fields
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                (label.clone(), index.status.clone(), attrs)
            })
            .collect()
    }
}
