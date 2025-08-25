use std::{
    collections::HashMap,
    num::NonZeroUsize,
    rc::Rc,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

use itertools::Itertools;
use lru::LruCache;
use once_cell::sync::Lazy;
use ordermap::{OrderMap, OrderSet};
use orx_tree::DynTree;
use rayon::spawn;
use roaring::RoaringTreemap;

use crate::{
    ast::ExprIR,
    cypher::Parser,
    graph::{
        matrix::{Dup, ElementWiseAdd, ElementWiseMultiply, Matrix, MxM, New, Remove, Set, Size},
        tensor::Tensor,
    },
    indexer::{Document, EntityType, IndexInfo, IndexQuery, IndexType, Indexer},
    optimizer::optimize,
    planner::{IR, Planner},
    runtime::{pending::PendingRelationship, value::Value},
};

pub struct Plan {
    pub plan: Rc<DynTree<IR>>,
    pub cached: bool,
    pub parameters: HashMap<String, DynTree<ExprIR>>,
    pub parse_duration: Duration,
    pub plan_duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AttrId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationshipId(u64);

impl From<LabelId> for usize {
    fn from(val: LabelId) -> Self {
        val.0
    }
}

impl From<TypeId> for usize {
    fn from(val: TypeId) -> Self {
        val.0
    }
}

impl From<AttrId> for usize {
    fn from(val: AttrId) -> Self {
        val.0
    }
}

impl From<u64> for NodeId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<NodeId> for u64 {
    fn from(value: NodeId) -> Self {
        value.0
    }
}

impl From<RelationshipId> for u64 {
    fn from(value: RelationshipId) -> Self {
        value.0
    }
}

impl Plan {
    #[must_use]
    pub const fn new(
        plan: Rc<DynTree<IR>>,
        cached: bool,
        parameters: HashMap<String, DynTree<ExprIR>>,
        parse_duration: Duration,
        plan_duration: Duration,
    ) -> Self {
        Self {
            plan,
            cached,
            parameters,
            parse_duration,
            plan_duration,
        }
    }
}

pub struct Graph {
    node_cap: u64,
    relationship_cap: u64,
    reserved_node_count: u64,
    reserved_relationship_count: u64,
    node_count: u64,
    relationship_count: u64,
    deleted_nodes: RoaringTreemap,
    deleted_relationships: RoaringTreemap,
    zero_matrix: Matrix<bool>,
    adjacancy_matrix: Matrix<bool>,
    node_labels_matrix: Matrix<bool>,
    relationship_type_matrix: Matrix<bool>,
    all_nodes_matrix: Matrix<bool>,
    labels_matices: HashMap<usize, Arc<Mutex<Matrix<bool>>>>,
    relationship_matrices: HashMap<usize, Arc<Mutex<Tensor>>>,
    empty_map: OrderMap<AttrId, Value>,
    node_attrs: Arc<Mutex<HashMap<NodeId, OrderMap<AttrId, Value>>>>,
    relationship_attrs: HashMap<RelationshipId, OrderMap<AttrId, Value>>,
    node_indexer: Arc<Mutex<Indexer>>,
    node_labels: Vec<Arc<String>>,
    relationship_types: Vec<Arc<String>>,
    node_attrs_name: Arc<Mutex<Vec<Arc<String>>>>,
    relationship_attrs_name: Vec<Arc<String>>,
    cache: Mutex<LruCache<String, DynTree<IR>>>,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

static INDEXER_CHANNEL: Lazy<
    mpsc::Sender<(
        i32,
        Arc<String>,
        Arc<Mutex<Matrix<bool>>>,
        Arc<Mutex<Indexer>>,
        Arc<Mutex<HashMap<NodeId, OrderMap<AttrId, Value>>>>,
        Arc<Mutex<Vec<Arc<String>>>>,
    )>,
> = Lazy::new(|| {
    let (sender, receiver) = mpsc::channel::<(
        i32,
        Arc<String>,
        Arc<Mutex<Matrix<bool>>>,
        Arc<Mutex<Indexer>>,
        Arc<Mutex<HashMap<NodeId, OrderMap<AttrId, Value>>>>,
        Arc<Mutex<Vec<Arc<String>>>>,
    )>();
    spawn(move || {
        loop {
            let Ok((action, label, lm, node_indexer, node_attrs, node_attrs_name)) =
                receiver.recv()
            else {
                break;
            };

            if action == 0 {
                if node_indexer.lock().unwrap().pending_changes(label.clone()) > 1 {
                    node_indexer.lock().unwrap().enable(label);
                    continue;
                }
                let attr_ids = {
                    let node_indexer = node_indexer.lock().unwrap();

                    node_indexer
                        .get_fields(label.clone())
                        .into_iter()
                        .filter_map(|(attr, field)| {
                            node_attrs_name
                                .lock()
                                .unwrap()
                                .iter()
                                .position(|p| p.as_str() == attr.as_str())
                                .map(AttrId)
                                .map(|id| (id, field))
                        })
                        .collect::<Vec<_>>()
                };
                let mut add_docs = vec![];
                let node_attrs = node_attrs.lock().unwrap();
                let value = lm.lock().unwrap().iter(0, u64::MAX);
                for (n, _) in value {
                    let mut doc = Document::new(n);
                    for (attr_id, fields) in &attr_ids {
                        if let Some(value) = node_attrs
                            .get(&NodeId(n))
                            .map_or_else(|| None, |attrs| attrs.get(attr_id).cloned())
                        {
                            for field in fields {
                                doc.set(field.clone(), value.clone());
                            }
                        }
                    }
                    add_docs.push(doc);
                }
                if node_indexer.lock().unwrap().pending_changes(label.clone()) > 1 {
                    node_indexer.lock().unwrap().enable(label);
                    continue;
                }
                let mut index_add_docs = HashMap::new();
                index_add_docs.insert(label.clone(), add_docs);
                let add_docs = Arc::new(Mutex::new(index_add_docs));

                let mut node_indexer = node_indexer.lock().unwrap();
                let mut add_docs = add_docs.lock().unwrap();
                node_indexer.commit(&mut add_docs, &mut HashMap::new());
                node_indexer.enable(label);
            } else if action == 1 {
                let mut node_indexer = node_indexer.lock().unwrap();
                node_indexer.remove(label);
            }
        }
    });
    sender
});

impl Graph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
    ) -> Self {
        Self {
            node_cap: n,
            relationship_cap: e,
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count: 0,
            relationship_count: 0,
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: RoaringTreemap::new(),
            zero_matrix: Matrix::<bool>::new(0, 0),
            adjacancy_matrix: Matrix::<bool>::new(n, n),
            node_labels_matrix: Matrix::<bool>::new(0, 0),
            relationship_type_matrix: Matrix::<bool>::new(0, 0),
            all_nodes_matrix: Matrix::<bool>::new(n, n),
            labels_matices: HashMap::new(),
            relationship_matrices: HashMap::new(),
            empty_map: OrderMap::new(),
            node_attrs: Arc::new(Mutex::new(HashMap::new())),
            relationship_attrs: HashMap::new(),
            node_indexer: Arc::new(Mutex::new(Indexer::default())),
            node_labels: Vec::new(),
            relationship_types: Vec::new(),
            node_attrs_name: Arc::new(Mutex::new(Vec::new())),
            relationship_attrs_name: Vec::new(),
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(cache_size).unwrap())),
        }
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        debug_assert_eq!(self.reserved_node_count, 0);
        debug_assert_eq!(self.reserved_relationship_count, 0);
        Self {
            node_cap: self.node_cap,
            relationship_cap: self.relationship_cap,
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count: self.node_count,
            relationship_count: self.relationship_count,
            deleted_nodes: self.deleted_nodes.clone(),
            deleted_relationships: self.deleted_relationships.clone(),
            zero_matrix: self.zero_matrix.dup(),
            adjacancy_matrix: self.adjacancy_matrix.dup(),
            node_labels_matrix: self.node_labels_matrix.dup(),
            relationship_type_matrix: self.relationship_type_matrix.dup(),
            all_nodes_matrix: self.all_nodes_matrix.dup(),
            labels_matices: self.labels_matices.clone(),
            relationship_matrices: self.relationship_matrices.clone(),
            empty_map: self.empty_map.clone(),
            node_attrs: self.node_attrs.clone(),
            relationship_attrs: self.relationship_attrs.clone(),
            node_indexer: self.node_indexer.clone(),
            node_labels: self.node_labels.clone(),
            relationship_types: self.relationship_types.clone(),
            node_attrs_name: self.node_attrs_name.clone(),
            relationship_attrs_name: self.relationship_attrs_name.clone(),
            cache: Mutex::new(LruCache::new(self.cache.lock().unwrap().cap())),
        }
    }

    pub const fn get_labels_count(&self) -> usize {
        self.node_labels.len()
    }

    pub fn get_labels(&self) -> Vec<Arc<String>> {
        self.node_labels.clone()
    }

    pub fn get_label_by_id(
        &self,
        id: LabelId,
    ) -> Arc<String> {
        self.node_labels[id.0].clone()
    }

    pub fn get_types(&self) -> Vec<Arc<String>> {
        self.relationship_types.clone()
    }

    pub fn get_type(
        &self,
        id: TypeId,
    ) -> Option<Arc<String>> {
        self.relationship_types.get(id.0).cloned()
    }

    pub fn get_attrs(&self) -> Vec<Arc<String>> {
        self.node_attrs_name
            .lock()
            .unwrap()
            .iter()
            .chain(self.relationship_attrs_name.iter())
            .cloned()
            .collect()
    }

    pub fn get_label_id(
        &self,
        label: &str,
    ) -> Option<LabelId> {
        self.node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(LabelId)
    }

    pub fn get_type_id(
        &self,
        relationship_type: &str,
    ) -> Option<TypeId> {
        self.relationship_types
            .iter()
            .position(|t| t.as_str() == relationship_type)
            .map(TypeId)
    }

    pub fn get_plan(
        &self,
        query: &str,
    ) -> Result<Plan, String> {
        let mut parse_duration = Duration::ZERO;
        let mut plan_duration = Duration::ZERO;

        let mut parser = Parser::new(query);
        let (parameters, query) = parser.parse_parameters()?;

        match self.cache.lock() {
            Ok(mut cache) => {
                if let Some(plan) = cache.get(query) {
                    let optimize_plan = optimize(plan, self);
                    Ok(Plan::new(
                        Rc::new(optimize_plan),
                        true,
                        parameters,
                        parse_duration,
                        plan_duration,
                    ))
                } else {
                    let start = Instant::now();
                    let ir = parser.parse()?;
                    parse_duration = start.elapsed();

                    let mut planner = Planner::default();
                    let start = Instant::now();
                    let plan = planner.plan(ir);
                    let optimize_plan = optimize(&plan, self);
                    plan_duration = start.elapsed();

                    cache.push(query.to_string(), plan);
                    Ok(Plan::new(
                        Rc::new(optimize_plan),
                        false,
                        parameters,
                        parse_duration,
                        plan_duration,
                    ))
                }
            }
            Err(_) => Err("Failed to acquire read lock on cache".to_string()),
        }
    }

    fn get_label_matrix(
        &self,
        label: &str,
    ) -> Option<Arc<Mutex<Matrix<bool>>>> {
        self.node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(|i| self.labels_matices[&i].clone())
    }

    fn get_label_matrix_mut(
        &mut self,
        label: &Arc<String>,
    ) -> Arc<Mutex<Matrix<bool>>> {
        if !self.node_labels.contains(label) {
            self.node_labels.push(label.clone());

            self.labels_matices.insert(
                self.node_labels.len() - 1,
                Arc::new(Mutex::new(Matrix::<bool>::new(
                    self.node_cap,
                    self.node_cap,
                ))),
            );
        }

        self.labels_matices
            .get(
                &self
                    .node_labels
                    .iter()
                    .position(|l| l.as_str() == label.as_str())
                    .unwrap(),
            )
            .unwrap()
            .clone()
    }

    fn get_relationship_matrix_mut(
        &mut self,
        relationship_type: &Arc<String>,
    ) -> Arc<Mutex<Tensor>> {
        if !self.relationship_types.contains(relationship_type) {
            self.relationship_types.push(relationship_type.clone());

            self.relationship_matrices.insert(
                self.relationship_types.len() - 1,
                Arc::new(Mutex::new(Tensor::new(self.node_cap, self.node_cap))),
            );
        }

        self.relationship_matrices
            .get(
                &self
                    .relationship_types
                    .iter()
                    .position(|l| l.as_str() == relationship_type.as_str())
                    .unwrap(),
            )
            .unwrap()
            .clone()
    }

    fn get_relationship_matrix(
        &self,
        relationship_type: &Arc<String>,
    ) -> Option<Arc<Mutex<Tensor>>> {
        if !self.relationship_types.contains(relationship_type) {
            return None;
        }

        self.relationship_matrices
            .get(
                &self
                    .relationship_types
                    .iter()
                    .position(|l| l.as_str() == relationship_type.as_str())
                    .unwrap(),
            )
            .cloned()
    }

    pub fn get_node_attribute_id(
        &self,
        key: &str,
    ) -> Option<AttrId> {
        self.node_attrs_name
            .lock()
            .unwrap()
            .iter()
            .position(|p| p.as_str() == key)
            .map(AttrId)
    }

    pub fn get_node_attribute_string(
        &self,
        id: AttrId,
    ) -> Option<Arc<String>> {
        self.node_attrs_name.lock().unwrap().get(id.0).cloned()
    }

    pub fn get_or_add_node_attribute_id(
        &mut self,
        key: &Arc<String>,
    ) -> AttrId {
        let mut node_attrs_name = self.node_attrs_name.lock().unwrap();
        AttrId(
            node_attrs_name
                .iter()
                .position(|p| p.as_str() == key.as_str())
                .unwrap_or_else(|| {
                    let len = node_attrs_name.len();
                    node_attrs_name.push(key.clone());
                    len
                }),
        )
    }

    pub fn get_or_add_relationship_attribute_id(
        &mut self,
        key: &String,
    ) -> AttrId {
        AttrId(
            self.relationship_attrs_name
                .iter()
                .position(|p| p.as_str() == key)
                .unwrap_or_else(|| {
                    let len = self.relationship_attrs_name.len();
                    self.relationship_attrs_name.push(Arc::new(key.clone()));
                    len
                }),
        )
    }

    pub fn get_relationship_attribute_id(
        &self,
        key: &str,
    ) -> Option<AttrId> {
        self.relationship_attrs_name
            .iter()
            .position(|p| p.as_str() == key)
            .map(AttrId)
    }

    pub fn get_relationship_attribute_string(
        &self,
        id: AttrId,
    ) -> Option<Arc<String>> {
        self.relationship_attrs_name.get(id.0).cloned()
    }

    pub fn reserve_node(&mut self) -> NodeId {
        if self.reserved_node_count < self.deleted_nodes.len() {
            let id = self.deleted_nodes.select(self.reserved_node_count).unwrap();
            self.reserved_node_count += 1;
            return NodeId(id);
        }
        self.reserved_node_count += 1;
        NodeId(self.node_count + self.reserved_node_count - 1)
    }

    pub fn create_nodes(
        &mut self,
        nodes: &RoaringTreemap,
    ) {
        self.node_count += nodes.len();
        self.reserved_node_count -= nodes.len();
        self.deleted_nodes -= nodes;

        self.resize();

        for id in nodes {
            self.all_nodes_matrix.set(id, id, true);
        }
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        attr_id: AttrId,
        value: Value,
        index_add_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
        index_remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) -> bool {
        let attr_name = self.get_node_attribute_string(attr_id).unwrap();
        let mut node_attrs = self.node_attrs.lock().unwrap();
        let attrs = node_attrs.entry(id).or_default();
        if value == Value::Null {
            let removed = attrs.remove(&attr_id).is_some();
            if attrs.is_empty() {
                node_attrs.remove(&id);
            }
            if removed {
                if node_attrs.contains_key(&id) {
                    for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                        let label = self.node_labels[label_id as usize].clone();
                        if self
                            .node_indexer
                            .lock()
                            .unwrap()
                            .is_attr_indexed(label.clone(), attr_name.clone())
                        {
                            index_add_docs
                                .entry(label)
                                .or_default()
                                .insert(u64::from(id));
                        }
                    }
                } else {
                    let node_indexer = self.node_indexer.lock().unwrap();
                    for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                        let label = self.node_labels[label_id as usize].clone();
                        if node_indexer.is_attr_indexed(label.clone(), attr_name.clone()) {
                            index_remove_docs
                                .entry(label)
                                .or_default()
                                .insert(u64::from(id));
                        }
                    }
                }
            }
            removed
        } else {
            let res = attrs.insert(attr_id, value).is_some();
            for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                let label = self.node_labels[label_id as usize].clone();
                if self
                    .node_indexer
                    .lock()
                    .unwrap()
                    .is_attr_indexed(label.clone(), attr_name.clone())
                {
                    index_add_docs
                        .entry(label)
                        .or_default()
                        .insert(u64::from(id));
                }
            }
            res
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<Arc<String>>,
        index_add_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        for label in labels {
            let label_matrix = self.get_label_matrix_mut(label);
            label_matrix.lock().unwrap().set(id.0, id.0, true);
            let label_id = self.get_label_id(label).unwrap();
            self.resize();
            self.node_labels_matrix.set(id.0, label_id.0 as u64, true);
            if self
                .node_indexer
                .lock()
                .unwrap()
                .is_label_indexed(label.clone())
                && self.node_attrs.lock().unwrap().contains_key(&id)
            {
                index_add_docs
                    .entry(label.clone())
                    .or_default()
                    .insert(u64::from(id));
            }
        }
    }

    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<Arc<String>>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        for label in labels {
            if !self.node_labels.contains(label) {
                continue;
            }
            let label_matrix = self.get_label_matrix_mut(label);
            label_matrix.lock().unwrap().remove(id.0, id.0);
            let label_id = self.get_label_id(label).unwrap();
            self.node_labels_matrix.remove(id.0, label_id.0 as u64);
            let node_indexer = self.node_indexer.lock().unwrap();
            if node_indexer.is_label_indexed(label.clone()) {
                remove_docs
                    .entry(label.clone())
                    .or_default()
                    .insert(u64::from(id));
            }
        }
    }

    pub fn delete_node(
        &mut self,
        id: NodeId,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        self.deleted_nodes.insert(id.0);
        self.node_count -= 1;
        self.all_nodes_matrix.remove(id.0, id.0);

        for label_matrix in self.labels_matices.values_mut() {
            label_matrix.lock().unwrap().remove(id.0, id.0);
        }
        let mut node_attrs = self.node_attrs.lock().unwrap();
        let node_indexer = self.node_indexer.lock().unwrap();
        for label_id in self.labels_matices.keys() {
            let label = self.node_labels[*label_id].clone();
            self.node_labels_matrix.remove(id.0, *label_id as _);
            let mut indexed = false;
            for (attr_id, _) in node_attrs.get(&id).unwrap_or(&self.empty_map) {
                let attr_name = self.get_node_attribute_string(*attr_id).unwrap();
                if node_indexer.is_attr_indexed(label.clone(), attr_name) {
                    indexed = true;
                    break;
                }
            }
            if indexed {
                remove_docs
                    .entry(label.clone())
                    .or_default()
                    .insert(u64::from(id));
            }
        }

        node_attrs.remove(&id);
    }

    pub fn get_node_relationships(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + '_ {
        self.relationship_matrices
            .values()
            .flat_map(move |m| {
                let m = m.lock().unwrap();
                m.iter(id.0, id.0, false)
                    .chain(m.iter(id.0, id.0, true))
                    .collect::<Vec<_>>()
            })
            .map(|(src, dest, id)| {
                let src_node = NodeId(src);
                let dest_node = NodeId(dest);
                (src_node, dest_node, RelationshipId(id))
            })
    }

    pub fn get_nodes(
        &self,
        labels: &OrderSet<Arc<String>>,
    ) -> impl Iterator<Item = NodeId> + use<> {
        let iter = if labels.is_empty() {
            self.all_nodes_matrix.iter(0, u64::MAX)
        } else {
            let matrices = labels
                .iter()
                .map(|label| self.get_label_matrix(label))
                .collect::<Option<Vec<_>>>();
            matrices.map_or_else(
                || self.zero_matrix.iter(0, u64::MAX),
                |matrices| {
                    let mut iter = matrices.iter();
                    let mut m = iter.next().unwrap().lock().unwrap().dup();
                    for label_matrix in iter {
                        let label_matrix = &*label_matrix.lock().unwrap();
                        m.element_wise_multiply(label_matrix);
                    }
                    m.iter(0, u64::MAX)
                },
            )
        };
        iter.map(|(id, _)| NodeId(id))
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get_node_label_ids(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = LabelId> {
        self.node_labels_matrix
            .iter(id.0, id.0)
            .map(|(_, l)| LabelId(l as usize))
    }

    pub fn get_node_labels(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = Arc<String>> {
        self.get_node_label_ids(id)
            .map(move |label_id| self.node_labels[label_id.0].clone())
    }

    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr_id: AttrId,
    ) -> Option<Value> {
        let node_attrs = self.node_attrs.lock().unwrap();
        node_attrs
            .get(&id)
            .map_or_else(|| None, |attrs| attrs.get(&attr_id).cloned())
    }

    pub fn reserve_relationship(&mut self) -> RelationshipId {
        if self.reserved_relationship_count < self.deleted_relationships.len() {
            let id = self
                .deleted_relationships
                .select(self.reserved_relationship_count)
                .unwrap();
            self.reserved_relationship_count += 1;
            return RelationshipId(id);
        }
        self.reserved_relationship_count += 1;
        RelationshipId(self.relationship_count + self.reserved_relationship_count - 1)
    }

    pub fn create_relationships(
        &mut self,
        relationships: &OrderMap<RelationshipId, PendingRelationship>,
    ) {
        self.relationship_count += relationships.len() as u64;
        self.reserved_relationship_count -= relationships.len() as u64;

        for id in relationships.keys() {
            if self.deleted_relationships.is_empty() {
                break;
            }
            self.deleted_relationships.remove(id.0);
        }

        for (
            id,
            PendingRelationship {
                type_name,
                from: start,
                to: end,
                ..
            },
        ) in relationships
        {
            let relationship_type_matrix = self.get_relationship_matrix_mut(type_name);
            relationship_type_matrix
                .lock()
                .unwrap()
                .set(start.0, end.0, id.0);
        }

        self.resize();

        for (
            id,
            PendingRelationship {
                type_name,
                from: start,
                to: end,
            },
        ) in relationships
        {
            self.adjacancy_matrix.set(start.0, end.0, true);
            self.relationship_type_matrix.set(
                id.0,
                self.relationship_types
                    .iter()
                    .position(|p| p.as_str() == type_name.as_str())
                    .unwrap() as u64,
                true,
            );
        }
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        attr_id: AttrId,
        value: Value,
    ) -> bool {
        let attrs = self.relationship_attrs.entry(id).or_default();
        if value == Value::Null {
            attrs.remove(&attr_id).is_some()
        } else {
            attrs.insert(attr_id, value).is_some()
        }
    }

    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.0)
    }

    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.deleted_relationships.contains(id.0)
    }

    pub fn delete_relationships(
        &mut self,
        rels: OrderSet<(RelationshipId, NodeId, NodeId)>,
    ) {
        self.deleted_relationships
            .extend(rels.iter().map(|(id, _, _)| id.0));
        self.relationship_count -= rels.len() as u64;
        let mut r = vec![];
        for (type_id, rels) in &rels
            .into_iter()
            .chunk_by(|(id, _, _)| self.get_relationship_type_id(*id))
        {
            r.push((
                type_id,
                rels.map(|(id, src, dest)| (id.0, src.0, dest.0))
                    .collect::<Vec<_>>(),
            ));
        }

        for (type_id, rels) in r {
            let label = self.relationship_types.get(type_id.0).cloned().unwrap();
            for (id, _, _) in &rels {
                self.relationship_type_matrix.remove(*id, type_id.0 as u64);
                self.relationship_attrs.remove(&RelationshipId(*id));
            }
            let t = self.get_relationship_matrix_mut(&label);
            t.lock().unwrap().remove_all(rels);
        }
    }

    pub fn get_src_dest_relationships(
        &self,
        src: NodeId,
        dest: NodeId,
        types: &[Arc<String>],
    ) -> Vec<RelationshipId> {
        let mut vec = vec![];
        for relationship_type in if types.is_empty() {
            &self.relationship_types
        } else {
            types
        } {
            if let Some(relationship_matrix) = self.get_relationship_matrix(relationship_type) {
                for id in relationship_matrix.lock().unwrap().get(src.0, dest.0) {
                    vec.push(RelationshipId(id));
                }
            }
        }
        vec
    }

    pub fn get_relationships(
        &self,
        types: &[Arc<String>],
        src_lables: &OrderSet<Arc<String>>,
        dest_labels: &OrderSet<Arc<String>>,
    ) -> impl Iterator<Item = (NodeId, NodeId)> + use<> {
        let matrices = types
            .iter()
            .map(|relationship_type| self.get_relationship_matrix(relationship_type))
            .collect::<Option<Vec<_>>>();
        let src_labels_matrices = src_lables
            .iter()
            .map(|label| self.get_label_matrix(label))
            .collect::<Option<Vec<_>>>();
        let dest_labels_matrices = dest_labels
            .iter()
            .map(|label| self.get_label_matrix(label))
            .collect::<Option<Vec<_>>>();
        let iter = if let (Some(matrices), Some(src_labels_matrices), Some(dest_labels_matrices)) =
            (matrices, src_labels_matrices, dest_labels_matrices)
        {
            let mut iter = matrices.iter();
            let mut m = iter.next().map_or_else(
                || self.adjacancy_matrix.dup(),
                |relationship_matrix| relationship_matrix.lock().unwrap().dup_bool(),
            );
            for relationship_matrix in iter {
                m.element_wise_add(&relationship_matrix.lock().unwrap().dup_bool());
            }

            if !src_labels_matrices.is_empty() {
                let mut iter = src_labels_matrices.iter();
                let mut src_matrix = iter.next().unwrap().lock().unwrap().dup();
                for label_matrix in iter {
                    let label_matrix = &*label_matrix.lock().unwrap();
                    src_matrix.element_wise_multiply(label_matrix);
                }
                m.rmxm(&src_matrix);
            }
            if !dest_labels_matrices.is_empty() {
                let mut iter = dest_labels_matrices.iter();
                let mut dest_matrix = iter.next().unwrap().lock().unwrap().dup();
                for label_matrix in iter {
                    let label_matrix = &*label_matrix.lock().unwrap();
                    dest_matrix.element_wise_multiply(label_matrix);
                }
                m.lmxm(&dest_matrix);
            }
            m.iter(0, u64::MAX)
        } else {
            self.zero_matrix.iter(0, u64::MAX)
        };

        iter.map(|(src, dest)| (NodeId(src), NodeId(dest)))
    }

    pub fn get_relationship_type_id(
        &self,
        id: RelationshipId,
    ) -> TypeId {
        #[allow(clippy::cast_possible_truncation)]
        self.relationship_type_matrix
            .iter(id.0, id.0)
            .map(|(_, l)| TypeId(l as usize))
            .next()
            .unwrap()
    }

    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr_id: AttrId,
    ) -> Option<Value> {
        self.relationship_attrs
            .get(&id)
            .map_or_else(|| None, |attrs| attrs.get(&attr_id).cloned())
    }

    fn resize(&mut self) {
        if self.node_count > self.node_cap {
            while self.node_count > self.node_cap {
                self.node_cap *= 2;
            }
            self.adjacancy_matrix.resize(self.node_cap, self.node_cap);
            self.node_labels_matrix
                .resize(self.node_cap, self.labels_matices.len() as u64);
            self.all_nodes_matrix.resize(self.node_cap, self.node_cap);
            for label_matrix in self.labels_matices.iter_mut().map(|(_, m)| m) {
                label_matrix
                    .lock()
                    .unwrap()
                    .resize(self.node_cap, self.node_cap);
            }
            for relationship_matrix in self.relationship_matrices.iter_mut().map(|(_, m)| m) {
                relationship_matrix
                    .lock()
                    .unwrap()
                    .resize(self.node_cap, self.node_cap);
            }
        }

        if self.labels_matices.len() as u64 > self.node_labels_matrix.ncols() {
            self.node_labels_matrix
                .resize(self.node_cap, self.labels_matices.len() as u64);
        }

        if self.relationship_count > self.relationship_cap {
            while self.relationship_count > self.relationship_cap {
                self.relationship_cap *= 2;
            }
            self.relationship_type_matrix
                .resize(self.relationship_cap, self.relationship_types.len() as u64);
        }

        if self.relationship_types.len() as u64 > self.relationship_type_matrix.ncols() {
            self.relationship_type_matrix
                .resize(self.relationship_cap, self.relationship_types.len() as u64);
        }
    }

    pub fn get_node_attrs(
        &self,
        id: NodeId,
    ) -> OrderMap<AttrId, Value> {
        let node_attrs = self.node_attrs.lock().unwrap();
        node_attrs.get(&id).unwrap_or(&self.empty_map).clone()
    }

    pub fn get_relationship_attrs(
        &self,
        id: RelationshipId,
    ) -> &OrderMap<AttrId, Value> {
        self.relationship_attrs.get(&id).unwrap_or(&self.empty_map)
    }

    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
    ) -> Result<(), String> {
        match entity_type {
            EntityType::Node => {
                let len = self.get_label_matrix_mut(label).lock().unwrap().nvals();
                for attr in attrs {
                    self.get_or_add_node_attribute_id(attr);
                }
                {
                    let mut node_indexer = self.node_indexer.lock().unwrap();
                    node_indexer.create_index(index_type, label.clone(), attrs, len)?;
                }
                self.populate_index(label);
            }
            EntityType::Relationship => {}
        }
        Ok(())
    }

    fn populate_index(
        &self,
        label: &Arc<String>,
    ) {
        let lm = self.get_label_matrix(label).unwrap();
        let label = label.clone();
        INDEXER_CHANNEL.send((
            0,
            label,
            lm,
            self.node_indexer.clone(),
            self.node_attrs.clone(),
            self.node_attrs_name.clone(),
        ));
    }

    pub fn commit_index(
        &mut self,
        index_add_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let mut add_docs = HashMap::new();
        let mut node_indexer = self.node_indexer.lock().unwrap();
        let node_attrs = self.node_attrs.lock().unwrap();
        for (label, ids) in index_add_docs.drain() {
            let fields = node_indexer.get_fields(label.clone());
            let mut docs = vec![];
            for id in ids {
                let mut doc = Document::new(id);
                let attrs = node_attrs.get(&NodeId(id)).unwrap();
                for (key, fields) in &fields {
                    let attr_id = self.get_node_attribute_id(key.as_str()).unwrap();
                    let value = attrs.get(&attr_id).cloned().unwrap();
                    for field in fields {
                        doc.set(field.clone(), value.clone());
                    }
                }
                docs.push(doc);
            }
            add_docs.insert(label, docs);
        }

        node_indexer.commit(&mut add_docs, remove_docs);
    }

    pub fn drop_index(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
    ) {
        match entity_type {
            EntityType::Node => {
                let mut node_indexer = self.node_indexer.lock().unwrap();
                let all_attrs = node_indexer
                    .get_fields(label.clone())
                    .iter()
                    .filter_map(|(k, fields)| {
                        if fields.iter().any(|f| f.ty == IndexType::Fulltext) {
                            Some(k.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let total = self
                    .get_label_matrix(label)
                    .unwrap()
                    .lock()
                    .unwrap()
                    .nvals();
                let reindex = node_indexer
                    .drop_index(
                        label.clone(),
                        if index_type == &IndexType::Fulltext {
                            &all_attrs
                        } else {
                            attrs
                        },
                        index_type,
                        total,
                    )
                    .is_some();
                node_indexer.disable(label.clone());
                drop(node_indexer);
                if reindex {
                    self.populate_index(label);
                } else {
                    INDEXER_CHANNEL.send((
                        1,
                        label.clone(),
                        Arc::new(Mutex::new(Matrix::new(0, 0))),
                        self.node_indexer.clone(),
                        self.node_attrs.clone(),
                        self.node_attrs_name.clone(),
                    ));
                }
            }
            EntityType::Relationship => {}
        }
    }

    pub fn is_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        let node_indexer = self.node_indexer.lock().unwrap();
        node_indexer.is_attr_indexed(label.clone(), field.clone())
    }

    pub fn get_indexed_nodes(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> Vec<NodeId> {
        let node_indexer = self.node_indexer.lock().unwrap();
        node_indexer
            .query(label.clone(), query)
            .into_iter()
            .map(NodeId)
            .collect()
    }

    pub fn index_info(&self) -> Vec<IndexInfo> {
        let node_indexer = self.node_indexer.lock().unwrap();
        node_indexer.index_info()
    }
}

pub struct MvccGraph {
    graph: Arc<Graph>,
}

impl MvccGraph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
    ) -> Self {
        Self {
            graph: Arc::new(Graph::new(n, e, cache_size)),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<Graph> {
        self.graph.clone()
    }

    pub fn write<F: Fn(Graph) -> Result<Graph, String>>(
        &mut self,
        f: F,
    ) -> Result<(), String> {
        self.graph = Arc::new(f(self.graph.new_version())?);
        Ok(())
    }
}
