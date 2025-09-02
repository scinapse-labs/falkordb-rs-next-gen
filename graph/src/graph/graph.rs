use std::{
    collections::HashMap,
    hash::Hash,
    num::NonZeroUsize,
    rc::Rc,
    sync::{Arc, Mutex, RwLock, mpsc},
    time::{Duration, Instant},
};

use atomic_refcell::AtomicRefCell;
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
        versioned_matrix::VersionedMatrix,
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

#[derive(Clone)]
struct AttributeStore<T: Eq + Hash + Clone + Send + Sync> {
    attributes: Arc<RwLock<HashMap<T, OrderMap<AttrId, Value>>>>,
}

impl<T: Eq + Hash + Clone + Send + Sync> AttributeStore<T> {
    pub fn new() -> Self {
        Self {
            attributes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn remove(
        &mut self,
        key: &T,
    ) -> Option<OrderMap<AttrId, Value>> {
        self.attributes.write().unwrap().remove(key)
    }

    pub fn contains_key(
        &self,
        key: &T,
    ) -> bool {
        self.attributes.read().unwrap().contains_key(key)
    }

    pub fn get(
        &self,
        key: &T,
    ) -> Option<OrderMap<AttrId, Value>> {
        self.attributes.read().unwrap().get(key).cloned()
    }

    pub fn remove_attr(
        &mut self,
        key: &T,
        attr_id: AttrId,
    ) -> bool {
        let mut attributes = self.attributes.write().unwrap();
        if let Some(attrs) = attributes.get_mut(key) {
            let has_value = attrs.remove(&attr_id).is_some();
            if attrs.is_empty() {
                attributes.remove(key);
            }
            has_value
        } else {
            false
        }
    }

    pub fn insert_attr(
        &mut self,
        key: T,
        attr_id: AttrId,
        value: Value,
    ) -> bool {
        let mut attributes = self.attributes.write().unwrap();
        attributes
            .entry(key)
            .or_default()
            .insert(attr_id, value)
            .is_some()
    }

    pub fn new_version(&self) -> Self {
        let attrs = self.attributes.read().unwrap();

        Self {
            attributes: Arc::new(RwLock::new((*attrs).clone())),
        }
    }
}

unsafe impl<T: Eq + Hash + Clone + Send + Sync> Send for AttributeStore<T> {}
unsafe impl<T: Eq + Hash + Clone + Send + Sync> Sync for AttributeStore<T> {}

#[derive(Clone)]
pub struct Graph {
    node_cap: u64,
    relationship_cap: u64,
    reserved_node_count: u64,
    reserved_relationship_count: u64,
    node_count: u64,
    relationship_count: u64,
    deleted_nodes: RoaringTreemap,
    deleted_relationships: RoaringTreemap,
    zero_matrix: Matrix,
    adjacancy_matrix: Matrix,
    node_labels_matrix: VersionedMatrix,
    relationship_type_matrix: Matrix,
    all_nodes_matrix: Matrix,
    labels_matices: Vec<VersionedMatrix>,
    relationship_matrices: Vec<Tensor>,
    node_attrs: AttributeStore<NodeId>,
    relationship_attrs: AttributeStore<RelationshipId>,
    node_indexer: Indexer,
    node_labels: Vec<Arc<String>>,
    relationship_types: Vec<Arc<String>>,
    node_attrs_name: Vec<Arc<String>>,
    relationship_attrs_name: Vec<Arc<String>>,
    cache: Arc<Mutex<LruCache<String, PlanTree>>>,
    pub version: u64,
}

struct PlanTree(DynTree<IR>);

#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for PlanTree {}
unsafe impl Sync for PlanTree {}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

type IndexAction = (
    i32,
    Arc<String>,
    Option<VersionedMatrix>,
    Indexer,
    AttributeStore<NodeId>,
    Vec<Arc<String>>,
);

static INDEXER_CHANNEL: Lazy<mpsc::Sender<IndexAction>> = Lazy::new(|| {
    let (sender, receiver) = mpsc::channel::<IndexAction>();
    spawn(move || {
        loop {
            let Ok((action, label, lm, mut node_indexer, node_attrs, node_attrs_name)) =
                receiver.recv()
            else {
                break;
            };

            if action == 0 {
                if node_indexer.pending_changes(label.clone()) > 1 {
                    node_indexer.enable(label);
                    continue;
                }
                let attr_ids = {
                    node_indexer
                        .get_fields(label.clone())
                        .into_iter()
                        .filter_map(|(attr, field)| {
                            node_attrs_name
                                .iter()
                                .position(|p| p.as_str() == attr.as_str())
                                .map(AttrId)
                                .map(|id| (id, field))
                        })
                        .collect::<Vec<_>>()
                };
                let mut add_docs = vec![];
                let value = lm.unwrap().iter(0, u64::MAX);
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
                if node_indexer.pending_changes(label.clone()) > 1 {
                    node_indexer.enable(label);
                    continue;
                }
                let mut index_add_docs = HashMap::new();
                index_add_docs.insert(label.clone(), add_docs);

                node_indexer.commit(&mut index_add_docs, &mut HashMap::new());
                node_indexer.enable(label);
            } else if action == 1 {
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
        version: u64,
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
            zero_matrix: Matrix::new(0, 0),
            adjacancy_matrix: Matrix::new(n, n),
            node_labels_matrix: VersionedMatrix::new(0, 0),
            relationship_type_matrix: Matrix::new(0, 0),
            all_nodes_matrix: Matrix::new(n, n),
            labels_matices: Vec::new(),
            relationship_matrices: Vec::new(),
            node_attrs: AttributeStore::new(),
            relationship_attrs: AttributeStore::new(),
            node_indexer: Indexer::default(),
            node_labels: Vec::new(),
            relationship_types: Vec::new(),
            node_attrs_name: Vec::new(),
            relationship_attrs_name: Vec::new(),
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size).unwrap(),
            ))),
            version,
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
            labels_matices: self
                .labels_matices
                .iter()
                .map(VersionedMatrix::dup)
                .collect(),
            relationship_matrices: self.relationship_matrices.iter().map(Tensor::dup).collect(),
            node_attrs: self.node_attrs.new_version(),
            relationship_attrs: self.relationship_attrs.new_version(),
            node_indexer: self.node_indexer.clone(),
            node_labels: self.node_labels.clone(),
            relationship_types: self.relationship_types.clone(),
            node_attrs_name: self.node_attrs_name.clone(),
            relationship_attrs_name: self.relationship_attrs_name.clone(),
            cache: self.cache.clone(),
            version: self.version + 1,
        }
    }

    #[must_use]
    pub const fn get_labels_count(&self) -> usize {
        self.node_labels.len()
    }

    #[must_use]
    pub fn get_labels(&self) -> Vec<Arc<String>> {
        self.node_labels.clone()
    }

    #[must_use]
    pub fn get_label_by_id(
        &self,
        id: LabelId,
    ) -> Arc<String> {
        self.node_labels[id.0].clone()
    }

    #[must_use]
    pub fn get_types(&self) -> Vec<Arc<String>> {
        self.relationship_types.clone()
    }

    #[must_use]
    pub fn get_type(
        &self,
        id: TypeId,
    ) -> Option<Arc<String>> {
        self.relationship_types.get(id.0).cloned()
    }

    #[must_use]
    pub fn get_attrs(&self) -> Vec<Arc<String>> {
        self.node_attrs_name
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
                    let optimize_plan = optimize(&plan.0, self);
                    Ok(Plan::new(
                        Rc::new(optimize_plan),
                        true,
                        parameters,
                        parse_duration,
                        plan_duration,
                    ))
                } else {
                    drop(cache);
                    let start = Instant::now();
                    let ir = parser.parse()?;
                    parse_duration = start.elapsed();

                    let mut planner = Planner::default();
                    let start = Instant::now();
                    let plan = planner.plan(ir);
                    let optimize_plan = optimize(&plan, self);
                    plan_duration = start.elapsed();

                    self.cache
                        .lock()
                        .unwrap()
                        .push(query.to_string(), PlanTree(plan));
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
    ) -> Option<VersionedMatrix> {
        self.node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(|i| self.labels_matices[i].clone())
    }

    fn get_label_matrix_mut(
        &mut self,
        label: &Arc<String>,
    ) -> VersionedMatrix {
        if !self.node_labels.contains(label) {
            self.node_labels.push(label.clone());

            let m = VersionedMatrix::new(self.node_cap, self.node_cap);
            self.labels_matices
                .insert(self.node_labels.len() - 1, m.clone());
            return m;
        }

        self.get_label_matrix(label).unwrap()
    }

    fn get_relationship_matrix_mut(
        &mut self,
        relationship_type: &Arc<String>,
    ) -> Tensor {
        if !self.relationship_types.contains(relationship_type) {
            self.relationship_types.push(relationship_type.clone());

            self.relationship_matrices.insert(
                self.relationship_types.len() - 1,
                Tensor::new(self.node_cap, self.node_cap),
            );
        }

        self.relationship_types
            .iter()
            .position(|l| l.as_str() == relationship_type.as_str())
            .map(|i| self.relationship_matrices[i].clone())
            .unwrap()
    }

    fn get_relationship_matrix(
        &self,
        relationship_type: &Arc<String>,
    ) -> Option<Tensor> {
        if !self.relationship_types.contains(relationship_type) {
            return None;
        }

        self.relationship_types
            .iter()
            .position(|l| l.as_str() == relationship_type.as_str())
            .map(|i| self.relationship_matrices[i].clone())
    }

    pub fn get_node_attribute_id(
        &self,
        key: &str,
    ) -> Option<AttrId> {
        self.node_attrs_name
            .iter()
            .position(|p| p.as_str() == key)
            .map(AttrId)
    }

    #[must_use]
    pub fn get_node_attribute_string(
        &self,
        id: AttrId,
    ) -> Option<Arc<String>> {
        self.node_attrs_name.get(id.0).cloned()
    }

    pub fn get_or_add_node_attribute_id(
        &mut self,
        key: &Arc<String>,
    ) -> AttrId {
        AttrId(
            self.node_attrs_name
                .iter()
                .position(|p| p.as_str() == key.as_str())
                .unwrap_or_else(|| {
                    let len = self.node_attrs_name.len();
                    self.node_attrs_name.push(key.clone());
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

    #[must_use]
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
        if value == Value::Null {
            let removed = self.node_attrs.remove_attr(&id, attr_id);

            if removed {
                if self.node_attrs.contains_key(&id) {
                    for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                        let label = self.node_labels[label_id as usize].clone();
                        if self
                            .node_indexer
                            .is_attr_indexed(label.clone(), attr_name.clone())
                        {
                            index_add_docs
                                .entry(label)
                                .or_default()
                                .insert(u64::from(id));
                        }
                    }
                } else {
                    for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                        let label = self.node_labels[label_id as usize].clone();
                        if self
                            .node_indexer
                            .is_attr_indexed(label.clone(), attr_name.clone())
                        {
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
            let res = self.node_attrs.insert_attr(id, attr_id, value);
            for (_, label_id) in self.node_labels_matrix.iter(id.into(), id.into()) {
                let label = self.node_labels[label_id as usize].clone();
                if self
                    .node_indexer
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
            let mut label_matrix = self.get_label_matrix_mut(label);
            label_matrix.set(id.0, id.0, true);
            let label_id = self.get_label_id(label).unwrap();
            self.resize();
            self.node_labels_matrix.set(id.0, label_id.0 as u64, true);
            if self.node_indexer.is_label_indexed(label.clone())
                && self.node_attrs.contains_key(&id)
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
            let mut label_matrix = self.get_label_matrix_mut(label);
            label_matrix.remove(id.0, id.0);
            let label_id = self.get_label_id(label).unwrap();
            self.node_labels_matrix.remove(id.0, label_id.0 as u64);
            if self.node_indexer.is_label_indexed(label.clone()) {
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

        for label_matrix in &mut self.labels_matices {
            label_matrix.remove(id.0, id.0);
        }
        for label_id in 0..self.labels_matices.len() {
            let label = self.node_labels[label_id].clone();
            self.node_labels_matrix.remove(id.0, label_id as _);
            let mut indexed = false;
            if let Some(attrs) = self.node_attrs.get(&id) {
                for (attr_id, _) in attrs {
                    let attr_name = self.get_node_attribute_string(attr_id).unwrap();
                    if self.node_indexer.is_attr_indexed(label.clone(), attr_name) {
                        indexed = true;
                        break;
                    }
                }
            }
            if indexed {
                remove_docs
                    .entry(label.clone())
                    .or_default()
                    .insert(u64::from(id));
            }
        }

        self.node_attrs.remove(&id);
    }

    pub fn get_node_relationships(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + '_ {
        self.relationship_matrices
            .iter()
            .flat_map(move |m| m.iter(id.0, id.0, false).chain(m.iter(id.0, id.0, true)))
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
                    let mut m = iter.next().unwrap().to_matrix();
                    for label_matrix in iter {
                        m.element_wise_multiply(&label_matrix.to_matrix());
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

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr_id: AttrId,
    ) -> Option<Value> {
        self.node_attrs
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
            self.get_relationship_matrix_mut(type_name)
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
        if value == Value::Null {
            self.relationship_attrs.remove_attr(&id, attr_id)
        } else {
            self.relationship_attrs.insert_attr(id, attr_id, value)
        }
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.0)
    }

    #[must_use]
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
            self.get_relationship_matrix_mut(&label).remove_all(rels);
        }
    }

    #[must_use]
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
                for id in relationship_matrix.get(src.0, dest.0) {
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
            let mut m = iter
                .next()
                .map_or_else(|| &self.adjacancy_matrix, Tensor::matrix)
                .dup();
            for relationship_matrix in iter {
                m.element_wise_add(relationship_matrix.matrix());
            }

            if !src_labels_matrices.is_empty() {
                let mut iter = src_labels_matrices.iter();
                let mut src_matrix = iter.next().unwrap().to_matrix();
                for label_matrix in iter {
                    src_matrix.element_wise_multiply(&label_matrix.to_matrix());
                }
                m.rmxm(&src_matrix);
            }
            if !dest_labels_matrices.is_empty() {
                let mut iter = dest_labels_matrices.iter();
                let mut dest_matrix = iter.next().unwrap().to_matrix();
                for label_matrix in iter {
                    dest_matrix.element_wise_multiply(&label_matrix.to_matrix());
                }
                m.lmxm(&dest_matrix);
            }
            m.iter(0, u64::MAX)
        } else {
            self.zero_matrix.iter(0, u64::MAX)
        };

        iter.map(|(src, dest)| (NodeId(src), NodeId(dest)))
    }

    #[must_use]
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

    #[must_use]
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
            for label_matrix in &mut self.labels_matices {
                label_matrix.resize(self.node_cap, self.node_cap);
            }
            for relationship_matrix in &mut self.relationship_matrices {
                relationship_matrix.resize(self.node_cap, self.node_cap);
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

    #[must_use]
    pub fn get_node_attrs(
        &self,
        id: NodeId,
    ) -> OrderMap<AttrId, Value> {
        self.node_attrs.get(&id).unwrap_or_default()
    }

    #[must_use]
    pub fn get_relationship_attrs(
        &self,
        id: RelationshipId,
    ) -> OrderMap<AttrId, Value> {
        self.relationship_attrs.get(&id).unwrap_or_default()
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
                let len = self.get_label_matrix_mut(label).nvals();
                for attr in attrs {
                    self.get_or_add_node_attribute_id(attr);
                }
                {
                    self.node_indexer
                        .create_index(index_type, label.clone(), attrs, len)?;
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
        INDEXER_CHANNEL
            .send((
                0,
                label,
                Some(lm),
                self.node_indexer.clone(),
                self.node_attrs.clone(),
                self.node_attrs_name.clone(),
            ))
            .unwrap();
    }

    pub fn commit_index(
        &mut self,
        index_add_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
        remove_docs: &mut HashMap<Arc<String>, RoaringTreemap>,
    ) {
        let mut add_docs = HashMap::new();
        for (label, ids) in index_add_docs.drain() {
            let fields = self.node_indexer.get_fields(label.clone());
            let mut docs = vec![];
            for id in ids {
                let mut doc = Document::new(id);
                let attrs = self.node_attrs.get(&NodeId(id)).unwrap();
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

        self.node_indexer.commit(&mut add_docs, remove_docs);
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
                let all_attrs = self
                    .node_indexer
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
                let total = self.get_label_matrix(label).unwrap().nvals();
                let reindex = self
                    .node_indexer
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
                self.node_indexer.disable(label.clone());

                if reindex {
                    self.populate_index(label);
                } else {
                    INDEXER_CHANNEL
                        .send((
                            1,
                            label.clone(),
                            None,
                            self.node_indexer.clone(),
                            self.node_attrs.clone(),
                            self.node_attrs_name.clone(),
                        ))
                        .unwrap();
                }
            }
            EntityType::Relationship => {}
        }
    }

    #[must_use]
    pub fn is_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
    ) -> bool {
        self.node_indexer
            .is_attr_indexed(label.clone(), field.clone())
    }

    pub fn get_indexed_nodes(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> Vec<NodeId> {
        self.node_indexer
            .query(label.clone(), query)
            .into_iter()
            .map(NodeId)
            .collect()
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        self.node_indexer.index_info()
    }
}

pub struct MvccGraph {
    graph: Arc<AtomicRefCell<Graph>>,
}

unsafe impl Send for MvccGraph {}
unsafe impl Sync for MvccGraph {}

impl MvccGraph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
    ) -> Self {
        Self {
            graph: Arc::new(AtomicRefCell::new(Graph::new(n, e, cache_size, 0))),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    #[must_use]
    pub fn write(&self) -> Arc<AtomicRefCell<Graph>> {
        Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()))
    }

    pub fn commit(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) {
        if self.graph.borrow().version + 1 == new_graph.borrow().version {
            self.graph = new_graph;
            self.graph.borrow().node_labels_matrix.wait();
        } else {
            todo!();
        }
    }
}
