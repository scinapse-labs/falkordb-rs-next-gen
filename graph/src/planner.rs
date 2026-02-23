//! Query plan generation from bound AST.
//!
//! The planner converts a bound Cypher AST into a logical execution plan (IR tree).
//! This phase determines the order of operations and which algorithms to use for
//! pattern matching.
//!
//! ## Plan Structure
//!
//! The plan is a tree where:
//! - Leaf nodes produce tuples (scans, argument)
//! - Internal nodes transform/filter tuples from children
//! - The root produces the final result
//!
//! ## Key Planning Decisions
//!
//! 1. **Scan selection**: Chooses between label scans, index scans, or ID lookups
//! 2. **Join ordering**: Determines order of pattern matching for efficiency
//! 3. **Projection placement**: Decides when to project/aggregate
//! 4. **Filter pushdown**: Places filters as early as possible
//!
//! ## IR Operators
//!
//! - **NodeByLabelScan**: Scan all nodes with a label
//! - **NodeByIndexScan**: Use an index for node lookup
//! - **CondTraverse**: Traverse relationships conditionally
//! - **ExpandInto**: Check for relationship between known nodes
//! - **Filter**: Apply predicate to filter tuples
//! - **Project**: Compute new values from existing
//! - **Aggregate**: Group and aggregate tuples
//! - **Sort/Skip/Limit**: Order and paginate results

use std::{collections::HashSet, fmt::Display, sync::Arc};

use orx_tree::{DynTree, NodeRef, Side, Traversal, Traverser};

use crate::{
    ast::{
        BoundQueryIR, ExprIR, QueryExpr, QueryGraph, QueryIR, QueryNode, QueryPath,
        QueryRelationship, SetItem, SupportAggregation, Variable,
    },
    indexer::{EntityType, IndexQuery, IndexType},
    runtime::functions::GraphFn,
    tree,
};

/// Intermediate Representation (IR) for execution plan operators.
///
/// Each variant represents a physical operation in the query execution plan.
/// The plan forms a tree where data flows from leaves to root.
#[derive(Clone, Debug)]
pub enum IR {
    /// Empty result set (used as placeholder)
    Empty,
    /// Receives input from parent operator
    Argument,
    /// OPTIONAL MATCH - returns nulls if no match
    Optional(Vec<Variable>),
    /// CALL procedure with arguments, yielding outputs
    ProcedureCall(Arc<GraphFn>, Vec<QueryExpr<Variable>>, Vec<Variable>),
    /// UNWIND list AS variable
    Unwind(QueryExpr<Variable>, Variable),
    /// CREATE pattern
    Create(QueryGraph<Arc<String>, Arc<String>, Variable>),
    /// MERGE pattern with ON CREATE/ON MATCH actions
    Merge(
        QueryGraph<Arc<String>, Arc<String>, Variable>,
        Vec<SetItem<Arc<String>, Variable>>,
        Vec<SetItem<Arc<String>, Variable>>,
    ),
    /// DELETE entities (detach flag for relationships)
    Delete(Vec<QueryExpr<Variable>>, bool),
    /// SET properties/labels
    Set(Vec<SetItem<Arc<String>, Variable>>),
    /// REMOVE properties/labels
    Remove(Vec<QueryExpr<Variable>>),
    /// Scan nodes by label
    NodeByLabelScan(Arc<QueryNode<Arc<String>, Variable>>),
    /// Scan nodes using an index
    NodeByIndexScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
    },
    /// Lookup node by label and id
    NodeByLabelAndIdScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Lookup node by id only
    NodeByIdSeek {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Traverse relationships from known nodes
    CondTraverse(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Check relationship between two known nodes
    ExpandInto(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Build path objects from matched patterns
    PathBuilder(Vec<Arc<QueryPath<Variable>>>),
    /// Apply filter predicate
    Filter(QueryExpr<Variable>),
    /// Cartesian product of child results
    CartesianProduct,
    /// Load CSV file
    LoadCsv {
        file_path: QueryExpr<Variable>,
        headers: bool,
        delimiter: QueryExpr<Variable>,
        var: Variable,
    },
    /// Sort by expressions (bool = descending)
    Sort(Vec<(QueryExpr<Variable>, bool)>),
    /// Skip first N rows
    Skip(QueryExpr<Variable>),
    /// Limit to N rows
    Limit(QueryExpr<Variable>),
    /// Aggregate with grouping keys, aggregations, and projections
    Aggregate(
        Vec<Variable>,
        Vec<(Variable, QueryExpr<Variable>)>,
        Vec<(Variable, QueryExpr<Variable>)>,
    ),
    /// Project expressions to new variables
    Project(
        Vec<(Variable, QueryExpr<Variable>)>,
        Vec<(Variable, Variable)>,
    ),
    /// Remove duplicate rows
    Distinct,
    /// Commit write operations to graph
    Commit,
    /// CREATE INDEX operation
    CreateIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
        options: Option<QueryExpr<Variable>>,
    },
    /// DROP INDEX operation
    DropIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
    },
}

#[cfg_attr(tarpaulin, skip)]
impl Display for IR {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty"),
            Self::Argument => write!(f, "Argument"),
            Self::Optional(_) => write!(f, "Optional"),
            Self::ProcedureCall(_, _, _) => write!(f, "ProcedureCall"),
            Self::Unwind(_, _) => {
                write!(f, "Unwind")
            }
            Self::Create(pattern) => write!(f, "Create | {pattern}"),
            Self::Merge(pattern, _, _) => write!(f, "Merge | {pattern}"),
            Self::Delete(_, _) => write!(f, "Delete"),
            Self::Set(_) => write!(f, "Set"),
            Self::Remove(_) => write!(f, "Remove"),
            Self::NodeByLabelScan(node) => write!(f, "Node By Label Scan | {node}"),
            Self::NodeByIndexScan { node, .. } => {
                write!(f, "Node By Index Scan | {node}")
            }
            Self::NodeByLabelAndIdScan { node, .. } => {
                write!(f, "Node By Label and ID Scan | {node}")
            }
            Self::NodeByIdSeek { .. } => write!(f, "NodeByIdSeek"),
            Self::CondTraverse(rel) => write!(f, "Conditional Traverse | {rel}"),
            Self::ExpandInto(rel) => write!(f, "Expand Into | {rel}"),
            Self::PathBuilder(_) => write!(f, "PathBuilder"),
            Self::Filter(_) => write!(f, "Filter"),
            Self::CartesianProduct => write!(f, "Cartesian Product"),
            Self::LoadCsv { .. } => write!(f, "Load CSV"),
            Self::Sort(_) => write!(f, "Sort"),
            Self::Skip(_) => write!(f, "Skip"),
            Self::Limit(_) => write!(f, "Limit"),
            Self::Aggregate(_, _, _) => write!(f, "Aggregate"),
            Self::Project(_, _) => write!(f, "Project"),
            Self::Commit => write!(f, "Commit"),
            Self::Distinct => write!(f, "Distinct"),
            Self::CreateIndex { label, attrs, .. } => {
                write!(f, "Create Index | :{label}({attrs:?})")
            }
            Self::DropIndex { label, attrs, .. } => {
                write!(f, "Drop Index | :{label}({attrs:?})")
            }
        }
    }
}

#[derive(Default)]
pub struct Planner {
    visited: HashSet<u32>,
}

impl Planner {
    fn add_argument_to_leaves(tree: &mut DynTree<IR>) {
        let mut tr = Traversal.bfs().over_nodes();

        let leaves: Vec<_> = tree
            .root()
            .walk_with(&mut tr)
            .filter(orx_tree::NodeRef::is_leaf)
            .map(|x| x.idx())
            .collect();

        // Add Argument node as a child to each leaf
        for leaf_idx in leaves {
            tree.node_mut(leaf_idx).push_child(IR::Argument);
        }
    }

    fn plan_match(
        &mut self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        filter: Option<QueryExpr<Variable>>,
    ) -> DynTree<IR> {
        let mut vec = vec![];
        for component in pattern.connected_components() {
            let relationships = component.relationships();
            let mut iter = relationships.into_iter();
            let Some(relationship) = iter.next() else {
                let nodes = component.nodes();
                debug_assert_eq!(nodes.len(), 1);
                let node = nodes[0].clone();
                let mut res = tree!(IR::NodeByLabelScan(node.clone()));
                self.visited.insert(node.alias.id);
                let paths = component.paths();
                if !paths.is_empty() {
                    res = tree!(IR::PathBuilder(paths), res);
                }
                vec.push(res);
                continue;
            };
            let mut res = if relationship.from.alias.id == relationship.to.alias.id {
                tree!(
                    IR::ExpandInto(relationship.clone()),
                    tree!(IR::NodeByLabelScan(relationship.from.clone()))
                )
            } else {
                tree!(IR::CondTraverse(relationship.clone()))
            };
            self.visited.insert(relationship.from.alias.id);
            self.visited.insert(relationship.to.alias.id);
            self.visited.insert(relationship.alias.id);
            for relationship in iter {
                res = if relationship.from.alias.id == relationship.to.alias.id {
                    tree!(
                        IR::ExpandInto(relationship.clone()),
                        tree!(IR::NodeByLabelScan(relationship.from.clone()), res)
                    )
                } else {
                    tree!(IR::CondTraverse(relationship.clone()), res)
                };
                self.visited.insert(relationship.from.alias.id);
                self.visited.insert(relationship.to.alias.id);
                self.visited.insert(relationship.alias.id);
            }
            let paths = component.paths();
            if !paths.is_empty() {
                res = tree!(IR::PathBuilder(paths), res);
            }
            vec.push(res);
        }
        let mut res = if vec.len() == 1 {
            vec.pop().unwrap()
        } else {
            tree!(IR::CartesianProduct; vec)
        };
        if let Some(filter) = filter {
            res = tree!(IR::Filter(filter), res);
        }
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_project(
        &mut self,
        exprs: Vec<(Variable, QueryExpr<Variable>)>,
        copy_from_parent: Vec<(Variable, Variable)>,
        orderby: Vec<(QueryExpr<Variable>, bool)>,
        skip: Option<QueryExpr<Variable>>,
        limit: Option<QueryExpr<Variable>>,
        filter: Option<QueryExpr<Variable>>,
        distinct: bool,
        write: bool,
    ) -> DynTree<IR> {
        for expr in &exprs {
            self.visited.insert(expr.0.id);
        }
        let mut res = if exprs.iter().any(|e| e.1.is_aggregation()) {
            let mut group_by_keys = Vec::new();
            let mut aggregations = Vec::new();
            let mut names = Vec::new();
            for (name, expr) in exprs {
                names.push(name.clone());
                if expr.is_aggregation() {
                    aggregations.push((name, expr));
                } else {
                    group_by_keys.push((name, expr));
                }
            }
            tree!(IR::Aggregate(names, group_by_keys, aggregations))
        } else {
            tree!(IR::Project(exprs, copy_from_parent))
        };
        if write {
            res.root_mut().push_child(IR::Commit);
        }
        if distinct {
            res = tree!(IR::Distinct, res);
        }
        if !orderby.is_empty() {
            res = tree!(IR::Sort(orderby), res);
        }
        if let Some(skip_expr) = skip {
            res = tree!(IR::Skip(skip_expr), res);
        }
        if let Some(limit_expr) = limit {
            res = tree!(IR::Limit(limit_expr), res);
        }
        if let Some(filter) = filter {
            res = tree!(IR::Filter(filter), res);
        }
        res
    }

    fn plan_query(
        &mut self,
        q: Vec<QueryIR<Variable>>,
        write: bool,
    ) -> DynTree<IR> {
        let mut plans = Vec::with_capacity(q.len());
        for ir in q {
            plans.push(self.plan(ir));
        }
        let mut iter = plans.into_iter().rev();
        let mut res = iter.next().unwrap();
        let mut idx = res.root().idx();
        while matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
            | IR::Limit(_)
            | IR::Distinct
            | IR::Filter(_))
        {
            idx = res.node(idx).child(0).idx();
        }
        if matches!(res.node(idx).data(), |IR::Project(_, _)| IR::Aggregate(
            _,
            _,
            _
        )) && res.node(idx).num_children() > 0
            && matches!(res.node(idx).child(0).data(), IR::Commit)
        {
            idx = res.node(idx).child(0).idx();
        }
        for n in iter {
            if res.node(idx).num_children() > 0 {
                idx = res
                    .node_mut(idx)
                    .child_mut(0)
                    .push_sibling_tree(Side::Left, n);
            } else {
                idx = res.node_mut(idx).push_child_tree(n);
            }
            while matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
                | IR::Limit(_)
                | IR::Distinct
                | IR::Filter(_))
            {
                idx = res.node(idx).child(0).idx();
            }
            if matches!(res.node(idx).data(), |IR::Project(_, _)| IR::Aggregate(
                _,
                _,
                _
            )) && res.node(idx).num_children() > 0
                && matches!(res.node(idx).child(0).data(), IR::Commit)
            {
                idx = res.node(idx).child(0).idx();
            }
        }
        if write {
            res = tree!(IR::Commit, res);
        }
        res
    }

    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn plan(
        &mut self,
        ir: BoundQueryIR,
    ) -> DynTree<IR> {
        match ir {
            QueryIR::Call(proc, exprs, named_outputs, filter) => {
                if let Some(filter) = filter {
                    return tree!(
                        IR::Filter(filter),
                        tree!(IR::ProcedureCall(proc, exprs, named_outputs))
                    );
                }
                if proc.name == "db.idx.fulltext.drop" {
                    let ExprIR::String(label) = exprs[0].root().data() else {
                        unreachable!()
                    };
                    return tree!(IR::DropIndex {
                        label: label.clone(),
                        attrs: vec![],
                        index_type: IndexType::Fulltext,
                        entity_type: EntityType::Node,
                    });
                }
                if proc.name == "db.idx.fulltext.createNodeIndex" {
                    let label = match exprs[0].root().data() {
                        ExprIR::String(label) => label.clone(),
                        ExprIR::Map => {
                            let mut ret = None;
                            for child in exprs[0].root().children() {
                                if let ExprIR::String(label) = child.data()
                                    && label.as_str() == "label"
                                {
                                    ret = Some(label.clone());
                                    break;
                                }
                            }
                            ret.unwrap_or_else(|| {
                                unreachable!();
                            })
                        }
                        _ => unreachable!(),
                    };
                    return tree!(IR::CreateIndex {
                        label,
                        attrs: vec![],
                        index_type: IndexType::Fulltext,
                        entity_type: EntityType::Node,
                        options: None,
                    });
                }
                tree!(IR::ProcedureCall(proc, exprs, named_outputs))
            }
            QueryIR::Match {
                pattern,
                filter,
                optional,
            } => {
                if optional {
                    tree!(
                        IR::Optional(
                            pattern
                                .variables()
                                .iter()
                                .filter(|v| !self.visited.contains(&v.id))
                                .cloned()
                                .collect()
                        ),
                        self.plan_match(&pattern, filter)
                    )
                } else {
                    self.plan_match(&pattern, filter)
                }
            }
            QueryIR::Unwind(expr, alias) => tree!(IR::Unwind(expr, alias)),
            QueryIR::Merge(pattern, on_create_set_items, on_match_set_items) => {
                let create_pattern = pattern.filter_visited(&self.visited);
                let mut match_branch = self.plan_match(&pattern, None);
                Self::add_argument_to_leaves(&mut match_branch);

                tree!(
                    IR::Merge(create_pattern, on_create_set_items, on_match_set_items),
                    match_branch
                )
            }
            QueryIR::Create(pattern) => {
                tree!(IR::Create(pattern.filter_visited(&self.visited)))
            }
            QueryIR::Delete(exprs, is_detach) => tree!(IR::Delete(exprs, is_detach)),
            QueryIR::Set(items) => tree!(IR::Set(items)),
            QueryIR::Remove(items) => tree!(IR::Remove(items)),
            QueryIR::LoadCsv {
                file_path,
                headers,
                delimiter,
                var,
            } => {
                tree!(IR::LoadCsv {
                    file_path,
                    headers,
                    delimiter,
                    var,
                })
            }
            QueryIR::With {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                distinct,
                write,
            ),
            QueryIR::Return {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                None,
                distinct,
                write,
            ),
            QueryIR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            } => tree!(IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options
            }),
            QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                tree!(IR::DropIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type
                })
            }
            QueryIR::Query(q, write) => self.plan_query(q, write),
            QueryIR::Union(_) => {
                // UNION execution is not yet implemented.
                // Currently, only column-name validation is performed (in the binder).
                todo!("UNION execution not yet implemented")
            }
        }
    }
}
