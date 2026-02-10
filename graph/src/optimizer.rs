//! Query plan optimization passes.
//!
//! The optimizer transforms the logical execution plan to improve performance.
//! Current optimizations include:
//!
//! ## Index Utilization
//!
//! Replaces `NodeByLabelScan` + `Filter` with `NodeByIndexScan` when:
//! - A range index exists on the filtered property
//! - The filter uses equality (=), less than (<), or greater than (>)
//!
//! Example transformation:
//! ```text
//! Before: NodeByLabelScan(:Person) → Filter(n.age = 30)
//! After:  NodeByIndexScan(:Person, age, Equal(30))
//! ```
//!
//! ## Node By ID Optimization
//!
//! Replaces label scan + ID filter with direct ID lookup:
//! ```text
//! Before: NodeByLabelScan(:Person) → Filter(id(n) = 42)
//! After:  NodeByIdScan(:Person, 42)
//! ```
//!
//! ## Good Practice
//!
//! The optimizer uses a collect-then-iterate pattern when modifying the tree
//! to avoid issues with mutable iteration. This is a common pattern when
//! working with tree structures that need in-place modification.

use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::{
    ast::{ExprIR, QueryExpr, QueryNode, Variable},
    graph::graph::Graph,
    indexer::IndexQuery,
    planner::IR,
    runtime::functions::{FnType, get_functions},
    tree,
};

type IndexScanResult = Option<(
    Arc<QueryNode<Arc<String>, Variable>>,
    Arc<String>,
    Arc<IndexQuery<QueryExpr<Variable>>>,
)>;

fn extract_attribute_from_subtree(
    tree: &DynTree<ExprIR<Variable>>,
    root_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
) -> Option<Arc<String>> {
    let indices = tree.node(root_idx).indices::<Bfs>().collect::<Vec<_>>();
    for idx in indices {
        let node = tree.node(idx);
        if let ExprIR::FuncInvocation(func) = node.data() {
            if func.name == "property" {
                if let ExprIR::String(attr) = node.child(1).data() {
                    return Some(attr.clone());
                }
            }
        }
    }
    None
}

/// Extract the attribute and the expression from the filter
///
/// Look for single attribute (either left or right child)
/// And a an expression that does not depends on graph entity - so we can pass it to the filter for a single query
///
/// Returns the attribute, the expression, and the child index of the attribute.
fn extract_attribute_and_expression_from_filter(
    filter: &DynTree<ExprIR<Variable>>
) -> Option<(Arc<String>, usize)> {
    let lhs_idx = filter.root().child(0).idx();
    let rhs_idx = filter.root().child(1).idx();

    match (
        extract_attribute_from_subtree(filter, lhs_idx),
        extract_attribute_from_subtree(filter, rhs_idx),
    ) {
        (Some(_), Some(_)) => None,
        (Some(attr), _) => Some((attr, 0)),
        (_, Some(attr)) => Some((attr, 1)),
        _ => None,
    }
}

/// Builds an index scan for a simple property filter (e.g. `n.age = 30`).
fn try_property_index_scan(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    attr: &Arc<String>,
    op: &ExprIR<Variable>,
    constant_node: DynTree<ExprIR<Variable>>,
) -> IndexScanResult {
    match op {
        ExprIR::Eq => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Equal(attr.clone(), Arc::new(constant_node))),
        )),
        ExprIR::Gt => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range(
                attr.clone(),
                Some(Arc::new(constant_node)),
                None,
            )),
        )),
        ExprIR::Lt => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range(
                attr.clone(),
                None,
                Some(Arc::new(constant_node)),
            )),
        )),
        _ => None,
    }
}

/// Builds an index scan for a distance filter (e.g. `distance(n.loc, point(...)) < 100`).
fn try_distance_index_scan(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    attr: &Arc<String>,
    filter: &DynTree<ExprIR<Variable>>,
    attribute_side: usize,
    constant_node: DynTree<ExprIR<Variable>>,
) -> IndexScanResult {
    let attribute_side_idx = filter.root().child(attribute_side).idx();
    let child_0_idx = filter.node(attribute_side_idx).child(0).idx();
    let child_1_idx = filter.node(attribute_side_idx).child(1).idx();
    match (
        extract_attribute_from_subtree(filter, child_0_idx),
        extract_attribute_from_subtree(filter, child_1_idx),
    ) {
        (Some(_), None) => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Point {
                key: attr.clone(),
                point: Arc::new(filter.node(child_1_idx).clone_as_tree()),
                radius: Arc::new(constant_node),
            }),
        )),
        (None, Some(_)) => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Point {
                key: attr.clone(),
                point: Arc::new(filter.node(child_0_idx).clone_as_tree()),
                radius: Arc::new(constant_node),
            }),
        )),
        _ => None,
    }
}

/// Attempts to replace label scans with index scans where applicable.
///
/// Scans the plan for patterns like:
/// `NodeByLabelScan` → `Filter(property = value)`
///
/// If an index exists on the filtered property, replaces with `NodeByIndexScan`.
fn utilize_index(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

    for idx in indices {
        // Try to replace label scans with index scans where applicable.
        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(idx).data()
            && !node.labels.is_empty()
            && let IR::Filter(filter) = optimized_plan.node(idx).parent().unwrap().data()
            // If the filter is an equality, greater than, or less than expression
            && matches!(filter.root().data(), ExprIR::Eq | ExprIR::Gt | ExprIR::Lt)
            // If we managed to extract the attribute and expression from the filter
            && let Some((attr, attribute_side)) = extract_attribute_and_expression_from_filter(&filter)
            // If the attribute is indexed
            && graph.is_indexed(&node.labels[0], &attr)
        {
            // Check if the attribute side is a propetry function or more complexed function
            // If it is a property function, we can handle it right away since we know everything: Attribute, expression and operation
            // If it is a more complexed function, we need to understand which function it is and if we can apply an index scan on.
            if let ExprIR::FuncInvocation(func) = filter.root().child(attribute_side).data() {
                let constant_node = filter.root().child(1 - attribute_side).clone_as_tree();
                match func.name.as_str() {
                    "property" => {
                        try_property_index_scan(node, &attr, filter.root().data(), constant_node)
                    }
                    "distance" => {
                        try_distance_index_scan(node, &attr, filter, attribute_side, constant_node)
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some((node, index, query)) = node {
            let mut op = optimized_plan.node_mut(idx);
            *op.data_mut() = IR::NodeByIndexScan { node, index, query };
            op.parent_mut().unwrap().take_out();
            break;
        }

        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(idx).data() {
            get_index(graph, node)
        } else {
            None
        };
        if let Some((node, attr, filter)) = node
            && !node.labels.is_empty()
        {
            let mut op = optimized_plan.node_mut(idx);
            *op.data_mut() = IR::NodeByIndexScan {
                node: node.clone(),
                index: node.labels[0].clone(),
                query: Arc::new(IndexQuery::Equal(
                    attr.clone(),
                    Arc::new(filter.root().child(1).clone_as_tree()),
                )),
            };
        }
    }
}

/// Replaces label scan + ID filter with direct node ID lookup.
fn utilize_node_by_id(optimized_plan: &mut DynTree<IR>) {
    let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

    for idx in indices {
        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(idx).data()
            && let IR::Filter(filter) = optimized_plan.node(idx).parent().unwrap().data()
            && matches!(filter.root().data(), ExprIR::Eq | ExprIR::Gt | ExprIR::Lt)
            && let ExprIR::FuncInvocation(inner_func) = filter.root().child(0).data()
            && inner_func.name == "id"
            && let ExprIR::Variable(var) = filter.root().child(0).child(0).data()
            && node.alias == *var
        {
            Some((
                node.clone(),
                Arc::new(filter.root().child(1).clone_as_tree()),
                filter.root().data().clone(),
            ))
        } else {
            None
        };
        if let Some((node, id, op)) = node {
            let mut new_op = optimized_plan.node_mut(idx);
            *new_op.data_mut() = IR::NodeByIdScan { node, id, op };
            new_op.parent_mut().unwrap().take_out();
        }
    }
}

/// Optimizes a query execution plan.
///
/// Applies all optimization passes to the plan and returns the optimized version.
/// The original plan is not modified.
///
/// # Arguments
/// * `plan` - The unoptimized execution plan
/// * `graph` - The graph (needed to check for index availability)
///
/// # Returns
/// An optimized copy of the plan
#[must_use]
pub fn optimize(
    plan: &DynTree<IR>,
    graph: &Graph,
) -> DynTree<IR> {
    let mut optimized_plan = plan.clone();

    utilize_index(&mut optimized_plan, graph);
    utilize_node_by_id(&mut optimized_plan);

    optimized_plan
}

/// Checks if a node pattern has an indexed property filter.
fn get_index(
    graph: &Graph,
    node: &Arc<QueryNode<Arc<String>, Variable>>,
) -> Option<(
    Arc<QueryNode<Arc<String>, Variable>>,
    Arc<String>,
    DynTree<ExprIR<Variable>>,
)> {
    for label in node.labels.iter() {
        for attr in node.attrs.root().children() {
            if let ExprIR::String(attr_str) = attr.data()
                && graph.is_indexed(label, attr_str)
            {
                return Some((
                    node.clone(),
                    attr_str.clone(),
                    tree!(
                        ExprIR::Eq,
                        tree!(
                            ExprIR::FuncInvocation(
                                get_functions().get("property", &FnType::Internal).unwrap()
                            ),
                            tree!(ExprIR::Variable(node.alias.clone())),
                            tree!(ExprIR::String(attr_str.clone()))
                        ),
                        attr.child(0).as_cloned_subtree()
                    ),
                ));
            }
        }
    }
    None
}
