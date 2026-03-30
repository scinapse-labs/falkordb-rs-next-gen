//! Index utilization optimizer pass.
//!
//! Scans the execution plan for `NodeByLabelScan` operators that sit below a
//! `Filter` on an indexed property, and replaces the pair with a single
//! `NodeByIndexScan` that pushes the predicate into the index engine.
//!
//! ## Supported Patterns
//!
//! **Single comparison filter:**
//!
//! ```text
//! Before:                       After:
//!
//! Filter(n.age = 30)            NodeByIndexScan(:Person, age, Equal(30))
//!   |
//!   v
//! NodeByLabelScan(:Person)
//! ```
//!
//! **AND filter with multiple indexed conjuncts:**
//!
//! When a Filter contains `AND(n.year >= 1980, n.year < 1990)`, the pass
//! merges both conjuncts into a single `Range` index query:
//!
//! ```text
//! Before:                         After:
//!
//! Filter(AND(year>=1980,          NodeByIndexScan(:Movie, year,
//!            year<1990))            Range{min:1980, max:1990})
//!   |
//!   v
//! NodeByLabelScan(:Movie)
//! ```
//!
//! If only some AND conjuncts are indexable, the indexable ones are merged
//! into the scan and the remaining conjuncts stay as a reduced Filter.
//!
//! **Inline node attributes:**
//!
//! Also converts `NodeByLabelScan` nodes that carry inline property attributes
//! (e.g. `(n:Person {name: 'Alice'})`) into `NodeByIndexScan` when the
//! attribute is indexed.
//!
//! ## Supported operators and index types
//!
//! - Equality (`=`), less-than (`<`, `<=`), greater-than (`>`, `>=`)
//! - `distance()` function for point indexes
//! - Range indexes only (fulltext indexes are handled separately)

use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::{
    graph::graph::Graph,
    index::indexer::{IndexQuery, IndexType},
    parser::ast::{ExprIR, QueryExpr, QueryNode, Variable},
    tree,
};

use super::super::IR;

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
        if let ExprIR::Property(attr) = node.data() {
            return Some(attr.clone());
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
) -> Option<(
    Arc<String>,
    NodeIdx<Dyn<ExprIR<Variable>>>,
    NodeIdx<Dyn<ExprIR<Variable>>>,
)> {
    let lhs_idx = filter.root().child(0).idx();
    let rhs_idx = filter.root().child(1).idx();

    match (
        extract_attribute_from_subtree(filter, lhs_idx),
        extract_attribute_from_subtree(filter, rhs_idx),
    ) {
        (Some(_), Some(_)) => None,
        (Some(attr), _) => Some((attr, lhs_idx, rhs_idx)),
        (_, Some(attr)) => Some((attr, rhs_idx, lhs_idx)),
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
            Arc::new(IndexQuery::Equal {
                key: attr.clone(),
                value: Arc::new(constant_node),
            }),
        )),
        ExprIR::Gt => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range {
                key: attr.clone(),
                min: Some(Arc::new(constant_node)),
                max: None,
                include_min: false,
                include_max: false,
            }),
        )),
        ExprIR::Ge => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range {
                key: attr.clone(),
                min: Some(Arc::new(constant_node)),
                max: None,
                include_min: true,
                include_max: false,
            }),
        )),
        ExprIR::Lt => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range {
                key: attr.clone(),
                min: None,
                max: Some(Arc::new(constant_node)),
                include_min: false,
                include_max: false,
            }),
        )),
        ExprIR::Le => Some((
            node.clone(),
            node.labels[0].clone(),
            Arc::new(IndexQuery::Range {
                key: attr.clone(),
                min: None,
                max: Some(Arc::new(constant_node)),
                include_min: false,
                include_max: true,
            }),
        )),
        _ => None,
    }
}

/// Builds an index scan for a distance filter (e.g. `distance(n.loc, point(...)) < 100`).
fn try_distance_index_scan(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    attr: &Arc<String>,
    filter: &DynTree<ExprIR<Variable>>,
    attribute_side: NodeIdx<Dyn<ExprIR<Variable>>>,
    constant_node: DynTree<ExprIR<Variable>>,
) -> IndexScanResult {
    let operand = filter.root().data();
    // If attribute side is 0 than operand must be <
    // else if attribute_side == 1 than operand must be >
    // Fail in any other case
    match operand {
        ExprIR::Lt => {
            if filter.root().child(0).idx() != attribute_side {
                return None;
            }
        }
        ExprIR::Gt => {
            if filter.root().child(1).idx() != attribute_side {
                return None;
            }
        }
        _ => return None,
    }
    let child_0_idx = filter.node(attribute_side).child(0).idx();
    let child_1_idx = filter.node(attribute_side).child(1).idx();
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

/// Merges two index queries on the same attribute into a single Range query.
///
/// For example, `year >= 1980` and `year < 1990` become `Range { min: 1980, max: 1990, include_min: true, include_max: false }`.
///
/// When both queries specify the same bound (both min or both max), we cannot
/// determine at plan time which is stricter (the values are expression trees),
/// so we fall back to `And` and let the index engine intersect them.
fn merge_range_queries(
    a: IndexQuery<QueryExpr<Variable>>,
    b: IndexQuery<QueryExpr<Variable>>,
) -> IndexQuery<QueryExpr<Variable>> {
    match (a, b) {
        (
            IndexQuery::Range {
                key,
                min: min_a,
                max: max_a,
                include_min: inc_min_a,
                include_max: inc_max_a,
            },
            IndexQuery::Range {
                min: min_b,
                max: max_b,
                include_min: inc_min_b,
                include_max: inc_max_b,
                ..
            },
        ) => {
            // If both specify the same bound, we can't compare expression
            // values at plan time to pick the stricter one — fall back to And.
            if min_a.is_some() && min_b.is_some() || max_a.is_some() && max_b.is_some() {
                return IndexQuery::And(vec![
                    IndexQuery::Range {
                        key: key.clone(),
                        min: min_a,
                        max: max_a,
                        include_min: inc_min_a,
                        include_max: inc_max_a,
                    },
                    IndexQuery::Range {
                        key,
                        min: min_b,
                        max: max_b,
                        include_min: inc_min_b,
                        include_max: inc_max_b,
                    },
                ]);
            }
            // Complementary bounds: one provides min, the other max.
            // The unused include flag is always false, so or/select works.
            let (min, include_min) = if min_a.is_some() {
                (min_a, inc_min_a)
            } else {
                (min_b, inc_min_b)
            };
            let (max, include_max) = if max_a.is_some() {
                (max_a, inc_max_a)
            } else {
                (max_b, inc_max_b)
            };
            IndexQuery::Range {
                key,
                min,
                max,
                include_min,
                include_max,
            }
        }
        (a, b) => IndexQuery::And(vec![a, b]),
    }
}

/// Tries to convert a single comparison filter into an index scan for the given node.
fn try_single_filter_index_scan(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filter: &DynTree<ExprIR<Variable>>,
    graph: &Graph,
) -> IndexScanResult {
    if !matches!(
        filter.root().data(),
        ExprIR::Eq | ExprIR::Gt | ExprIR::Ge | ExprIR::Lt | ExprIR::Le
    ) {
        return None;
    }
    let (attr, attr_side, constant_side) = extract_attribute_and_expression_from_filter(filter)?;
    if !graph.is_indexed(&node.labels[0], &attr, &IndexType::Range) {
        return None;
    }
    match filter.node(attr_side).data() {
        ExprIR::FuncInvocation(func) => {
            let constant_node = filter.node(constant_side).clone_as_tree();
            match func.name.as_str() {
                "distance" => {
                    try_distance_index_scan(node, &attr, filter, attr_side, constant_node)
                }
                _ => None,
            }
        }
        ExprIR::Property(attr) => {
            let constant_node = filter.node(constant_side).clone_as_tree();
            // If the property is on the right side (e.g., `1980 <= m.year`),
            // flip the operator so the index scan uses the correct direction.
            let op = if attr_side == filter.root().child(0).idx() {
                filter.root().data().clone()
            } else {
                match filter.root().data() {
                    ExprIR::Eq => ExprIR::Eq,
                    ExprIR::Gt => ExprIR::Lt,
                    ExprIR::Ge => ExprIR::Le,
                    ExprIR::Lt => ExprIR::Gt,
                    ExprIR::Le => ExprIR::Ge,
                    _ => unreachable!(),
                }
            };
            try_property_index_scan(node, attr, &op, constant_node)
        }
        _ => None,
    }
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
                && graph.is_indexed(label, attr_str, &IndexType::Range)
            {
                return Some((
                    node.clone(),
                    attr_str.clone(),
                    tree!(
                        ExprIR::Eq,
                        tree!(
                            ExprIR::Property(attr_str.clone()),
                            tree!(ExprIR::Variable(node.alias.clone()))
                        ),
                        attr.child(0).as_cloned_subtree()
                    ),
                ));
            }
        }
    }
    None
}

/// Attempts to replace label scans with index scans where applicable.
///
/// Scans the plan for patterns like:
/// `NodeByLabelScan` → `Filter(property = value)`
///
/// If an index exists on the filtered property, replaces with `NodeByIndexScan`.
/// Also handles AND filters where multiple conjuncts can each be converted to index queries.
pub(super) fn utilize_index(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

        for idx in indices {
            let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(idx).data()
                && !node.labels.is_empty()
                && let IR::Filter(filter) = optimized_plan.node(idx).parent().unwrap().data()
            {
                if matches!(filter.root().data(), ExprIR::And) {
                    // AND filter: try to merge conjuncts into a single range index query
                    let mut merged: Option<IndexQuery<QueryExpr<Variable>>> = None;
                    let mut remaining_conjuncts = Vec::new();
                    for child in filter.root().children() {
                        let conjunct = child.clone_as_tree();
                        if let Some((_, _, query)) =
                            try_single_filter_index_scan(node, &conjunct, graph)
                        {
                            let query = Arc::try_unwrap(query).unwrap();
                            merged = Some(match merged {
                                None => query,
                                Some(prev) => merge_range_queries(prev, query),
                            });
                        } else {
                            remaining_conjuncts.push(conjunct);
                        }
                    }
                    merged.map(|combined_query| {
                        (
                            node.clone(),
                            node.labels[0].clone(),
                            Arc::new(combined_query),
                            remaining_conjuncts,
                        )
                    })
                } else {
                    // Single comparison filter
                    try_single_filter_index_scan(node, filter, graph)
                        .map(|(n, l, q)| (n, l, q, Vec::new()))
                }
            } else {
                None
            };
            if let Some((node, index, query, remaining)) = node {
                let mut op = optimized_plan.node_mut(idx);
                *op.data_mut() = IR::NodeByIndexScan { node, index, query };
                if remaining.is_empty() {
                    op.parent_mut().unwrap().take_out();
                } else {
                    // Replace the AND filter with only the remaining conjuncts
                    let remaining_filter = if remaining.len() == 1 {
                        Arc::new(remaining.into_iter().next().unwrap())
                    } else {
                        Arc::new(tree!(ExprIR::And; remaining))
                    };
                    *op.parent_mut().unwrap().data_mut() = IR::Filter(remaining_filter);
                }
                changed = true;
                break; // Restart traversal after structural modification
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
                    query: Arc::new(IndexQuery::Equal {
                        key: attr.clone(),
                        value: Arc::new(filter.root().child(1).clone_as_tree()),
                    }),
                };
            }
        }

        if !changed {
            break;
        }
    }
}
