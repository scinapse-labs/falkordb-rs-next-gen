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
//! ## Node By Label And ID Optimization
//!
//! Replaces label scan + ID filter with direct ID lookup:
//! ```text
//! Before: NodeByLabelScan(:Person) → Filter(id(n) = 42)
//! After:  NodeByLabelAndIdScan(:Person, 42)
//! ```
//!
//! ## Good Practice
//!
//! The optimizer uses a collect-then-iterate pattern when modifying the tree
//! to avoid issues with mutable iteration. This is a common pattern when
//! working with tree structures that need in-place modification.

use std::collections::HashSet;
use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynNode, DynTree, NodeIdx, NodeRef};

use crate::{
    graph::graph::Graph,
    index::indexer::{IndexQuery, IndexType},
    parser::ast::{ExprIR, QueryExpr, QueryNode, QueryRelationship, Variable},
    runtime::{eval::ExprEval, runtime::GetVariables, value::Value},
    tree,
};

use super::{IR, inline_node_attrs_to_filter, subtree_contains};

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

/// Attempts to replace label scans with index scans where applicable.
///
/// Scans the plan for patterns like:
/// `NodeByLabelScan` → `Filter(property = value)`
///
/// If an index exists on the filtered property, replaces with `NodeByIndexScan`.
/// Also handles AND filters where multiple conjuncts can each be converted to index queries.
fn utilize_index(
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

fn get_id_filter(
    filter: &DynNode<ExprIR<Variable>>,
    node_alias: &Variable,
) -> Option<(QueryExpr<Variable>, ExprIR<Variable>)> {
    if matches!(
        filter.data(),
        ExprIR::Eq | ExprIR::Gt | ExprIR::Ge | ExprIR::Lt | ExprIR::Le
    ) && let ExprIR::FuncInvocation(inner_func) = filter.child(0).data()
        && inner_func.name == "id"
        && let ExprIR::Variable(var) = filter.child(0).child(0).data()
        && var == node_alias
    {
        Some((
            Arc::new(filter.child(1).clone_as_tree()),
            filter.data().clone(),
        ))
    } else if matches!(
        filter.data(),
        ExprIR::Eq | ExprIR::Gt | ExprIR::Ge | ExprIR::Lt | ExprIR::Le
    ) && let ExprIR::FuncInvocation(inner_func) = filter.child(1).data()
        && inner_func.name == "id"
        && let ExprIR::Variable(var) = filter.child(1).child(0).data()
        && var == node_alias
    {
        let op = match filter.data() {
            ExprIR::Eq => ExprIR::Eq,
            ExprIR::Gt => ExprIR::Lt,
            ExprIR::Ge => ExprIR::Le,
            ExprIR::Lt => ExprIR::Gt,
            ExprIR::Le => ExprIR::Ge,
            _ => unreachable!(),
        };
        Some((Arc::new(filter.child(0).clone_as_tree()), op))
    } else {
        None
    }
}

/// Collects all variable IDs referenced in an expression tree.
fn collect_expr_variables(expr: &DynTree<ExprIR<Variable>>) -> HashSet<u32> {
    let mut vars = HashSet::new();
    for idx in expr.root().indices::<Bfs>() {
        if let ExprIR::Variable(var) = expr.node(idx).data() {
            vars.insert(var.id);
        }
    }
    vars
}

/// Collects all variable IDs provided by a plan subtree.
fn collect_subtree_variables(node: &DynNode<IR>) -> HashSet<u32> {
    let mut vars = HashSet::new();
    for var in node.get_variables() {
        vars.insert(var.id);
    }
    vars
}

/// Pushes filter conjuncts down through nodes.
///
/// Transforms:
/// ```text
/// Filter(AND(cond_a, cond_b))
///   └─ CartesianProduct
///        ├─ ChildA
///        └─ ChildB
/// ```
/// Into:
/// ```text
/// CartesianProduct
///   ├─ Filter(cond_a)
///   │    └─ ChildA
///   └─ Filter(cond_b)
///        └─ ChildB
/// ```
///
/// Each conjunct is routed to the child whose variables fully cover the
/// conjunct's referenced variables. Conjuncts that span multiple children
/// remain at the current level.
fn push_filters_down(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();
        for idx in indices {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };

            // Merge stacked filters: if this filter's child is also a filter,
            // combine their conjuncts into a single AND filter.
            if let Some(child) = optimized_plan.node(idx).get_child(0)
                && let IR::Filter(child_filter) = child.data()
            {
                let child_filter = child_filter.clone();
                let filter = filter.clone();
                let child_idx = child.idx();

                // Flatten conjuncts from both filters
                let mut conjuncts: Vec<DynTree<ExprIR<Variable>>> = vec![];
                for f in [&filter, &child_filter] {
                    if matches!(f.root().data(), ExprIR::And) {
                        conjuncts.extend(f.root().children().map(|c| c.clone_as_tree()));
                    } else {
                        conjuncts.push((**f).clone());
                    }
                }

                let merged = if conjuncts.len() == 1 {
                    Arc::new(conjuncts.into_iter().next().unwrap())
                } else {
                    Arc::new(tree!(ExprIR::And; conjuncts))
                };
                *optimized_plan.node_mut(idx).data_mut() = IR::Filter(merged);
                optimized_plan.node_mut(child_idx).take_out();

                changed = true;
                break;
            }

            if !optimized_plan
                .node(idx)
                .children()
                .any(|c| c.num_children() > 0)
            {
                continue; // Skip if filter already is downstream
            }
            let filter = filter.clone();

            // Split filter into individual conjuncts
            let conjuncts: Vec<DynTree<ExprIR<Variable>>> =
                if matches!(filter.root().data(), ExprIR::And) {
                    filter
                        .root()
                        .children()
                        .map(|c| c.clone_as_tree())
                        .collect()
                } else {
                    vec![(*filter).clone()]
                };

            // Collect children and the variables they provide
            let mut children: Vec<_> = optimized_plan
                .node(idx)
                .children()
                .filter(|c| {
                    c.num_children() > 0
                        && !matches!(
                            c.data(),
                            IR::Project { .. }
                                | IR::Aggregate { .. }
                                | IR::Merge { .. }
                                | IR::Argument
                                | IR::SemiApply
                                | IR::AntiSemiApply
                                | IR::OrApplyMultiplexer(_)
                        )
                })
                .flat_map(|c| c.children().collect::<Vec<_>>())
                .map(|c| (c.idx(), collect_subtree_variables(&c)))
                .collect();

            // Compute inherited variables from Apply context.
            // When Apply propagates bound variables via Argument leaves,
            // the right branch effectively has access to the left branch's
            // variables. We augment variable sets accordingly so filters
            // referencing bound variables can be pushed down.
            let mut inherited = HashSet::new();

            // Case 1: Filter's child is Apply — left branch vars are
            // available in the right branch via Argument.
            if let Some(child) = optimized_plan.node(idx).get_child(0)
                && matches!(child.data(), IR::Apply)
                && let Some((_, left_vars)) = children.first()
            {
                inherited.extend(left_vars.iter());
            }

            // Case 2: Filter is inside an Apply's right branch.
            // The left branch's variables are available in the right branch
            // via Argument propagation. Only applies when the filter is
            // actually in the right subtree (not the left subtree itself).
            // Stop the ancestor walk at Merge nodes — Merge's internal
            // match sub-plan has its own Argument leaves that receive
            // variables from Merge's input, not from an enclosing Apply.
            {
                let mut ancestor = idx;
                while let Some(parent) = optimized_plan.node(ancestor).parent() {
                    if matches!(parent.data(), IR::Apply) {
                        // Only inherit left-branch variables if the filter
                        // is NOT inside the left branch (child 0) itself.
                        let left_child_idx = parent.child(0).idx();
                        let filter_in_left = {
                            let mut cur = idx;
                            loop {
                                if cur == left_child_idx {
                                    break true;
                                }
                                if let Some(p) = optimized_plan.node(cur).parent() {
                                    if p.idx() == parent.idx() {
                                        break false;
                                    }
                                    cur = p.idx();
                                } else {
                                    break false;
                                }
                            }
                        };
                        if !filter_in_left {
                            let left_vars = collect_subtree_variables(&parent.child(0));
                            inherited.extend(left_vars);
                        }
                        break;
                    }
                    if matches!(parent.data(), IR::Merge { .. }) {
                        break;
                    }
                    ancestor = parent.idx();
                }
            }

            // Augment variable sets for subtrees containing Argument leaves.
            if !inherited.is_empty() {
                for (child_idx, vars) in &mut children {
                    if subtree_contains(optimized_plan, *child_idx, |ir| matches!(ir, IR::Argument))
                    {
                        vars.extend(&inherited);
                    }
                }
            }

            // Route each conjunct to the child that provides all its variables
            let mut child_conjuncts: Vec<Vec<DynTree<ExprIR<Variable>>>> =
                vec![vec![]; children.len()];
            let mut remaining: Vec<DynTree<ExprIR<Variable>>> = vec![];

            for conjunct in conjuncts {
                let conj_vars = collect_expr_variables(&conjunct);
                let matched_child = children
                    .iter()
                    .enumerate()
                    .find(|(_, (_, child_vars))| conj_vars.iter().all(|v| child_vars.contains(v)))
                    .map(|(i, _)| i);
                if let Some(i) = matched_child {
                    child_conjuncts[i].push(conjunct);
                } else {
                    remaining.push(conjunct);
                }
            }

            // Skip if nothing can be pushed down
            if child_conjuncts.iter().all(Vec::is_empty) {
                continue;
            }

            // For each child with matching conjuncts: add a Filter-wrapped
            // clone as a sibling, then prune the original.
            for (i, conjuncts) in child_conjuncts.into_iter().enumerate() {
                if conjuncts.is_empty() {
                    continue;
                }

                let child_idx = children[i].0;

                // Build the filter expression for this child
                let filter_expr = if conjuncts.len() == 1 {
                    Arc::new(conjuncts.into_iter().next().unwrap())
                } else {
                    Arc::new(tree!(ExprIR::And; conjuncts))
                };

                // Insert the new filter node above the child
                optimized_plan
                    .node_mut(child_idx)
                    .push_parent(IR::Filter(filter_expr));
            }

            // Update or remove the original Filter
            if remaining.is_empty() {
                optimized_plan.node_mut(idx).take_out();
            } else if remaining.len() == 1 {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(remaining.into_iter().next().unwrap()));
            } else {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(tree!(ExprIR::And; remaining)));
            }

            changed = true;
            break; // Restart traversal after structural modification
        }

        if !changed {
            break;
        }
    }
}

/// Replaces label scan + ID filter with direct node ID lookup.
fn utilize_node_by_id(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

        for idx in indices {
            let mut filters = vec![];
            let node = match optimized_plan.node(idx).data() {
                IR::NodeByLabelScan(node) | IR::AllNodeScan(node) => node.clone(),
                _ => continue,
            };
            if let IR::Filter(filter) = optimized_plan.node(idx).parent().unwrap().data() {
                if let Some((id, op)) = get_id_filter(&filter.root(), &node.alias) {
                    filters.push((id, op));
                } else if matches!(filter.root().data(), ExprIR::And) {
                    for child in filter.root().children() {
                        if let Some((id, op)) = get_id_filter(&child, &node.alias) {
                            filters.push((id, op));
                        } else {
                            filters.clear();
                            break;
                        }
                    }
                }
            }
            if !filters.is_empty() {
                let mut new_op = optimized_plan.node_mut(idx);
                if node.labels.is_empty() {
                    *new_op.data_mut() = IR::NodeByIdSeek {
                        node: node.clone(),
                        filter: filters,
                    };
                } else {
                    *new_op.data_mut() = IR::NodeByLabelAndIdScan {
                        node: node.clone(),
                        filter: filters,
                    };
                }
                new_op.parent_mut().unwrap().take_out();
                changed = true;
                break; // Restart traversal after structural modification
            }
        }

        if !changed {
            break;
        }
    }
}

/// Returns true if the filter expression evaluates to a constant `true`.
fn is_constant_true(filter: &DynTree<ExprIR<Variable>>) -> bool {
    matches!(
        ExprEval::constant().eval(filter, filter.root().idx(), None, None),
        Ok(Value::Bool(true))
    )
}

/// Eliminates filter nodes whose expression evaluates to constant `true`.
///
/// Also removes `Bool(true)` conjuncts from AND filters, and removes the
/// entire filter if all conjuncts are true.
fn eliminate_true_filters(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

        for idx in indices {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };

            if is_constant_true(filter) {
                optimized_plan.node_mut(idx).take_out();
                changed = true;
                break;
            }

            // For AND filters, remove constant-true conjuncts.
            if matches!(filter.root().data(), ExprIR::And) {
                let remaining: Vec<DynTree<ExprIR<Variable>>> = filter
                    .root()
                    .children()
                    .filter(|c| !is_constant_true(&c.clone_as_tree()))
                    .map(|c| c.clone_as_tree())
                    .collect();

                if remaining.len() < filter.root().num_children() {
                    if remaining.is_empty() {
                        optimized_plan.node_mut(idx).take_out();
                    } else if remaining.len() == 1 {
                        *optimized_plan.node_mut(idx).data_mut() =
                            IR::Filter(Arc::new(remaining.into_iter().next().unwrap()));
                    } else {
                        *optimized_plan.node_mut(idx).data_mut() =
                            IR::Filter(Arc::new(tree!(ExprIR::And; remaining)));
                    }
                    changed = true;
                    break;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Absorbs edge filters into CondVarLenTraverse operators.
///
/// Detects patterns where a Filter sits directly above a CondVarLenTraverse
/// and the filter expression references only the VLT edge alias variable.
/// In that case, the filter is absorbed into the VLT's `edge_filter` field
/// so it can be evaluated per-hop during traversal.
///
/// For AND filters where some conjuncts reference the edge and others don't,
/// only the edge-referencing conjuncts are absorbed.
fn absorb_edge_filters_into_vlt(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

        for idx in indices {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };

            // Find a CondVarLenTraverse child (possibly nested under other nodes)
            let vlt_idx = find_descendant_vlt(optimized_plan, idx);
            let Some(vlt_idx) = vlt_idx else {
                continue;
            };

            let edge_alias_id = match optimized_plan.node(vlt_idx).data() {
                IR::CondVarLenTraverse { relationship, .. } => relationship.alias.id,
                _ => continue,
            };

            let filter = filter.clone();

            // Split AND conjuncts: those referencing only the edge alias go into VLT
            let conjuncts: Vec<DynTree<ExprIR<Variable>>> =
                if matches!(filter.root().data(), ExprIR::And) {
                    filter
                        .root()
                        .children()
                        .map(|c| c.clone_as_tree())
                        .collect()
                } else {
                    vec![(*filter).clone()]
                };

            let mut edge_conjuncts = Vec::new();
            let mut remaining = Vec::new();

            for conjunct in conjuncts {
                let vars = collect_expr_variables(&conjunct);
                if !vars.is_empty() && vars.iter().all(|v| *v == edge_alias_id) {
                    edge_conjuncts.push(conjunct);
                } else {
                    remaining.push(conjunct);
                }
            }

            if edge_conjuncts.is_empty() {
                continue;
            }

            // Build the edge filter expression
            let edge_filter_expr: Arc<DynTree<ExprIR<Variable>>> = if edge_conjuncts.len() == 1 {
                Arc::new(edge_conjuncts.into_iter().next().unwrap())
            } else {
                Arc::new(tree!(ExprIR::And; edge_conjuncts))
            };

            // Merge with existing edge_filter if present
            match optimized_plan.node_mut(vlt_idx).data_mut() {
                IR::CondVarLenTraverse { edge_filter, .. } => {
                    if let Some(existing) = edge_filter.take() {
                        *edge_filter = Some(Arc::new(
                            tree!(ExprIR::And; [(*existing).clone(), (*edge_filter_expr).clone()]),
                        ));
                    } else {
                        *edge_filter = Some(edge_filter_expr);
                    }
                }
                _ => unreachable!(),
            }

            // Update or remove the original Filter
            if remaining.is_empty() {
                optimized_plan.node_mut(idx).take_out();
            } else if remaining.len() == 1 {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(remaining.into_iter().next().unwrap()));
            } else {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(tree!(ExprIR::And; remaining)));
            }

            changed = true;
            break;
        }

        if !changed {
            break;
        }
    }
}

/// Find a CondVarLenTraverse node that is a direct child of the given filter,
/// or is the only child chain below the filter (e.g., Filter → VLT or Filter → PathBuilder → VLT).
fn find_descendant_vlt(
    plan: &DynTree<IR>,
    filter_idx: NodeIdx<Dyn<IR>>,
) -> Option<NodeIdx<Dyn<IR>>> {
    let mut current = filter_idx;
    loop {
        let node = plan.node(current);
        if node.num_children() != 1 {
            return None;
        }
        let child = node.child(0);
        if matches!(child.data(), IR::CondVarLenTraverse { .. }) {
            return Some(child.idx());
        }
        // Walk through transparent operators like PathBuilder, Filter
        if matches!(child.data(), IR::PathBuilder(_) | IR::Filter(_)) {
            current = child.idx();
            continue;
        }
        return None;
    }
}

/// Replaces `Filter(a = b) -> CartesianProduct(ChildA, ChildB)` with
/// `ValueHashJoin(ChildA, ChildB)` when the equality predicate's left side
/// references only variables from one child and the right side references
/// only variables from the other.
///
/// Uses a recursive tree-rebuild approach to cleanly restructure multi-child
/// CartesianProduct nodes.
fn replace_cartesian_with_hash_join(optimized_plan: &mut DynTree<IR>) {
    let new_plan = rebuild_with_hash_joins(&optimized_plan.root());
    *optimized_plan = new_plan;
}

/// Recursively rebuilds the plan tree, converting Filter -> CartesianProduct
/// patterns into ValueHashJoin where applicable.
fn rebuild_with_hash_joins(node: &DynNode<IR>) -> DynTree<IR> {
    if let IR::Filter(filter) = node.data()
        && let Some(child) = node.get_child(0)
        && matches!(child.data(), IR::CartesianProduct)
        && let Some(result) = try_hash_join_rewrite(filter, &child)
    {
        // Recursively apply to the result (may find more join opportunities).
        return rebuild_with_hash_joins(&result.root());
    }

    // Default: clone this node's data and recursively rebuild children.
    let mut new_tree = DynTree::new(node.data().clone());
    for child in node.children() {
        let child_tree = rebuild_with_hash_joins(&child);
        new_tree.root_mut().push_child_tree(child_tree);
    }
    new_tree
}

/// Attempts to find one equality conjunct in the filter that can be converted
/// to a ValueHashJoin. Returns the rebuilt subtree if successful.
fn try_hash_join_rewrite(
    filter: &QueryExpr<Variable>,
    cp_node: &DynNode<IR>,
) -> Option<DynTree<IR>> {
    let conjuncts: Vec<DynTree<ExprIR<Variable>>> = if matches!(filter.root().data(), ExprIR::And) {
        filter
            .root()
            .children()
            .map(|c| c.clone_as_tree())
            .collect()
    } else {
        vec![(**filter).clone()]
    };

    let cp_children_vars: Vec<HashSet<u32>> = cp_node
        .children()
        .map(|c| collect_subtree_variables(&c))
        .collect();

    if cp_children_vars.len() < 2 {
        return None;
    }

    let mut found = None;
    for (ci, conjunct) in conjuncts.iter().enumerate() {
        if !matches!(conjunct.root().data(), ExprIR::Eq) || conjunct.root().num_children() < 2 {
            continue;
        }
        let lhs_tree = conjunct.root().child(0).clone_as_tree();
        let rhs_tree = conjunct.root().child(1).clone_as_tree();
        let lhs_vars = collect_expr_variables(&lhs_tree);
        let rhs_vars = collect_expr_variables(&rhs_tree);
        if lhs_vars.is_empty() || rhs_vars.is_empty() {
            continue;
        }

        'outer: for (li, l_vars) in cp_children_vars.iter().enumerate() {
            for (ri, r_vars) in cp_children_vars.iter().enumerate() {
                if li == ri {
                    continue;
                }
                if lhs_vars.iter().all(|v| l_vars.contains(v))
                    && rhs_vars.iter().all(|v| r_vars.contains(v))
                {
                    found = Some((
                        ci,
                        li,
                        ri,
                        Arc::new(lhs_tree.clone()),
                        Arc::new(rhs_tree.clone()),
                    ));
                    break 'outer;
                }
                if rhs_vars.iter().all(|v| l_vars.contains(v))
                    && lhs_vars.iter().all(|v| r_vars.contains(v))
                {
                    found = Some((
                        ci,
                        li,
                        ri,
                        Arc::new(rhs_tree.clone()),
                        Arc::new(lhs_tree.clone()),
                    ));
                    break 'outer;
                }
            }
        }
        if found.is_some() {
            break;
        }
    }

    let (conj_idx, left_pos, right_pos, lhs_exp, rhs_exp) = found?;

    let cp_child_trees: Vec<DynTree<IR>> = cp_node.children().map(|c| c.clone_as_tree()).collect();

    let vhj = tree!(
        IR::ValueHashJoin { lhs_exp, rhs_exp },
        cp_child_trees[left_pos].clone(),
        cp_child_trees[right_pos].clone()
    );

    let other_children: Vec<DynTree<IR>> = cp_child_trees
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i != left_pos && *i != right_pos)
        .map(|(_, t)| t)
        .collect();

    let join_subtree = if other_children.is_empty() {
        vhj
    } else {
        let mut children = other_children;
        children.push(vhj);
        tree!(IR::CartesianProduct; children)
    };

    let remaining: Vec<DynTree<ExprIR<Variable>>> = conjuncts
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i != conj_idx)
        .map(|(_, c)| c)
        .collect();

    if remaining.is_empty() {
        Some(join_subtree)
    } else {
        let remaining_filter = if remaining.len() == 1 {
            Arc::new(remaining.into_iter().next().unwrap())
        } else {
            Arc::new(tree!(ExprIR::And; remaining))
        };
        let mut result = DynTree::new(IR::Filter(remaining_filter));
        result.root_mut().push_child_tree(join_subtree);
        Some(result)
    }
}

/// Scores a candidate scan endpoint for the scan node selection optimizer.
///
/// Higher score = better starting point. Priority:
/// - Bound variable (provided by child operator): score 3
/// - Filtered variable (referenced by a Filter ancestor): score 2
/// - Labeled variable: score 1
/// - Neither: score 0
///
/// When scores are equal, the endpoint with fewer label nodes is preferred.
fn score_endpoint(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filtered_vars: &HashSet<u32>,
    bound_vars: &HashSet<u32>,
    graph: &Graph,
) -> (u32, u64) {
    let mut score = 0u32;
    if bound_vars.contains(&node.alias.id) {
        score += 3;
    }
    if filtered_vars.contains(&node.alias.id) {
        score += 2;
    }
    // Node attribute filters (e.g. {name: "Nicolas Cage"}) also count as filters.
    if node.attrs.root().num_children() > 0 {
        score += 2;
    }
    if !node.labels.is_empty() {
        score += 1;
    }
    // Cardinality: minimum label node count (lower is better).
    // For nodes with no labels, use u64::MAX so labeled nodes win ties.
    let cardinality = if node.labels.is_empty() {
        u64::MAX
    } else {
        node.labels
            .iter()
            .map(|l| graph.label_node_count(l))
            .min()
            .unwrap_or(u64::MAX)
    };
    (score, cardinality)
}

/// Collects variable IDs referenced by Filter nodes that are ancestors of
/// the given node index, up to the first non-Filter/non-CondTraverse ancestor.
fn collect_filtered_vars(
    plan: &DynTree<IR>,
    start_idx: NodeIdx<Dyn<IR>>,
) -> HashSet<u32> {
    let mut vars = HashSet::new();
    let mut current = start_idx;
    while let Some(parent) = plan.node(current).parent() {
        match parent.data() {
            IR::Filter(filter) => {
                for idx in filter.root().indices::<Bfs>() {
                    if let ExprIR::Variable(v) = filter.node(idx).data() {
                        vars.insert(v.id);
                    }
                }
            }
            // Walk through transparent operators to find filters higher up
            IR::CondTraverse { .. } | IR::CondVarLenTraverse { .. } | IR::PathBuilder(_) => {}
            _ => break,
        }
        current = parent.idx();
    }
    // Also check the node at current if it has a parent that is a filter
    // (the loop above moves through parents)
    vars
}

/// Creates a scan subtree for the given node, with an optional inline attr filter.
/// Returns a `DynTree<IR>` containing `[Filter →] NodeByLabelScan|AllNodeScan`.
fn make_scan_subtree(node: &Arc<QueryNode<Arc<String>, Variable>>) -> DynTree<IR> {
    let (clean_node, attr_filter) = inline_node_attrs_to_filter(node);
    let mut scan = if clean_node.labels.is_empty() {
        DynTree::new(IR::AllNodeScan(clean_node))
    } else {
        DynTree::new(IR::NodeByLabelScan(clean_node))
    };
    if let Some(filter_expr) = attr_filter {
        scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
    }
    scan
}

/// Creates a new `QueryRelationship` with from and to swapped.
fn swap_relationship(
    rel: &Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
    new_from: Arc<QueryNode<Arc<String>, Variable>>,
    new_to: Arc<QueryNode<Arc<String>, Variable>>,
) -> Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>> {
    let mut swapped = QueryRelationship::new(
        rel.alias.clone(),
        rel.types.clone(),
        rel.attrs.clone(),
        new_from,
        new_to,
        rel.bidirectional,
        rel.min_hops,
        rel.max_hops,
    );
    swapped.all_shortest_paths = rel.all_shortest_paths;
    Arc::new(swapped)
}

/// Collects output variable alias IDs from an IR node.
/// Used to detect which variables a child operator provides.
fn collect_output_aliases(ir: &IR) -> HashSet<u32> {
    let mut aliases = HashSet::new();
    match ir {
        IR::AllNodeScan(n) | IR::NodeByLabelScan(n) => {
            aliases.insert(n.alias.id);
        }
        IR::NodeByIndexScan { node, .. } | IR::NodeByLabelAndIdScan { node, .. } => {
            aliases.insert(node.alias.id);
        }
        IR::Project { exprs, copies } => {
            for (var, _) in exprs {
                aliases.insert(var.id);
            }
            for (var, _) in copies {
                aliases.insert(var.id);
            }
        }
        IR::Aggregate { names, .. } => {
            for var in names {
                aliases.insert(var.id);
            }
        }
        IR::Unwind { var, .. } => {
            aliases.insert(var.id);
        }
        _ => {}
    }
    aliases
}

/// Selects the optimal scan node for leaf `CondTraverse` operators.
///
/// For each bottom-of-chain CondTraverse (leaf or with a non-CT child),
/// determines the best endpoint to scan from based on: (1) bound variables
/// from child, (2) filter presence, (3) label presence, (4) label cardinality.
/// Adds a `NodeByLabelScan` or `AllNodeScan` child for leaf chains, and
/// optionally swaps from/to with `transposed=true` if the better endpoint is
/// at the other end.
///
/// For chains of CondTraverse operators, walks up to find the best endpoint
/// across the entire chain. If the best endpoint is not at the bottom, reverses
/// the chain direction.
fn select_scan_node(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    // Collect all bottom-of-chain CondTraverse indices.
    // A "bottom CT" is a CT that either has no children (leaf) or whose
    // only child is not a CT (e.g., Project, AllNodeScan).
    let bottom_ct_indices: Vec<_> = {
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();
        indices
            .into_iter()
            .filter(|&idx| {
                let node = optimized_plan.node(idx);
                if !matches!(node.data(), IR::CondTraverse { .. }) {
                    return false;
                }
                if node.num_children() == 0 {
                    return true; // leaf CT
                }
                if node.num_children() != 1 {
                    return false;
                }
                // Walk through single-child Filter nodes to find the real child.
                let mut child = node.child(0);
                while matches!(child.data(), IR::Filter(_)) && child.num_children() == 1 {
                    child = child.child(0);
                }
                !matches!(child.data(), IR::CondTraverse { .. })
            })
            .collect()
    };

    for bottom_idx in bottom_ct_indices {
        let is_leaf = optimized_plan.node(bottom_idx).num_children() == 0;
        // Detect if the child is a planner-added scan (not an outer-context op).
        let has_planner_scan = !is_leaf && {
            let child_data = optimized_plan.node(bottom_idx).child(0).data().clone();
            matches!(
                child_data,
                IR::AllNodeScan(_) | IR::NodeByLabelScan(_) | IR::Filter(_)
            )
        };
        // Treat CTs with planner-added scans like leaf CTs for scan selection.
        let effectively_leaf = is_leaf || has_planner_scan;

        // Walk up the chain of CondTraverse nodes to collect all endpoints.
        // The walk skips single-child Filter nodes between CTs (these are
        // inline attribute filters on destination nodes).
        let mut chain: Vec<NodeIdx<Dyn<IR>>> = vec![bottom_idx];
        {
            let mut current = bottom_idx;
            while let Some(parent) = optimized_plan.node(current).parent() {
                if matches!(parent.data(), IR::CondTraverse { .. }) {
                    chain.push(parent.idx());
                    current = parent.idx();
                } else if matches!(parent.data(), IR::Filter(_)) && parent.num_children() == 1 {
                    // Skip single-child Filter between CTs, but check if
                    // its parent is a CT to continue the chain.
                    let filter_idx = parent.idx();
                    if let Some(grandparent) = optimized_plan.node(filter_idx).parent()
                        && matches!(grandparent.data(), IR::CondTraverse { .. })
                    {
                        chain.push(grandparent.idx());
                        current = grandparent.idx();
                        continue;
                    }
                    break;
                } else {
                    break;
                }
            }
        }

        // Collect filtered variables from Filter ancestors above the chain.
        let top_of_chain = *chain.last().unwrap();
        let filtered_vars = collect_filtered_vars(optimized_plan, top_of_chain);

        // For non-leaf chains, detect bound variables from the child.
        // Only consider vars as "bound" if they come from an outer context
        // (Project, Aggregate, Argument, etc.), NOT from scan children
        // added by the planner — those just provide starting nodes.
        let bound_vars = if is_leaf {
            HashSet::new()
        } else {
            let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
            let child_data = optimized_plan.node(child_idx).data();
            match child_data {
                IR::AllNodeScan(_)
                | IR::NodeByLabelScan(_)
                | IR::NodeByIndexScan { .. }
                | IR::NodeByLabelAndIdScan { .. }
                | IR::Filter(_) => HashSet::new(),
                _ => collect_output_aliases(child_data),
            }
        };

        // Collect all candidate endpoints from the chain.
        // Each endpoint is (node, chain_position, is_from).
        // chain_position 0 = bottom, higher = closer to root.
        let mut candidates: Vec<(Arc<QueryNode<Arc<String>, Variable>>, usize, bool)> = vec![];

        // Track which alias IDs we've already seen to avoid duplicates
        // (a node that is the `to` of one CT and `from` of the next).
        let mut seen_aliases = HashSet::new();

        for (pos, &ct_idx) in chain.iter().enumerate() {
            if let IR::CondTraverse { relationship, .. } = optimized_plan.node(ct_idx).data() {
                if seen_aliases.insert(relationship.from.alias.id) {
                    candidates.push((relationship.from.clone(), pos, true));
                }
                if seen_aliases.insert(relationship.to.alias.id) {
                    candidates.push((relationship.to.clone(), pos, false));
                }
            }
        }

        // Score each candidate and find the best.
        let best = candidates.iter().max_by(|a, b| {
            let (score_a, card_a) = score_endpoint(&a.0, &filtered_vars, &bound_vars, graph);
            let (score_b, card_b) = score_endpoint(&b.0, &filtered_vars, &bound_vars, graph);
            score_a
                .cmp(&score_b)
                .then_with(|| card_b.cmp(&card_a)) // lower cardinality = better
                // Prefer leaf position (0) and `from` side to preserve
                // the original traversal direction when all else is equal.
                .then_with(|| b.1.cmp(&a.1)) // lower chain pos = better
                .then_with(|| {
                    // Prefer is_from=true (original `from` node)
                    a.2.cmp(&b.2)
                })
        });

        let Some((best_node, best_pos, best_is_from)) = best.cloned() else {
            continue;
        };

        // Determine if we need to reverse the chain.
        // The best endpoint should become the `from` of the bottom CT.
        let need_swap = if best_pos == 0 {
            // Best is at the bottom CT. Swap only if it's the `to`.
            !best_is_from
        } else {
            // Best is at a parent CT. Need to reverse the chain.
            true
        };

        if need_swap && chain.len() == 1 {
            // Simple case: single CondTraverse, swap from/to.
            let ct_idx = chain[0];
            if let IR::CondTraverse {
                relationship,
                emit_relationship,
                sibling_edges,
                ..
            } = optimized_plan.node(ct_idx).data()
            {
                let new_from = relationship.to.clone();
                let new_to = relationship.from.clone();
                let new_rel = swap_relationship(relationship, new_from, new_to);
                let emit = *emit_relationship;
                let edges = sibling_edges.clone();
                let scan_node = relationship.to.clone();

                // Check if child is a planner-added scan before mutating
                let child_is_planner_scan = if is_leaf {
                    false
                } else {
                    matches!(
                        optimized_plan.node(ct_idx).child(0).data(),
                        IR::AllNodeScan(_) | IR::NodeByLabelScan(_) | IR::Filter(_)
                    )
                };

                // Remove old scan child if it was a planner-added scan
                if child_is_planner_scan {
                    let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                    optimized_plan.node_mut(child_idx).prune();
                }

                // Build scan subtree before taking mutable borrow
                let scan_subtree = make_scan_subtree(&scan_node);

                let mut op = optimized_plan.node_mut(ct_idx);
                *op.data_mut() = IR::CondTraverse {
                    relationship: new_rel,
                    emit_relationship: emit,
                    sibling_edges: edges,
                    transposed: true,
                };

                if is_leaf || child_is_planner_scan {
                    // Add scan subtree (with optional attr filter) as child.
                    op.push_child_tree(scan_subtree);
                }
                // else: child is from outer context, keep it.
            }
        } else if need_swap && chain.len() > 1 {
            // Chain reversal: reverse the order of CondTraverse nodes and
            // swap from/to on each.

            // Collect relationship data from each CT in the chain (bottom to root).
            let mut rels: Vec<(
                Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
                bool,
                Vec<u32>,
            )> = Vec::new();
            // Also collect Filter nodes between CTs (keyed by destination alias).
            // These are inline attribute filters on destination nodes.
            let mut inter_ct_filters: Vec<DynTree<IR>> = Vec::new();
            for (i, &ct_idx) in chain.iter().enumerate() {
                if let IR::CondTraverse {
                    relationship,
                    emit_relationship,
                    sibling_edges,
                    ..
                } = optimized_plan.node(ct_idx).data()
                {
                    rels.push((
                        relationship.clone(),
                        *emit_relationship,
                        sibling_edges.clone(),
                    ));
                }
                // Collect Filter nodes between this CT and the next CT in chain.
                if i < chain.len() - 1 {
                    let next_ct_idx = chain[i + 1];
                    // Walk from next_ct -> ... -> current_ct, collect Filters.
                    let mut walk = optimized_plan.node(next_ct_idx).child(0).idx();
                    while walk != ct_idx {
                        let walk_data = optimized_plan.node(walk).data();
                        if matches!(walk_data, IR::Filter(_)) {
                            // Clone just the Filter node (without its children)
                            let filter_expr = match walk_data {
                                IR::Filter(expr) => expr.clone(),
                                _ => unreachable!(),
                            };
                            inter_ct_filters.push(tree!(IR::Filter(filter_expr)));
                        }
                        if optimized_plan.node(walk).num_children() > 0 {
                            walk = optimized_plan.node(walk).child(0).idx();
                        } else {
                            break;
                        }
                    }
                }
            }

            // Detach existing child of the bottom CT (if non-leaf) for reattachment,
            // but only if it's NOT a planner-added scan (those get replaced).
            let existing_child = if is_leaf {
                None
            } else {
                let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
                let child_is_planner_scan = matches!(
                    optimized_plan.node(child_idx).data(),
                    IR::AllNodeScan(_) | IR::NodeByLabelScan(_) | IR::Filter(_)
                );
                if child_is_planner_scan {
                    None // Will create a new scan for best_node instead
                } else {
                    Some(optimized_plan.node_mut(child_idx).clone_as_tree())
                }
            };

            // Reverse the chain and swap from/to on each relationship.
            rels.reverse();
            let mut new_rels: Vec<(
                Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
                bool,
                Vec<u32>,
                bool, // transposed
            )> = Vec::new();

            for (rel, emit, edges) in &rels {
                let new_from = rel.to.clone();
                let new_to = rel.from.clone();
                let new_rel = swap_relationship(rel, new_from, new_to);
                new_rels.push((new_rel, *emit, edges.clone(), true));
            }

            // Build the new subtree bottom-up.
            let mut subtree = existing_child.unwrap_or_else(|| make_scan_subtree(&best_node));
            for (rel, emit, edges, transposed) in new_rels.into_iter().rev() {
                subtree = tree!(
                    IR::CondTraverse {
                        relationship: rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed
                    },
                    subtree
                );
            }
            // Re-insert collected inter-CT filters above the bottom of the
            // new chain.  They filtered on intermediate dest nodes which are
            // still produced by the reversed CTs.
            for filter_tree in inter_ct_filters {
                let filter_data = filter_tree.root().data().clone();
                subtree = tree!(filter_data, subtree);
            }

            // Replace the chain in the plan.
            let top_idx = *chain.last().unwrap();

            // Detach all children of the top CT (the old chain below it).
            while optimized_plan.node(top_idx).num_children() > 0 {
                let child_idx = optimized_plan.node(top_idx).child(0).idx();
                optimized_plan.node_mut(child_idx).prune();
            }

            // Replace the top CT with the root of the new subtree.
            let new_root = subtree.root();
            *optimized_plan.node_mut(top_idx).data_mut() = new_root.data().clone();

            // Add children of the new subtree root to the top CT node.
            for child in new_root.children() {
                let child_tree: DynTree<IR> = child.clone_as_tree();
                optimized_plan.node_mut(top_idx).push_child_tree(child_tree);
            }
        } else if effectively_leaf {
            // No swap needed. Add/replace a scan for the current `from` node.
            let ct_idx = chain[0];
            if let IR::CondTraverse { relationship, .. } = optimized_plan.node(ct_idx).data() {
                let scan_node = relationship.from.clone();
                let from_node = relationship.from.clone();
                let to_node = relationship.to.clone();
                let new_rel = swap_relationship(relationship, from_node, to_node);

                if let IR::CondTraverse {
                    emit_relationship,
                    sibling_edges,
                    transposed,
                    ..
                } = optimized_plan.node(ct_idx).data()
                {
                    let emit = *emit_relationship;
                    let edges = sibling_edges.clone();
                    let trans = *transposed;

                    // Remove old planner-added scan child if present
                    if has_planner_scan {
                        let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                        optimized_plan.node_mut(child_idx).prune();
                    }

                    // Build scan subtree with optional attr filter
                    let scan_subtree = make_scan_subtree(&scan_node);

                    let mut op = optimized_plan.node_mut(ct_idx);
                    *op.data_mut() = IR::CondTraverse {
                        relationship: new_rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed: trans,
                    };

                    op.push_child_tree(scan_subtree);
                }
            }
        }
        // else: no swap needed and non-leaf — nothing to do, child already attached.
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

    eliminate_true_filters(&mut optimized_plan);
    select_scan_node(&mut optimized_plan, graph);
    push_filters_down(&mut optimized_plan);
    replace_cartesian_with_hash_join(&mut optimized_plan);
    absorb_edge_filters_into_vlt(&mut optimized_plan);
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
