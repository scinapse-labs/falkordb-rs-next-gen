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

mod absorb_edge_filters_into_vlt;
mod eliminate_true_filters;
mod push_filters_down;
mod replace_cartesian_with_hash_join;
mod select_scan_node;
mod utilize_index;
mod utilize_node_by_id;

use std::collections::HashSet;

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, Variable},
};

use super::IR;

use absorb_edge_filters_into_vlt::absorb_edge_filters_into_vlt;
use eliminate_true_filters::eliminate_true_filters;
use push_filters_down::push_filters_down;
use replace_cartesian_with_hash_join::replace_cartesian_with_hash_join;
use select_scan_node::select_scan_node;
use utilize_index::utilize_index;
use utilize_node_by_id::utilize_node_by_id;

/// Collects all variable IDs referenced in an expression tree.
pub(crate) fn collect_expr_variables(expr: &DynTree<ExprIR<Variable>>) -> HashSet<u32> {
    let mut vars = HashSet::new();
    for idx in expr.root().indices::<Bfs>() {
        if let ExprIR::Variable(var) = expr.node(idx).data() {
            vars.insert(var.id);
        }
    }
    vars
}

/// Collects all variable IDs provided by a plan subtree.
pub(crate) fn collect_subtree_variables(node: &orx_tree::DynNode<IR>) -> HashSet<u32> {
    use crate::runtime::runtime::GetVariables;
    let mut vars = HashSet::new();
    for var in node.get_variables() {
        vars.insert(var.id);
    }
    vars
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
