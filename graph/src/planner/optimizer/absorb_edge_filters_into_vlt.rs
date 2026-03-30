use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::{
    parser::ast::{ExprIR, Variable},
    tree,
};

use super::super::IR;
use super::collect_expr_variables;

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

/// Absorbs edge filters into CondVarLenTraverse operators.
///
/// Detects patterns where a Filter sits directly above a CondVarLenTraverse
/// and the filter expression references only the VLT edge alias variable.
/// In that case, the filter is absorbed into the VLT's `edge_filter` field
/// so it can be evaluated per-hop during traversal.
///
/// For AND filters where some conjuncts reference the edge and others don't,
/// only the edge-referencing conjuncts are absorbed.
pub(super) fn absorb_edge_filters_into_vlt(optimized_plan: &mut DynTree<IR>) {
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
