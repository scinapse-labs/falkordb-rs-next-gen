use std::sync::Arc;

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    parser::ast::{ExprIR, Variable},
    runtime::{eval::ExprEval, value::Value},
    tree,
};

use super::super::IR;

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
pub(super) fn eliminate_true_filters(optimized_plan: &mut DynTree<IR>) {
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
