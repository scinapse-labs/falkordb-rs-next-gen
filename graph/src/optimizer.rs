use std::{rc::Rc, sync::Arc};

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    ast::{ExprIR, QueryNode},
    graph::graph::Graph,
    indexer::IndexQuery,
    planner::IR,
    runtime::functions::{FnType, get_functions},
    tree,
};

fn utilize_index(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

    for idx in indices {
        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(&idx).data()
            && !node.labels.is_empty()
            && let IR::Filter(filter) = optimized_plan.node(&idx).parent().unwrap().data()
            && matches!(filter.root().data(), ExprIR::Eq | ExprIR::Gt | ExprIR::Lt)
            && let ExprIR::FuncInvocation(inner_func) = filter.root().child(0).data()
            && inner_func.name == "property"
            && let ExprIR::String(attr) = filter.root().child(0).child(1).data()
            && graph.is_indexed(&node.labels[0], attr)
        {
            if matches!(filter.root().data(), ExprIR::Eq) {
                Some((
                    node.clone(),
                    node.labels[0].clone(),
                    Rc::new(IndexQuery::Equal(
                        attr.clone(),
                        Rc::new(filter.root().child(1).clone_as_tree()),
                    )),
                ))
            } else if matches!(filter.root().data(), ExprIR::Gt) {
                Some((
                    node.clone(),
                    node.labels[0].clone(),
                    Rc::new(IndexQuery::Range(
                        attr.clone(),
                        Some(Rc::new(filter.root().child(1).clone_as_tree())),
                        None,
                    )),
                ))
            } else if matches!(filter.root().data(), ExprIR::Lt) {
                Some((
                    node.clone(),
                    node.labels[0].clone(),
                    Rc::new(IndexQuery::Range(
                        attr.clone(),
                        None,
                        Some(Rc::new(filter.root().child(1).clone_as_tree())),
                    )),
                ))
            } else {
                None
            }
        } else {
            None
        };
        if let Some((node, index, query)) = node {
            let mut op = optimized_plan.node_mut(&idx);
            *op.data_mut() = IR::NodeByIndexScan { node, index, query };
            op.parent_mut().unwrap().take_out();
            break;
        }

        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(&idx).data() {
            get_index(graph, node)
        } else {
            None
        };
        if let Some((node, attr, filter)) = node
            && !node.labels.is_empty()
        {
            let mut op = optimized_plan.node_mut(&idx);
            *op.data_mut() = IR::NodeByIndexScan {
                node: node.clone(),
                index: node.labels[0].clone(),
                query: Rc::new(IndexQuery::Equal(
                    attr.clone(),
                    Rc::new(filter.root().child(1).clone_as_tree()),
                )),
            };
        }
    }
}

fn utilize_node_by_id(optimized_plan: &mut DynTree<IR>) {
    let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

    for idx in indices {
        let node = if let IR::NodeByLabelScan(node) = optimized_plan.node(&idx).data()
            && let IR::Filter(filter) = optimized_plan.node(&idx).parent().unwrap().data()
            && matches!(filter.root().data(), ExprIR::Eq)
            && let ExprIR::FuncInvocation(inner_func) = filter.root().child(0).data()
            && inner_func.name == "id"
            && let ExprIR::Variable(var) = filter.root().child(0).child(0).data()
            && node.alias == *var
        {
            Some((
                node.clone(),
                Rc::new(filter.root().child(1).clone_as_tree()),
            ))
        } else {
            None
        };
        if let Some((node, id)) = node {
            let mut op = optimized_plan.node_mut(&idx);
            *op.data_mut() = IR::NodeByIdScan { node, id };
            op.parent_mut().unwrap().take_out();
        }
    }
}

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

fn get_index(
    graph: &Graph,
    node: &Rc<QueryNode<Arc<String>>>,
) -> Option<(Rc<QueryNode<Arc<String>>>, Arc<String>, DynTree<ExprIR>)> {
    for label in &node.labels {
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
