//! Label scan operator — iterates all nodes with a given label.
//!
//! Implements the basic `MATCH (n:Label)` scan. For each incoming row,
//! iterates all nodes carrying the specified label(s) via the graph's
//! label matrix. When the node pattern includes inline attribute filters
//! (e.g. `(n:Person {name: "Alice"})`), nodes are filtered in-place
//! against those property values.

use std::sync::Arc;

use super::OpIter;
use crate::parser::ast::{QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct NodeByLabelScanOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByLabelScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            current: None,
            node_pattern,
            idx,
        }
    }
}

impl Iterator for NodeByLabelScanOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current {
                if let Some(item) = current.next() {
                    self.runtime.inspect_result(self.idx, &item);
                    return Some(item);
                }
                self.current = None;
            }
            let vars = match self.iter.next()? {
                Ok(vars) => vars,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let has_inline_attrs = self.node_pattern.attrs.root().children().next().is_some();
            let nodes_iter = self
                .runtime
                .g
                .borrow()
                .get_nodes(&self.node_pattern.labels, 0);
            let runtime = self.runtime;
            let node_pattern = self.node_pattern;

            if has_inline_attrs {
                self.current = Some(Box::new(nodes_iter.filter_map(move |v| {
                    let mut vars = vars.clone();
                    vars.insert(&node_pattern.alias, Value::Node(v));
                    let attrs = match runtime.run_expr(
                        &node_pattern.attrs,
                        node_pattern.attrs.root().idx(),
                        &vars,
                        None,
                    ) {
                        Ok(attrs) => attrs,
                        Err(e) => return Some(Err(e)),
                    };
                    if let Value::Map(attrs) = &attrs
                        && !attrs.is_empty()
                    {
                        let g = runtime.g.borrow();
                        for (attr, avalue) in attrs.iter() {
                            if let Some(pvalue) = g.get_node_attribute(v, attr) {
                                if *avalue == pvalue {
                                    continue;
                                }
                                return None;
                            }
                            return None;
                        }
                    }
                    Some(Ok(vars))
                })));
            } else {
                self.current = Some(Box::new(nodes_iter.map(move |v| {
                    let mut vars = vars.clone();
                    vars.insert(&node_pattern.alias, Value::Node(v));
                    Ok(vars)
                })));
            }
        }
    }
}
