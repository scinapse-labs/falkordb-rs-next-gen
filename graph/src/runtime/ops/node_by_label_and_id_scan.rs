//! Label-and-ID scan operator — retrieves nodes by label filtered by ID range.
//!
//! Combines a label scan with an ID range constraint from the optimizer.
//! Scans nodes with the given label starting from the minimum candidate ID,
//! and only yields those whose ID falls within the evaluated filter range.
//! More efficient than a full label scan when the ID range is narrow.

use std::sync::Arc;

use super::OpIter;
use crate::parser::ast::{ExprIR, QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx};

pub struct NodeByLabelAndIdScanOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByLabelAndIdScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            current: None,
            node_pattern,
            filter,
            idx,
        }
    }
}

impl Iterator for NodeByLabelAndIdScanOp<'_> {
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
            match self.runtime.evaluate_id_filter(self.filter, &vars) {
                Ok(Some(range)) => {
                    if let Some(min_id) = range.min() {
                        let g = self.runtime.g.borrow();
                        let node_pattern = self.node_pattern;
                        self.current = Some(Box::new(
                            g.get_nodes(&node_pattern.labels, min_id)
                                .filter_map(move |nid| {
                                    if range.contains(u64::from(nid)) {
                                        let mut vars = vars.clone();
                                        vars.insert(&node_pattern.alias, Value::Node(nid));
                                        Some(Ok(vars))
                                    } else {
                                        None
                                    }
                                }),
                        ));
                    } else {
                        self.current = Some(Box::new(std::iter::empty()));
                    }
                }
                Ok(None) => {
                    self.current = Some(Box::new(std::iter::empty()));
                }
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            }
        }
    }
}
