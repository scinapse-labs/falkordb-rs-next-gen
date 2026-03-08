//! Index scan operator — retrieves nodes using a secondary index.
//!
//! Implements optimized lookups via equality, range, or point-distance
//! index queries. The index query parameters are evaluated at runtime
//! from the current environment. When the node pattern has inline
//! attribute filters, matching nodes are further filtered against those
//! property constraints.

use std::sync::Arc;

use super::OpIter;
use crate::index::indexer::IndexQuery;
use crate::parser::ast::{QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct NodeByIndexScanOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    index: &'a Arc<String>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByIndexScanOp<'a> {
    fn evaluate_index_query(
        runtime: &Runtime,
        query: &IndexQuery<QueryExpr<Variable>>,
        vars: &Env,
    ) -> Result<IndexQuery<Value>, String> {
        match query {
            IndexQuery::Equal(key, value) => {
                let value = runtime.run_expr(value, value.root().idx(), vars, None)?;
                Ok(IndexQuery::Equal(key.clone(), value))
            }
            IndexQuery::Range(key, min, max) => {
                let (min, max) = match (min, max) {
                    (Some(min), Some(max)) => {
                        let min = runtime.run_expr(min, min.root().idx(), vars, None)?;
                        let max = runtime.run_expr(max, max.root().idx(), vars, None)?;
                        (Some(min), Some(max))
                    }
                    (Some(min), None) => {
                        let min = runtime.run_expr(min, min.root().idx(), vars, None)?;
                        (Some(min), None)
                    }
                    (None, Some(max)) => {
                        let max = runtime.run_expr(max, max.root().idx(), vars, None)?;
                        (None, Some(max))
                    }
                    (None, None) => (None, None),
                };
                Ok(IndexQuery::Range(key.clone(), min, max))
            }
            IndexQuery::Point { key, point, radius } => {
                let point = runtime.run_expr(point, point.root().idx(), vars, None)?;
                let radius = runtime.run_expr(radius, radius.root().idx(), vars, None)?;
                Ok(IndexQuery::Point {
                    key: key.clone(),
                    point,
                    radius,
                })
            }
            _ => todo!(),
        }
    }

    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        index: &'a Arc<String>,
        query: &'a IndexQuery<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            current: None,
            node_pattern,
            index,
            query,
            idx,
        }
    }
}

impl Iterator for NodeByIndexScanOp<'_> {
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
            let q = match Self::evaluate_index_query(self.runtime, self.query, &vars) {
                Ok(q) => q,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let has_inline_attrs = self.node_pattern.attrs.root().children().next().is_some();
            let runtime = self.runtime;
            let node_pattern = self.node_pattern;

            if has_inline_attrs {
                self.current = Some(Box::new(
                    runtime
                        .g
                        .borrow()
                        .get_indexed_nodes(self.index, q)
                        .filter_map(move |v| {
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
                        }),
                ));
            } else {
                self.current = Some(Box::new(
                    runtime
                        .g
                        .borrow()
                        .get_indexed_nodes(self.index, q)
                        .map(move |v| {
                            let mut vars = vars.clone();
                            vars.insert(&node_pattern.alias, Value::Node(v));
                            Ok(vars)
                        }),
                ));
            }
        }
    }
}
