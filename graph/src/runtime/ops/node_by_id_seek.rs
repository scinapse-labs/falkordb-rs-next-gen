//! Node-by-ID seek operator — retrieves nodes by their internal ID.
//!
//! Implements optimized lookups when the query planner determines that a
//! node's ID is constrained (e.g. `WHERE id(n) = 42`). The ID filter
//! expressions are evaluated to produce a candidate range, and only
//! non-deleted nodes within that range are yielded.
//!
//! ```text
//!  child iter ──► env ──► evaluate ID filter ──► range of IDs
//!                                                    │
//!                              for each ID in range (if not deleted):
//!                                env += {alias: Node(id)}
//!                                        │
//!                                    yield Env
//! ```

use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::NodeId;
use crate::parser::ast::{ExprIR, QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx};

pub struct NodeByIdSeekOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByIdSeekOp<'a> {
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

impl Iterator for NodeByIdSeekOp<'_> {
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
                    let g = self.runtime.g.borrow();
                    let node_pattern = self.node_pattern;
                    self.current = Some(Box::new(range.into_iter().filter_map(move |nid| {
                        if g.is_node_deleted(NodeId::from(nid)) {
                            None
                        } else {
                            let mut vars = vars.clone();
                            vars.insert(&node_pattern.alias, Value::Node(NodeId::from(nid)));
                            Some(Ok(vars))
                        }
                    })));
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
