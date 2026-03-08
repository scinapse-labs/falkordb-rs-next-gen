//! Union operator — concatenates results from multiple sub-plans.
//!
//! Implements Cypher `UNION` / `UNION ALL`. Iterates through each child
//! sub-plan in order, yielding all rows from the first child before
//! moving to the second, and so on.
//!
//! ```text
//!  child_0 ──► yield all rows
//!  child_1 ──► yield all rows
//!  ...
//!  child_N ──► yield all rows
//! ```

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct UnionOp<'a> {
    runtime: &'a Runtime,
    current: Option<Box<OpIter<'a>>>,
    current_child: usize,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnionOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            current: None,
            current_child: 0,
            idx,
        }
    }
}

impl Iterator for UnionOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current {
                if let Some(item) = current.next() {
                    self.runtime.inspect_result(self.idx, &item);
                    return Some(item);
                }
                self.current = None;
                self.current_child += 1;
            }
            let current = self.runtime.plan.node(self.idx);
            if self.current_child >= current.num_children() {
                return None;
            }
            let child_idx = current.child(self.current_child).idx();
            match self.runtime.run(child_idx) {
                Ok(child_iter) => {
                    self.current = Some(Box::new(child_iter));
                }
                Err(e) => {
                    self.current_child = current.num_children();
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            }
        }
    }
}
