//! Semi-apply operator — existence-based filtering via a sub-plan.
//!
//! Implements `WHERE EXISTS { ... }` (semi-join) and
//! `WHERE NOT EXISTS { ... }` (anti-semi-join). For each incoming row,
//! runs the right sub-plan; if it produces at least one result the row
//! is passed through (or filtered out for anti mode).
//!
//! ```text
//!  left iter ──► env ──► right sub-plan(env)
//!                              │
//!                   ┌── has result? XOR is_anti ──┐
//!                   │ pass                         │ fail
//!                   ▼                              ▼
//!               yield env                        skip
//! ```

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct SemiApplyOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    is_anti: bool,
    right_child_idx: NodeIdx<Dyn<IR>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SemiApplyOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        is_anti: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let right_child_idx = runtime.plan.node(idx).child(1).idx();

        Self {
            runtime,
            iter,
            is_anti,
            right_child_idx,
            idx,
        }
    }
}

impl Iterator for SemiApplyOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let env = match self.iter.next()? {
                Ok(env) => env,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let has_result = match self.runtime.run(self.right_child_idx) {
                Ok(mut iter) => {
                    iter.set_argument_env(&env);
                    match iter.next() {
                        Some(Ok(_)) => true,
                        Some(Err(e)) => {
                            let result = Err(e);
                            self.runtime.inspect_result(self.idx, &result);
                            return Some(result);
                        }
                        None => false,
                    }
                }
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            if has_result ^ self.is_anti {
                let result = Ok(env);
                self.runtime.inspect_result(self.idx, &result);
                return Some(result);
            }
        }
    }
}
