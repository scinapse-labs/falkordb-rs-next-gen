//! OR-apply multiplexer operator — evaluates multiple existence-check branches.
//!
//! Implements disjunctive patterns like `WHERE EXISTS {...} OR EXISTS {...}`.
//! For each incoming row, iterates through branch sub-plans. A row is emitted
//! when any branch produces a result (or, for `anti` branches, when a branch
//! produces NO results). The `anti_flags` array controls whether each branch
//! uses normal or anti-semi-join semantics.
//!
//! ```text
//!  child iter ──► env ──► branch_0(env) ──► has result? XOR anti[0]
//!                    │         ├── pass ──► yield env
//!                    │         └── fail ──► try next branch
//!                    └──► branch_1(env) ──► ...
//! ```

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct OrApplyMultiplexerOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    anti_flags: &'a [bool],
    branch_indices: Vec<NodeIdx<Dyn<IR>>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> OrApplyMultiplexerOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        anti_flags: &'a [bool],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let num_branches = runtime.plan.node(idx).num_children() - 1;
        let mut branch_indices = Vec::with_capacity(num_branches);

        for i in 1..=num_branches {
            let branch_idx = runtime.plan.node(idx).child(i).idx();
            branch_indices.push(branch_idx);
        }

        Self {
            runtime,
            iter,
            anti_flags,
            branch_indices,
            idx,
        }
    }
}

impl Iterator for OrApplyMultiplexerOp<'_> {
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
            for (branch_num, branch_idx) in self.branch_indices.iter().enumerate() {
                let has_result = match self.runtime.run(*branch_idx) {
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
                let is_anti = self.anti_flags[branch_num];
                if has_result ^ is_anti {
                    let result = Ok(env);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            }
            // No branch matched for this input, continue to next
        }
    }
}
