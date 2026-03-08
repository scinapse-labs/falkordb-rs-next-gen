//! Cartesian product operator — combines rows from independent sub-plans.
//!
//! Produces every combination of rows from the left iterator and each right
//! branch by re-executing the right sub-plan for every left row. When there
//! are more than two children, intermediate `CartesianProductOp` nodes are
//! chained so the final result is the full cross-product of all branches.
//!
//! ```text
//!  left iter ──► env_L ──┐
//!                         ├──► right sub-plan ──► env_R
//!                         │         merge(env_L, env_R) ──► yield
//!                         ├──► right sub-plan ──► env_R'
//!                         │         merge(env_L, env_R') ──► yield
//!                         ...
//!  left iter ──► env_L' ──┘
//!                         ...
//! ```
//!
//! Corresponds to Cypher `MATCH (a), (b)` where `(a)` and `(b)` are
//! independent patterns with no shared variables.

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CartesianProductOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<(Env, Box<OpIter<'a>>)>,
    child_idx: NodeIdx<Dyn<IR>>,
    idx: Option<NodeIdx<Dyn<IR>>>,
    is_error: bool,
    pub(crate) argument_env: Option<Env>,
}

impl<'a> CartesianProductOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let node = runtime.plan.node(idx);
        let mut children: Vec<_> = node.children().skip(1).map(|c| c.idx()).collect();
        let last_child_idx = children
            .pop()
            .expect("CartesianProduct must have at least 2 children");

        let mut current_iter = iter;
        for child_idx in children {
            current_iter = Box::new(OpIter::CartesianProduct(CartesianProductOp {
                runtime,
                iter: current_iter,
                current: None,
                child_idx,
                idx: None,
                is_error: false,
                argument_env: None,
            }));
        }

        CartesianProductOp {
            runtime,
            iter: current_iter,
            current: None,
            child_idx: last_child_idx,
            idx: Some(idx),
            is_error: false,
            argument_env: None,
        }
    }
}

impl Iterator for CartesianProductOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_error {
            return None;
        }
        loop {
            if let Some((ref env, ref mut child_iter)) = self.current {
                match child_iter.next() {
                    Some(Ok(vars2)) => {
                        let mut vars = env.clone();
                        vars.merge(vars2);
                        let result = Ok(vars);
                        if let Some(idx) = self.idx {
                            self.runtime.inspect_result(idx, &result);
                        }
                        return Some(result);
                    }
                    Some(Err(e)) => {
                        self.is_error = true;
                        self.current = None;
                        let result = Err(e);
                        if let Some(idx) = self.idx {
                            self.runtime.inspect_result(idx, &result);
                        }
                        return Some(result);
                    }
                    None => {
                        self.current = None;
                    }
                }
            }
            match self.iter.next()? {
                Ok(vars1) => match self.runtime.run(self.child_idx) {
                    Ok(mut child_iter) => {
                        if let Some(ref env) = self.argument_env {
                            child_iter.set_argument_env(env);
                        }
                        self.current = Some((vars1, Box::new(child_iter)));
                    }
                    Err(e) => {
                        self.is_error = true;
                        let result = Err(e);
                        if let Some(idx) = self.idx {
                            self.runtime.inspect_result(idx, &result);
                        }
                        return Some(result);
                    }
                },
                Err(e) => {
                    self.is_error = true;
                    let result = Err(e);
                    if let Some(idx) = self.idx {
                        self.runtime.inspect_result(idx, &result);
                    }
                    return Some(result);
                }
            }
        }
    }
}
