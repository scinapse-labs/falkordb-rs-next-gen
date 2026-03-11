//! Apply operator — correlated sub-query execution.
//!
//! For each row from the left (driving) iterator, the right sub-plan is
//! instantiated and executed with the current environment passed via
//! `set_argument_env`. All rows produced by the right side are yielded.
//!
//! ```text
//!  left iter ──► env ──┐
//!                       ├──► right sub-plan(env) ──► yield each result
//!  left iter ──► env ──┘
//!                       ...
//! ```
//!
//! When the right child is an `Optional` node, the operator falls back to
//! emitting the parent env with NULL-filled optional variables if the right
//! sub-plan produces no results.

use super::OpIter;
use crate::parser::ast::Variable;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ApplyOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<OpIter<'a>>>,
    had_result: bool,
    fallback_env: Env,
    optional_vars: Option<Vec<Variable>>,
    child_idx: NodeIdx<Dyn<IR>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ApplyOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let right_child_idx = runtime.plan.node(idx).child(1).idx();
        let right_data = runtime.plan.node(right_child_idx).data().clone();

        let (optional_vars, child_idx) = match right_data {
            IR::Optional(ref vars) => {
                let optional_child_idx = runtime.plan.node(right_child_idx).child(0).idx();
                (Some(vars.clone()), optional_child_idx)
            }
            _ => (None, right_child_idx),
        };

        Self {
            runtime,
            iter,
            current: None,
            had_result: false,
            fallback_env: Env::default(),
            optional_vars,
            child_idx,
            idx,
        }
    }
}

impl Iterator for ApplyOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current {
                if let Some(item) = current.next() {
                    self.had_result = true;
                    self.runtime.inspect_result(self.idx, &item);
                    return Some(item);
                }
                self.current = None;
                if self.optional_vars.is_some() && !self.had_result {
                    let result = Ok(self.fallback_env.clone());
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            }
            let env = match self.iter.next()? {
                Ok(env) => env,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            if let Some(ref vars) = self.optional_vars {
                self.fallback_env = env.clone();
                for v in vars {
                    self.fallback_env.insert(v, Value::Null);
                }
            }
            let mut subtree = match self.runtime.run(self.child_idx) {
                Ok(iter) => iter,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            subtree.set_argument_env(&env);
            self.current = Some(Box::new(subtree));
            self.had_result = false;
        }
    }
}
