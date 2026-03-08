//! Optional operator — implements Cypher `OPTIONAL MATCH` semantics.
//!
//! Executes a sub-plan for each incoming row. If the sub-plan produces
//! results, they are yielded as-is. If it produces no results, a single
//! fallback row is emitted with the specified variables set to `NULL`.
//!
//! ```text
//!  child iter ──► env ──► run sub-plan
//!                              │
//!                   ┌── has results? ──┐
//!                   │ yes              │ no
//!                   ▼                  ▼
//!             yield results     yield env + NULLs
//! ```

use super::OpIter;
use crate::parser::ast::Variable;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct OptionalOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<OpIter<'a>>>,
    had_result: bool,
    fallback_env: Env,
    vars: &'a [Variable],
    optional_child_idx: NodeIdx<Dyn<IR>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> OptionalOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        vars: &'a [Variable],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let optional_child_idx = if runtime.plan.node(idx).num_children() == 1 {
            runtime.plan.node(idx).child(0).idx()
        } else {
            runtime.plan.node(idx).child(1).idx()
        };
        Self {
            runtime,
            iter,
            current: None,
            had_result: false,
            fallback_env: Env::default(),
            vars,
            optional_child_idx,
            idx,
        }
    }
}

impl Iterator for OptionalOp<'_> {
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
                if !self.had_result {
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
            self.fallback_env = env.clone();
            for v in self.vars {
                self.fallback_env.insert(v, Value::Null);
            }
            let mut subtree = match self.runtime.run(self.optional_child_idx) {
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
