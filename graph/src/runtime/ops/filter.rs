//! Filter operator — evaluates a boolean predicate on each row.
//!
//! Implements Cypher `WHERE` clauses. Rows where the predicate evaluates
//! to `true` are passed through; rows yielding `false` or `null` are
//! silently dropped. Non-boolean results produce a type-mismatch error.
//!
//! ```text
//!  child iter ──► env ──► evaluate predicate
//!                              │
//!                   ┌── true ──┼── false/null ──┐
//!                   ▼          │                ▼
//!               yield env      │             skip
//! ```

use super::OpIter;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct FilterOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    tree: &'a QueryExpr<Variable>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> FilterOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        tree: &'a QueryExpr<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            tree,
            idx,
        }
    }
}

impl Iterator for FilterOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let result = match self.iter.next()? {
                Ok(vars) => {
                    match self
                        .runtime
                        .run_expr(self.tree, self.tree.root().idx(), &vars, None)
                    {
                        Ok(Value::Bool(true)) => Ok(vars),
                        Ok(Value::Bool(false) | Value::Null) => continue,
                        Err(e) => Err(e),
                        Ok(value) => Err(format!(
                            "Type mismatch: expected Boolean but was {}",
                            value.name()
                        )),
                    }
                }
                Err(e) => Err(e),
            };
            self.runtime.inspect_result(self.idx, &result);
            return Some(result);
        }
    }
}
