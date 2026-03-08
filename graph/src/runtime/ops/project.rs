//! Project operator — evaluates return expressions and reshapes the environment.
//!
//! Implements Cypher `RETURN` / `WITH` projections. For each incoming row,
//! evaluates the projection expressions, copies carry-forward variables from
//! the parent scope, and emits a new environment containing only the
//! projected columns.

use super::OpIter;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ProjectOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    trees: &'a [(Variable, QueryExpr<Variable>)],
    copy_from_parent: &'a [(Variable, Variable)],
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ProjectOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        trees: &'a [(Variable, QueryExpr<Variable>)],
        copy_from_parent: &'a [(Variable, Variable)],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            trees,
            copy_from_parent,
            idx,
        }
    }
}

impl Iterator for ProjectOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.iter.next()? {
            Ok(vars) => {
                let mut return_vars = Env::default();
                for (old_var, new_var) in self.copy_from_parent {
                    if let Some(value) = vars.get(old_var) {
                        return_vars.insert(new_var, value.clone());
                    }
                }
                let mut err = None;
                for (name, tree) in self.trees {
                    match self.runtime.run_expr(tree, tree.root().idx(), &vars, None) {
                        Ok(value) => return_vars.insert(name, value),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                err.map_or_else(|| Ok(return_vars), Err)
            }
            Err(e) => Err(e),
        };
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
