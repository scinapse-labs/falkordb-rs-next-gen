//! Unwind operator — expands a list expression into individual rows.
//!
//! Implements Cypher `UNWIND list AS item`. For each incoming row,
//! evaluates the list expression, then yields one row per element with
//! the element bound to the specified variable.
//!
//! ```text
//!  child iter ──► env ──► evaluate list expr ──► [v1, v2, v3]
//!                                                  │   │   │
//!                            env + {item: v1} ◄────┘   │   │
//!                            env + {item: v2} ◄────────┘   │
//!                            env + {item: v3} ◄────────────┘
//! ```

use super::OpIter;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct UnwindOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    current: Option<Box<dyn Iterator<Item = Result<Env, String>> + 'a>>,
    list: &'a QueryExpr<Variable>,
    name: &'a Variable,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnwindOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        list: &'a QueryExpr<Variable>,
        name: &'a Variable,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            current: None,
            list,
            name,
            idx,
        }
    }
}

impl Iterator for UnwindOp<'_> {
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
            let value = match self
                .runtime
                .run_iter_expr(self.list, self.list.root().idx(), &vars)
            {
                Ok(v) => v,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let name = self.name;
            self.current = Some(Box::new(value.map(move |v| {
                let mut vars = vars.clone();
                vars.insert(name, v);
                Ok(vars)
            })));
        }
    }
}
