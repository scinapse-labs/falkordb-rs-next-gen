//! Path builder operator — assembles named path values from bound variables.
//!
//! Implements Cypher named paths like `p = (a)-[r]->(b)`. For each incoming
//! row, collects the values of the variables that make up the path and
//! stores them as a `Value::Path` list under the path's alias.

use super::OpIter;
use crate::parser::ast::{QueryPath, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx};
use std::sync::Arc;

pub struct PathBuilderOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    paths: &'a [Arc<QueryPath<Variable>>],
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> PathBuilderOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        paths: &'a [Arc<QueryPath<Variable>>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            paths,
            idx,
        }
    }
}

impl Iterator for PathBuilderOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.iter.next()? {
            Ok(mut vars) => {
                let mut paths = self.paths.to_vec();
                for path in &mut paths {
                    let p: Result<_, String> = path
                        .vars
                        .iter()
                        .map(|v| {
                            vars.get(v)
                                .map_or_else(
                                    || Err(format!("Variable {} not found", v.as_str())),
                                    Ok,
                                )
                                .cloned()
                        })
                        .collect();
                    match p {
                        Ok(p) => vars.insert(&path.var, Value::Path(p)),
                        Err(e) => {
                            let result = Err(e);
                            self.runtime.inspect_result(self.idx, &result);
                            return Some(result);
                        }
                    }
                }
                Ok(vars)
            }
            Err(e) => Err(e),
        };
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
