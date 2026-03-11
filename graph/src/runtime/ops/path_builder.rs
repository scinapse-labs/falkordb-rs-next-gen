//! Batch-mode path builder operator — assembles named path values.
//!
//! Implements Cypher named paths like `p = (a)-[r]->(b)`. For each path,
//! reads the component variable columns via `read_columns`, maps each row
//! into a `Value::Path`, and writes the result column back via `write_column`.

use std::sync::Arc;

use crate::parser::ast::{QueryPath, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};
use thin_vec::ThinVec;

pub struct PathBuilderOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    paths: &'a [Arc<QueryPath<Variable>>],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> PathBuilderOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        paths: &'a [Arc<QueryPath<Variable>>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            paths,
            idx,
        }
    }
}

impl<'a> Iterator for PathBuilderOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        for path in self.paths {
            let var_ids: Vec<u32> = path.vars.iter().map(|v| v.id).collect();

            // read_columns returns row-major: rows[row][var_index] = &Value
            let rows = batch.read_columns(&var_ids);

            let path_values: Result<Vec<Value>, String> = rows
                .iter()
                .map(|row| {
                    let elems: Result<ThinVec<Value>, String> = row
                        .iter()
                        .enumerate()
                        .map(|(i, val)| {
                            if matches!(val, Value::Null) {
                                Err(format!("Variable {} not found", path.vars[i].as_str()))
                            } else {
                                Ok((*val).clone())
                            }
                        })
                        .collect();
                    Ok(Value::Path(Arc::new(elems?)))
                })
                .collect();

            let path_values = match path_values {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };

            batch.write_column(path.var.id, path_values);
        }

        Some(Ok(batch))
    }
}
