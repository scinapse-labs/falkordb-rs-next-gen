//! Procedure call operator — invokes built-in graph procedures.
//!
//! Implements Cypher `CALL db.procedure(args) YIELD outputs`. Evaluates
//! argument expressions, validates types, invokes the procedure function,
//! and maps the returned list of maps to output environments. This is a
//! *blocking* operator: the procedure is called once and all results are
//! materialized before yielding.
//!
//! Only allowed in write queries when the procedure is marked as write;
//! returns an error for `GRAPH.RO_QUERY` on write procedures.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, functions::GraphFn, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::sync::Arc;
use thin_vec::ThinVec;

pub struct ProcedureCallOp<'a> {
    runtime: &'a Runtime,
    func: &'a Arc<GraphFn>,
    trees: &'a [QueryExpr<Variable>],
    name_outputs: &'a [Variable],
    results: Option<std::vec::IntoIter<Env>>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ProcedureCallOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        func: &'a Arc<GraphFn>,
        trees: &'a [QueryExpr<Variable>],
        name_outputs: &'a [Variable],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<Self, String> {
        if !runtime.write && func.write {
            return Err(String::from(
                "graph.RO_QUERY is to be executed only on read-only queries",
            ));
        }
        Ok(Self {
            runtime,
            func,
            trees,
            name_outputs,
            results: None,
            idx,
        })
    }

    fn init_results(&mut self) -> Result<(), String> {
        let args = self
            .trees
            .iter()
            .map(|ir| {
                self.runtime
                    .run_expr(ir, ir.root().idx(), &Env::default(), None)
            })
            .collect::<Result<ThinVec<_>, _>>()?;
        self.func.validate_args_type(&args)?;
        let res = (self.func.func)(self.runtime, args)?;
        match res {
            Value::List(arr) => {
                let results: Vec<Env> = arr
                    .into_iter()
                    .map(|v| {
                        let mut env = Env::default();
                        if let Value::Map(map) = v {
                            for output in self.name_outputs {
                                env.insert(
                                    output,
                                    map.get(output.name.as_ref().unwrap()).unwrap().clone(),
                                );
                            }
                        }
                        env
                    })
                    .collect();
                self.results = Some(results.into_iter());
                Ok(())
            }
            _ => unreachable!(),
        }
    }
}

impl Iterator for ProcedureCallOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.results.is_none()
            && let Err(e) = self.init_results()
        {
            return Some(Err(e));
        }
        let env = self.results.as_mut().unwrap().next()?;
        let result = Ok(env);
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
