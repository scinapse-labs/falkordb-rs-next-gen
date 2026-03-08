//! Sort operator — orders result rows by one or more expressions.
//!
//! Implements Cypher `ORDER BY`. This is a *blocking* operator: it
//! consumes all child rows on the first `next()` call, evaluates the
//! sort-key expressions for each row, sorts in-memory using a stable
//! sort with per-key ascending/descending control, and then yields
//! rows in sorted order.

use super::OpIter;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    env::Env,
    runtime::Runtime,
    value::{CompareValue, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::cmp::Ordering;

type SortItem = (Env, Vec<(Value, bool)>);

pub struct SortOp<'a> {
    runtime: &'a Runtime,
    iter: Option<Box<OpIter<'a>>>,
    trees: &'a [(QueryExpr<Variable>, bool)],
    results: std::vec::IntoIter<SortItem>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SortOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        trees: &'a [(QueryExpr<Variable>, bool)],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter: Some(iter),
            trees,
            results: Vec::new().into_iter(),
            idx,
        }
    }
}

impl Iterator for SortOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(iter) = self.iter.take() {
            let mut items: Vec<SortItem> = match iter
                .map(|item| {
                    let env = item?;
                    let sort_keys = self
                        .trees
                        .iter()
                        .map(|(tree, desc)| {
                            Ok((
                                self.runtime.run_expr(tree, tree.root().idx(), &env, None)?,
                                *desc,
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok((env, sort_keys))
                })
                .collect::<Result<_, String>>()
            {
                Ok(items) => items,
                Err(e) => return Some(Err(e)),
            };
            items.sort_by(|(_, a), (_, b)| {
                a.iter()
                    .zip(b)
                    .fold(Ordering::Equal, |acc, ((a, desc_a), (b, _))| {
                        if acc != Ordering::Equal {
                            return acc;
                        }
                        let (ordering, _) = a.compare_value(b);
                        if *desc_a {
                            ordering.reverse()
                        } else {
                            ordering
                        }
                    })
            });
            self.results = items.into_iter();
        }
        let (env, _) = self.results.next()?;
        let result = Ok(env);
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
