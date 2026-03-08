//! Commit operator — materializes pending mutations and flushes them to the graph.
//!
//! This is a *blocking* operator: it drains the entire child iterator first
//! (collecting all result environments), then calls `pending.commit()` to
//! apply batched creates, deletes, and property changes to the underlying
//! graph. After the commit succeeds, the collected environments are yielded.
//!
//! ```text
//!  child iter ──► collect all rows ──► pending.commit(graph) ──► yield rows
//! ```
//!
//! Only allowed in write queries; returns an error for `GRAPH.RO_QUERY`.

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx};

pub struct CommitOp<'a> {
    runtime: &'a Runtime,
    iter: Option<Box<OpIter<'a>>>,
    results: std::vec::IntoIter<Env>,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CommitOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<Self, String> {
        if !runtime.write {
            return Err(String::from(
                "graph.RO_QUERY is to be executed only on read-only queries",
            ));
        }
        Ok(Self {
            runtime,
            iter: Some(iter),
            results: Vec::new().into_iter(),
            idx,
        })
    }
}

impl Iterator for CommitOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(iter) = self.iter.take() {
            let results = match iter.collect::<Result<Vec<_>, String>>() {
                Ok(results) => results,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = self
                .runtime
                .pending
                .borrow_mut()
                .commit(&self.runtime.g, &self.runtime.stats)
            {
                return Some(Err(e));
            }
            self.results = results.into_iter();
        }
        let env = self.results.next()?;
        let result = Ok(env);
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
