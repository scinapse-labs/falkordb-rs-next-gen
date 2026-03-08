//! Distinct operator — deduplicates result rows.
//!
//! Implements Cypher `RETURN DISTINCT ...`. Hashes the projected return
//! columns of each row and skips rows whose hash has been seen before.
//!
//! ```text
//!  child iter ──► env
//!                  │
//!          hash(return columns)
//!                  │
//!        ┌─── seen before? ───┐
//!        │ yes                │ no
//!        ▼                    ▼
//!      skip              yield + remember hash
//! ```

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::ValuesDeduper};
use orx_tree::{Dyn, NodeIdx};
use std::hash::{DefaultHasher, Hash, Hasher};

pub struct DistinctOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    deduper: ValuesDeduper,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> DistinctOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            deduper: ValuesDeduper::default(),
            idx,
        }
    }
}

impl Iterator for DistinctOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let vars = match self.iter.next()? {
                Ok(vars) => vars,
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            let mut hasher = DefaultHasher::new();
            for name in &self.runtime.return_names {
                vars.get(name)
                    .unwrap_or_else(|| unreachable!("Variable {} not found", name.as_str()))
                    .hash(&mut hasher);
            }
            if self.deduper.has_hash(hasher.finish()) {
                continue;
            }
            let result = Ok(vars);
            self.runtime.inspect_result(self.idx, &result);
            return Some(result);
        }
    }
}
