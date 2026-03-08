//! Limit operator — caps the number of rows yielded.
//!
//! Implements Cypher `LIMIT n`. Passes through at most `n` rows from
//! the child iterator, then returns `None` for all subsequent calls.
//! Enables early termination of the upstream pipeline.

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx};

pub struct LimitOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    remaining: usize,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> LimitOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        limit: usize,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            remaining: limit,
            idx,
        }
    }
}

impl Iterator for LimitOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let result = self.iter.next()?;
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
