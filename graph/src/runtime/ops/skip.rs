//! Skip operator — discards the first N rows from the child iterator.
//!
//! Implements Cypher `SKIP n`. Consumes and drops the first `n` rows,
//! then passes all subsequent rows through unchanged.

use super::OpIter;
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime};
use orx_tree::{Dyn, NodeIdx};

pub struct SkipOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    remaining_skip: usize,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SkipOp<'a> {
    pub const fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        skip: usize,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            iter,
            remaining_skip: skip,
            idx,
        }
    }
}

impl Iterator for SkipOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining_skip > 0 {
            self.remaining_skip -= 1;
            match self.iter.next()? {
                Ok(_) => {}
                Err(e) => {
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            }
        }
        let result = self.iter.next()?;
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
