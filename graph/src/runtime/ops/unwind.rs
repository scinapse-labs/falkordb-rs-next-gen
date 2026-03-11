//! Batch-mode unwind operator — expands a list expression into individual rows.
//!
//! For each active row in each input batch, evaluates the list expression and
//! expands it into individual rows. Output rows are accumulated into batches
//! of up to `BATCH_SIZE`.
//!
//! Uses `batch.env_ref()` to evaluate expressions without cloning input rows,
//! and only clones when producing output rows.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::sync::Arc;

pub struct UnwindOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    list: &'a QueryExpr<Variable>,
    name: &'a Variable,
    /// Buffered output rows that didn't fit in the previous batch.
    buffer: Vec<Env<'a>>,
    /// Buffered input batch with remaining rows to process.
    /// Stored as (batch, remaining_active_indices) when we had to stop mid-batch
    /// because the output was full.
    pending_input: Option<(Batch<'a>, Vec<usize>)>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnwindOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        list: &'a QueryExpr<Variable>,
        name: &'a Variable,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            list,
            name,
            buffer: Vec::new(),
            pending_input: None,
            idx,
        }
    }

    /// Processes rows from a batch starting at `start_pos` in the active indices list.
    /// Returns `Err(error_string)` on expression evaluation failure, or
    /// `Ok(next_pos)` — the index into `active` where processing stopped.
    /// If `next_pos == active.len()`, all rows were processed.
    fn expand_batch(
        &mut self,
        batch: &Batch<'a>,
        active: &[usize],
        start_pos: usize,
        envs: &mut Vec<Env<'a>>,
    ) -> Result<usize, String> {
        let pool = self.runtime.env_pool;

        for (pos, &row) in active.iter().enumerate().skip(start_pos) {
            // Use env_ref to evaluate the expression without cloning the input row.
            let env_ref = batch.env_ref(row);
            let value = self
                .runtime
                .run_expr(self.list, self.list.root().idx(), env_ref, None)?;

            match value {
                Value::Null => {
                    // Null produces zero output rows — skip without cloning.
                }
                Value::List(list) => {
                    let items: Vec<Value> = Arc::unwrap_or_clone(list).into();
                    for item in items {
                        let mut out_row = env_ref.clone_pooled(pool);
                        out_row.insert(self.name, item);
                        if envs.len() < BATCH_SIZE {
                            envs.push(out_row);
                        } else {
                            self.buffer.push(out_row);
                        }
                    }
                }
                other => {
                    // Scalar value — produces exactly one output row.
                    let mut out_row = env_ref.clone_pooled(pool);
                    out_row.insert(self.name, other);
                    if envs.len() < BATCH_SIZE {
                        envs.push(out_row);
                    } else {
                        self.buffer.push(out_row);
                    }
                }
            }

            if envs.len() >= BATCH_SIZE {
                // Return the position *after* the current row so we resume
                // from the next unprocessed row.
                return Ok(pos + 1);
            }
        }

        Ok(active.len())
    }
}

impl<'a> Iterator for UnwindOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain any buffered output rows from a previous partial expansion.
        if !self.buffer.is_empty() {
            let drain_count = self.buffer.len().min(BATCH_SIZE);
            envs.extend(self.buffer.drain(..drain_count));
        }

        // Resume processing a pending input batch if we stopped mid-batch.
        if envs.len() < BATCH_SIZE
            && let Some((batch, active)) = self.pending_input.take()
        {
            // We resume from position 0 in the remaining active indices.
            match self.expand_batch(&batch, &active, 0, &mut envs) {
                Err(e) => return Some(Err(e)),
                Ok(next_pos) => {
                    if next_pos < active.len() {
                        // Still didn't finish this batch — save remainder.
                        self.pending_input = Some((batch, active[next_pos..].to_vec()));
                    }
                }
            }
        }

        // Pull from child until we fill a batch or exhaust input.
        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            let active: Vec<usize> = batch.active_indices().collect();
            match self.expand_batch(&batch, &active, 0, &mut envs) {
                Err(e) => return Some(Err(e)),
                Ok(next_pos) => {
                    if next_pos < active.len() {
                        // Output is full but input batch has remaining rows.
                        // Save them so we process them on the next call.
                        self.pending_input = Some((batch, active[next_pos..].to_vec()));
                        break;
                    }
                }
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
