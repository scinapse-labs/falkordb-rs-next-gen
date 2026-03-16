//! Batch-mode optional operator — implements OPTIONAL MATCH semantics.
//!
//! For each input batch, runs the sub-plan once with all active rows as a
//! multi-row argument batch. Uses `origin_row` on output envs to track which
//! input rows had results. For input rows with no results, emits a fallback
//! row with the specified variables set to NULL.
//!
//! Falls back to per-row sub-plan execution when the sub-plan contains blocking
//! operators (Aggregate) that accumulate state across all rows.

use std::collections::{HashSet, VecDeque};

use crate::parser::ast::Variable;
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    ops::apply::has_aggregate,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Active batched sub-plan for all rows from one input batch.
struct ActiveSubPlan<'a> {
    /// Saved input envs for fallback (indexed by origin_row).
    input_envs: Vec<Env<'a>>,
    /// The single sub-plan iterator producing result batches for all input rows.
    subtree: BatchOp<'a>,
    /// Tracks which origin_rows have produced at least one result.
    matched_origins: HashSet<u32>,
}

/// Per-row sub-plan state (used when batching is not possible).
struct PendingOptional<'a> {
    env: Env<'a>,
    subtree: BatchOp<'a>,
    had_result: bool,
    current_batch: Option<(Batch<'a>, usize)>,
}

pub struct OptionalOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    vars: &'a [Variable],
    optional_child_idx: NodeIdx<Dyn<IR>>,
    /// Batched mode state.
    active: Option<Box<ActiveSubPlan<'a>>>,
    /// Per-row mode state.
    pending: VecDeque<PendingOptional<'a>>,
    can_batch: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> OptionalOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        vars: &'a [Variable],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let optional_child_idx = if runtime.plan.node(idx).num_children() == 1 {
            runtime.plan.node(idx).child(0).idx()
        } else {
            runtime.plan.node(idx).child(1).idx()
        };

        let can_batch = !has_aggregate(&runtime.plan, optional_child_idx);

        Self {
            runtime,
            child,
            vars,
            optional_child_idx,
            active: None,
            pending: VecDeque::new(),
            can_batch,
            idx,
        }
    }

    // -----------------------------------------------------------------------
    // Batched mode helpers
    // -----------------------------------------------------------------------

    fn drain_active(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        while envs.len() < BATCH_SIZE {
            let Some(ref mut plan) = self.active else {
                break;
            };

            match plan.subtree.next() {
                Some(Ok(sub_batch)) => {
                    for env in sub_batch.active_env_iter() {
                        plan.matched_origins.insert(env.origin_row);
                        envs.push(env.clone_pooled(self.runtime.env_pool));
                        if envs.len() >= BATCH_SIZE {
                            return Ok(());
                        }
                    }
                }
                Some(Err(e)) => return Err(e),
                None => {
                    let plan = self.active.take().unwrap();
                    for (i, input_env) in plan.input_envs.iter().enumerate() {
                        if !plan.matched_origins.contains(&(i as u32)) {
                            let mut fallback = input_env.clone_pooled(self.runtime.env_pool);
                            for v in self.vars {
                                fallback.insert(v, Value::Null);
                            }
                            envs.push(fallback);
                        }
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    fn next_batched(&mut self) -> Option<Result<Batch<'a>, String>> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        if let Err(e) = self.drain_active(&mut envs) {
            return Some(Err(e));
        }

        while envs.len() < BATCH_SIZE {
            if self.active.is_some() {
                if let Err(e) = self.drain_active(&mut envs) {
                    return Some(Err(e));
                }
                continue;
            }

            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            let input_envs: Vec<Env<'a>> = batch
                .active_env_iter()
                .enumerate()
                .map(|(i, env)| {
                    let mut e = env.clone_pooled(self.runtime.env_pool);
                    e.origin_row = i as u32;
                    e
                })
                .collect();

            let arg_envs: Vec<Env<'a>> = input_envs
                .iter()
                .map(|e| e.clone_pooled(self.runtime.env_pool))
                .collect();

            let mut subtree = match self.runtime.run_batch(self.optional_child_idx) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            subtree.set_argument_batch(Batch::from_envs(arg_envs));

            self.active = Some(Box::new(ActiveSubPlan {
                input_envs,
                subtree,
                matched_origins: HashSet::new(),
            }));

            if let Err(e) = self.drain_active(&mut envs) {
                return Some(Err(e));
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }

    // -----------------------------------------------------------------------
    // Per-row mode helpers (fallback for sub-plans with Aggregate)
    // -----------------------------------------------------------------------

    fn drain_pending(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        while envs.len() < BATCH_SIZE {
            let Some(p) = self.pending.front_mut() else {
                break;
            };

            if let Some((batch, pos)) = &mut p.current_batch {
                let active: Vec<usize> = batch.active_indices().collect();
                while *pos < active.len() && envs.len() < BATCH_SIZE {
                    let row = batch.env_ref(active[*pos]);
                    p.had_result = true;
                    envs.push(row.clone_pooled(self.runtime.env_pool));
                    *pos += 1;
                }
                if *pos >= active.len() {
                    p.current_batch = None;
                } else {
                    return Ok(());
                }
            }

            match p.subtree.next() {
                Some(Ok(sub_batch)) => {
                    p.current_batch = Some((sub_batch, 0));
                }
                Some(Err(e)) => return Err(e),
                None => {
                    if !p.had_result {
                        let mut fallback = p.env.clone_pooled(self.runtime.env_pool);
                        for v in self.vars {
                            fallback.insert(v, Value::Null);
                        }
                        envs.push(fallback);
                    }
                    self.pending.pop_front();
                }
            }
        }
        Ok(())
    }

    fn next_per_row(&mut self) -> Option<Result<Batch<'a>, String>> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        if let Err(e) = self.drain_pending(&mut envs) {
            return Some(Err(e));
        }

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for env in batch.active_env_iter() {
                let mut subtree = match self.runtime.run_batch(self.optional_child_idx) {
                    Ok(iter) => iter,
                    Err(e) => return Some(Err(e)),
                };
                subtree.set_argument_batch(Batch::from_envs(vec![
                    env.clone_pooled(self.runtime.env_pool),
                ]));

                self.pending.push_back(PendingOptional {
                    env: env.clone_pooled(self.runtime.env_pool),
                    subtree,
                    had_result: false,
                    current_batch: None,
                });
            }

            if let Err(e) = self.drain_pending(&mut envs) {
                return Some(Err(e));
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}

impl<'a> Iterator for OptionalOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.can_batch {
            self.next_batched()
        } else {
            self.next_per_row()
        }
    }
}
