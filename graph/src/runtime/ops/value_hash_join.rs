//! Batch-mode value hash join operator — equi-join via build/probe hash table.
//!
//! Replaces CartesianProduct + equality Filter with a hash join when the
//! optimizer detects an equality predicate between a left and right expression.
//!
//! ```text
//!  Phase 1: BUILD (materialize right sub-plan into hash table)
//!
//!     Right child ──► for each row: hash(rhs_expr) ──► HashMap<hash, Vec<(key, envs)>>
//!
//!  Phase 2: PROBE (stream left rows, look up matches)
//!
//!     Left child ──► for each row: hash(lhs_expr) ──► probe table
//!                                                        │
//!                          ┌──────────────────────────────┘
//!                          │  for each matching right env:
//!                          │    merged = left_env + right_env
//!                          ▼
//!                     output batches
//! ```
//!
//! The hash table uses chaining for collision resolution: each bucket stores
//! a `Vec<(Value, Vec<Env>)>` where exact key equality is checked during
//! probe. NULL keys are skipped on both sides (Cypher NULL != NULL semantics).

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    eval::ExprEval,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ValueHashJoinOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pub(crate) right: Box<BatchOp<'a>>,
    pub(crate) lhs_exp: &'a QueryExpr<Variable>,
    pub(crate) rhs_exp: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Hash table: hash(value) -> Vec<(Value, Vec<Env>)>
    /// We use a Vec of (key, envs) pairs per bucket to handle hash collisions.
    pub(crate) hash_table: Option<HashMap<u64, Vec<(Value, Vec<Env<'a>>)>>>,
    /// Current block of left-side rows being probed.
    pub(crate) left_envs: Vec<Env<'a>>,
    /// Current position within `left_envs`.
    pub(crate) left_pos: usize,
    /// Current position within the matched right envs for the current left row.
    pub(crate) right_match_envs: Vec<Env<'a>>,
    pub(crate) right_match_pos: usize,
}

impl<'a> ValueHashJoinOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        right: Box<BatchOp<'a>>,
        lhs_exp: &'a QueryExpr<Variable>,
        rhs_exp: &'a QueryExpr<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            right,
            lhs_exp,
            rhs_exp,
            idx,
            hash_table: None,
            left_envs: Vec::new(),
            left_pos: 0,
            right_match_envs: Vec::new(),
            right_match_pos: 0,
        }
    }

    /// Build the hash table from the right sub-plan.
    fn materialize_right(&mut self) -> Result<HashMap<u64, Vec<(Value, Vec<Env<'a>>)>>, String> {
        let pool = self.runtime.env_pool;
        let eval = ExprEval::from_runtime(self.runtime);
        let rhs_idx = self.rhs_exp.root().idx();
        let mut table: HashMap<u64, Vec<(Value, Vec<Env<'a>>)>> = HashMap::new();

        for result in self.right.by_ref() {
            let batch = result?;
            for env in batch.active_env_iter() {
                let key = eval.eval(self.rhs_exp, rhs_idx, Some(env), None)?;
                if matches!(key, Value::Null) {
                    // NULL != NULL in Cypher, skip
                    continue;
                }
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                let hash = hasher.finish();

                let bucket = table.entry(hash).or_default();
                // Find existing entry with equal key, or create new
                if let Some(entry) = bucket.iter_mut().find(|(k, _)| *k == key) {
                    entry.1.push(env.clone_pooled(pool));
                } else {
                    bucket.push((key, vec![env.clone_pooled(pool)]));
                }
            }
        }

        Ok(table)
    }
}

impl<'a> Iterator for ValueHashJoinOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let pool = self.runtime.env_pool;

        // Lazy materialization of right side
        if self.hash_table.is_none() {
            match self.materialize_right() {
                Ok(table) => {
                    if table.is_empty() {
                        return None;
                    }
                    self.hash_table = Some(table);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let mut envs = Vec::with_capacity(BATCH_SIZE);

        loop {
            // Drain remaining matches from current left row
            while envs.len() < BATCH_SIZE && self.right_match_pos < self.right_match_envs.len() {
                let mut merged = self.left_envs[self.left_pos].clone_pooled(pool);
                merged.merge(&self.right_match_envs[self.right_match_pos]);
                envs.push(merged);
                self.right_match_pos += 1;
            }

            if self.right_match_pos >= self.right_match_envs.len() {
                // Move to next left row
                if self.right_match_pos > 0 || self.left_pos > 0 {
                    self.left_pos += 1;
                }
                self.right_match_envs.clear();
                self.right_match_pos = 0;
            }

            if envs.len() >= BATCH_SIZE {
                return Some(Ok(Batch::from_envs(envs)));
            }

            // Process more left rows
            while self.left_pos < self.left_envs.len() {
                // Inline probe to avoid borrow conflict with self.right_match_envs
                let eval = ExprEval::from_runtime(self.runtime);
                let lhs_idx = self.lhs_exp.root().idx();
                let key = match eval.eval(
                    self.lhs_exp,
                    lhs_idx,
                    Some(&self.left_envs[self.left_pos]),
                    None,
                ) {
                    Ok(k) => k,
                    Err(e) => return Some(Err(e)),
                };
                if matches!(key, Value::Null) {
                    self.left_pos += 1;
                    continue;
                }
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                let hash = hasher.finish();

                let table = self.hash_table.as_ref().unwrap();
                let matched = table.get(&hash).map_or_else(Vec::new, |bucket| {
                    bucket
                        .iter()
                        .filter(|(k, _)| *k == key)
                        .map(|(_, envs)| envs)
                        .collect::<Vec<_>>()
                });
                if matched.is_empty() {
                    self.left_pos += 1;
                    continue;
                }
                // Flatten all matching right envs
                self.right_match_envs.clear();
                self.right_match_pos = 0;
                for group in matched {
                    for env in group {
                        self.right_match_envs.push(env.clone_pooled(pool));
                    }
                }
                // Now drain from right_match
                while envs.len() < BATCH_SIZE && self.right_match_pos < self.right_match_envs.len()
                {
                    let mut merged = self.left_envs[self.left_pos].clone_pooled(pool);
                    merged.merge(&self.right_match_envs[self.right_match_pos]);
                    envs.push(merged);
                    self.right_match_pos += 1;
                }
                if self.right_match_pos >= self.right_match_envs.len() {
                    self.left_pos += 1;
                    self.right_match_envs.clear();
                    self.right_match_pos = 0;
                }
                if envs.len() >= BATCH_SIZE {
                    return Some(Ok(Batch::from_envs(envs)));
                }
            }

            // Need more left rows
            self.left_envs.clear();
            self.left_pos = 0;

            match self.child.next() {
                Some(Ok(batch)) => {
                    for env in batch.active_env_iter() {
                        self.left_envs.push(env.clone_pooled(pool));
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    if envs.is_empty() {
                        return None;
                    }
                    return Some(Ok(Batch::from_envs(envs)));
                }
            }
        }
    }
}
