//! Batch-mode conditional traverse operator — single-hop relationship expansion.
//!
//! For each active row in the input batch, extracts the source node and scans
//! matching relationships, producing output rows with relationship and endpoint
//! bindings. Uses per-row fallback through the row-based traversal logic.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CondTraverseOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    /// Buffered output rows from a partial expansion.
    pending: VecDeque<Env<'a>>,
    /// Current input batch being expanded (may span multiple output batches).
    current_batch: Option<Batch<'a>>,
    /// Index into active rows of the current batch.
    current_pos: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CondTraverseOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            relationship_pattern,
            pending: VecDeque::new(),
            current_batch: None,
            current_pos: 0,
            idx,
        }
    }

    fn expand_row(
        &self,
        env: &Env<'a>,
        out: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        let filter_attrs = runtime.run_expr(&rp.attrs, rp.attrs.root().idx(), env, None)?;
        let from_node_attrs =
            runtime.run_expr(&rp.from.attrs, rp.from.attrs.root().idx(), env, None)?;
        let to_node_attrs = runtime.run_expr(&rp.to.attrs, rp.to.attrs.root().idx(), env, None)?;

        let from_id = env.get(&rp.from.alias).and_then(|v| match v {
            Value::Node(id) => Some(*id),
            _ => None,
        });
        if from_id.is_none() && env.is_bound(&rp.from.alias) {
            return Ok(());
        }
        let to_id = env.get(&rp.to.alias).and_then(|v| match v {
            Value::Node(id) => Some(*id),
            _ => None,
        });
        if to_id.is_none() && env.is_bound(&rp.to.alias) {
            return Ok(());
        }

        let g = runtime.g.borrow();
        let forward = g.get_relationships(&rp.types, &rp.from.labels, &rp.to.labels);
        let pairs: Vec<_> = forward
            .map(|(src, dst)| (src, dst, false))
            .chain(if rp.bidirectional {
                let rev: Vec<_> = g
                    .get_relationships(&rp.types, &rp.to.labels, &rp.from.labels)
                    .filter(|(s, d)| s != d)
                    .map(|(s, d)| (s, d, true))
                    .collect();
                rev.into_iter()
            } else {
                Vec::new().into_iter()
            })
            .collect();

        for (src, dst, is_reverse) in pairs {
            let (from_node, to_node) = if is_reverse { (dst, src) } else { (src, dst) };
            if from_id.is_some() && from_id.unwrap() != from_node {
                continue;
            }
            if to_id.is_some() && to_id.unwrap() != to_node {
                continue;
            }
            // Check from node attrs
            if let Value::Map(ref attrs) = from_node_attrs
                && !attrs.is_empty()
            {
                let mut skip = false;
                for (attr, avalue) in attrs.iter() {
                    match g.get_node_attribute(from_node, attr) {
                        Some(pvalue) if pvalue == *avalue => {}
                        _ => {
                            skip = true;
                            break;
                        }
                    }
                }
                if skip {
                    continue;
                }
            }
            // Check to node attrs
            if let Value::Map(ref attrs) = to_node_attrs
                && !attrs.is_empty()
            {
                let mut skip = false;
                for (attr, avalue) in attrs.iter() {
                    match g.get_node_attribute(to_node, attr) {
                        Some(pvalue) if pvalue == *avalue => {}
                        _ => {
                            skip = true;
                            break;
                        }
                    }
                }
                if skip {
                    continue;
                }
            }
            // Scan edges
            for id in g.get_src_dest_relationships(src, dst, &rp.types) {
                if let Value::Map(ref filter_map) = filter_attrs
                    && !filter_map.is_empty()
                {
                    let mut matches = true;
                    for (attr, avalue) in filter_map.iter() {
                        if let Some(pvalue) = g.get_relationship_attribute(id, attr) {
                            if *avalue == pvalue {
                                continue;
                            }
                            matches = false;
                            break;
                        }
                        matches = false;
                        break;
                    }
                    if !matches {
                        continue;
                    }
                }
                let mut row = env.clone_pooled(runtime.env_pool);
                row.insert(&rp.alias, Value::Relationship(Box::new((id, src, dst))));
                row.insert(&rp.from.alias, Value::Node(from_node));
                row.insert(&rp.to.alias, Value::Node(to_node));
                out.push(row);
            }
        }
        Ok(())
    }

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending rows are exhausted.
    fn drain_pending(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) {
        while envs.len() < BATCH_SIZE {
            if let Some(row) = self.pending.pop_front() {
                envs.push(row);
            } else {
                break;
            }
        }
    }
}

impl<'a> Iterator for CondTraverseOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        self.drain_pending(&mut envs);

        loop {
            if envs.len() >= BATCH_SIZE {
                break;
            }

            // Get or fetch current batch.
            if self.current_batch.is_none() {
                match self.child.next() {
                    Some(Ok(b)) => {
                        self.current_batch = Some(b);
                        self.current_pos = 0;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }

            {
                let batch = self.current_batch.as_ref().unwrap();
                let active: Vec<usize> = batch.active_indices().collect();

                while self.current_pos < active.len() {
                    let row_idx = active[self.current_pos];
                    self.current_pos += 1;
                    let env = batch.env_ref(row_idx);
                    let mut expanded = Vec::new();
                    if let Err(e) = self.expand_row(env, &mut expanded) {
                        return Some(Err(e));
                    }
                    self.pending.extend(expanded);

                    if self.pending.len() >= BATCH_SIZE {
                        break;
                    }
                }
            }

            self.drain_pending(&mut envs);

            // Check if batch is exhausted.
            if let Some(ref batch) = self.current_batch {
                let active_len = batch.active_indices().count();
                if self.current_pos >= active_len {
                    self.current_batch = None;
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
