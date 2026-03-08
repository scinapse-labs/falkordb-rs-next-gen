//! Merge operator — implements Cypher `MERGE` (match-or-create) semantics.
//!
//! For each incoming row, executes the match sub-plan. If matches are found,
//! applies `ON MATCH SET` clauses; otherwise creates the pattern and applies
//! `ON CREATE SET` clauses. A pattern hash cache prevents duplicate creates
//! within the same query.
//!
//! ```text
//!  child iter ──► env
//!                  │
//!       ┌──────────┴──────────┐
//!       │  run match sub-plan │
//!       └──────────┬──────────┘
//!                  │
//!        ┌─ matches? ─┐
//!        │ yes         │ no
//!        ▼             ▼
//!   ON MATCH SET   CREATE pattern
//!        │         ON CREATE SET
//!        ▼             │
//!    yield rows        ▼
//!                  yield row
//! ```

use std::cell::OnceCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use super::OpIter;
use crate::graph::graph::LabelId;
use crate::parser::ast::{QueryGraph, SetItem, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, runtime::Runtime, value::Value};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct MergeOp<'a> {
    runtime: &'a Runtime,
    pub iter: Box<OpIter<'a>>,
    pending: Vec<Env>,
    merge_child_idx: NodeIdx<Dyn<IR>>,
    pattern: &'a QueryGraph<Arc<String>, Arc<String>, Variable>,
    resolved_pattern: OnceCell<QueryGraph<Arc<String>, LabelId, Variable>>,
    on_create_set_items: &'a [SetItem<Arc<String>, Variable>],
    resolved_on_create_set_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    on_match_set_items: &'a [SetItem<Arc<String>, Variable>],
    resolved_on_match_set_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    is_error: bool,
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> MergeOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        pattern: &'a QueryGraph<Arc<String>, Arc<String>, Variable>,
        on_create_set_items: &'a [SetItem<Arc<String>, Variable>],
        on_match_set_items: &'a [SetItem<Arc<String>, Variable>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let merge_child_idx = if runtime.plan.node(idx).num_children() == 1 {
            runtime.plan.node(idx).child(0).idx()
        } else {
            runtime.plan.node(idx).child(1).idx()
        };

        Self {
            runtime,
            iter,
            pending: Vec::new(),
            merge_child_idx,
            pattern,
            resolved_pattern: OnceCell::new(),
            on_create_set_items,
            resolved_on_create_set_items: OnceCell::new(),
            on_match_set_items,
            resolved_on_match_set_items: OnceCell::new(),
            is_error: false,
            idx,
        }
    }

    fn resolve_pattern(&self) -> &QueryGraph<Arc<String>, LabelId, Variable> {
        self.resolved_pattern.get_or_init(|| {
            let resolved = self.runtime.resolve_pattern(self.pattern);
            self.runtime.pending.borrow_mut().resize(
                self.runtime.g.borrow().node_cap(),
                self.runtime.g.borrow().labels_count(),
            );
            resolved
        })
    }

    fn resolve_on_create_set_items(&self) -> &Vec<SetItem<LabelId, Variable>> {
        self.resolved_on_create_set_items.get_or_init(|| {
            let resolved = self.runtime.resolve_set_items(self.on_create_set_items);
            self.runtime.pending.borrow_mut().resize(
                self.runtime.g.borrow().node_cap(),
                self.runtime.g.borrow().labels_count(),
            );
            resolved
        })
    }

    fn resolve_on_match_set_items(&self) -> &Vec<SetItem<LabelId, Variable>> {
        self.resolved_on_match_set_items.get_or_init(|| {
            let resolved = self.runtime.resolve_set_items(self.on_match_set_items);
            self.runtime.pending.borrow_mut().resize(
                self.runtime.g.borrow().node_cap(),
                self.runtime.g.borrow().labels_count(),
            );
            resolved
        })
    }

    fn do_create_fallback(
        &self,
        mut vars: Env,
    ) -> Result<Env, String> {
        let resolved_pattern = self.resolve_pattern();
        let pattern_hash = self.compute_merge_pattern_hash(resolved_pattern, &vars)?;

        let merge_cache = self.runtime.merge_pattern_cache.borrow_mut();

        if let Some(cached_vars) = merge_cache.get(&pattern_hash) {
            // Pattern already created, apply ON MATCH and return cached vars
            vars.merge(cached_vars.clone());
            drop(merge_cache);

            let resolved = self.resolve_on_match_set_items();
            self.runtime.set(resolved, &vars)?;
            Ok(vars)
        } else {
            // Pattern not yet created, create it
            drop(merge_cache);

            self.runtime.create(resolved_pattern, &mut vars)?;

            // Cache the created pattern
            self.runtime
                .merge_pattern_cache
                .borrow_mut()
                .insert(pattern_hash, vars.clone());

            let resolved = self.resolve_on_create_set_items();
            self.runtime.set(resolved, &vars)?;
            Ok(vars)
        }
    }

    fn compute_merge_pattern_hash(
        &self,
        pattern: &QueryGraph<Arc<String>, LabelId, Variable>,
        vars: &Env,
    ) -> Result<u64, String> {
        let mut hasher = DefaultHasher::new();

        // Hash nodes in the pattern
        for node in pattern.nodes() {
            // If the node variable exists in vars, hash its ID
            if let Some(value) = vars.get(&node.alias) {
                value.hash(&mut hasher);
            } else {
                // Hash the node structure (labels and attributes)
                for label in node.labels.iter() {
                    label.hash(&mut hasher);
                }
                let attrs =
                    self.runtime
                        .run_expr(&node.attrs, node.attrs.root().idx(), vars, None)?;

                // Validate that no attributes are NULL
                if let Value::Map(ref map) = attrs {
                    for (key, value) in map.iter() {
                        if *value == Value::Null {
                            return Err(format!(
                                "Cannot merge node using null property value for key '{key}'"
                            ));
                        }
                    }
                }

                attrs.hash(&mut hasher);
            }
        }

        // Hash relationships in the pattern
        for rel in pattern.relationships() {
            // Hash relationship type
            rel.types.hash(&mut hasher);

            // Hash from/to node references
            if let Some(value) = vars.get(&rel.from.alias) {
                value.hash(&mut hasher);
            }
            if let Some(value) = vars.get(&rel.to.alias) {
                value.hash(&mut hasher);
            }

            // Hash relationship attributes
            let attrs = self
                .runtime
                .run_expr(&rel.attrs, rel.attrs.root().idx(), vars, None)?;

            // Validate that no attributes are NULL
            if let Value::Map(ref map) = attrs {
                for (key, value) in map.iter() {
                    if *value == Value::Null {
                        return Err(format!(
                            "Cannot merge relationship using null property value for key '{key}'"
                        ));
                    }
                }
            }

            attrs.hash(&mut hasher);
        }

        Ok(hasher.finish())
    }
}

impl Iterator for MergeOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_error {
            return None;
        }

        loop {
            // Return buffered results
            if let Some(env) = self.pending.pop() {
                let result = Ok(env);
                self.runtime.inspect_result(self.idx, &result);
                return Some(result);
            }

            // Pull next parent env
            let env = match self.iter.next()? {
                Ok(env) => env,
                Err(e) => {
                    self.is_error = true;
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };

            // Check if all nodes in the pattern are already bound
            // If so, MERGE should only check existence (take 1 match)
            // If not, MERGE may need to return all matching nodes
            let all_nodes_bound = self
                .resolve_pattern()
                .nodes()
                .iter()
                .all(|node| env.get(&node.alias).is_some());

            let mut subtree = match self.runtime.run(self.merge_child_idx) {
                Ok(iter) => iter,
                Err(e) => {
                    self.is_error = true;
                    let result = Err(e);
                    self.runtime.inspect_result(self.idx, &result);
                    return Some(result);
                }
            };
            subtree.set_argument_env(&env);

            // Collect child matches
            let mut matches: Vec<Env> = Vec::new();
            for child_result in &mut subtree {
                match child_result {
                    Ok(v) => {
                        matches.push(v);
                        if all_nodes_bound {
                            break;
                        }
                    }
                    Err(e) => {
                        self.is_error = true;
                        let result = Err(e);
                        self.runtime.inspect_result(self.idx, &result);
                        return Some(result);
                    }
                }
            }

            if matches.is_empty() {
                // No matches found, do create fallback
                let result = self.do_create_fallback(env);
                if result.is_err() {
                    self.is_error = true;
                }
                self.runtime.inspect_result(self.idx, &result);
                return Some(result);
            }

            // Process matches: merge each with parent env, apply ON MATCH
            // Iterate in reverse and push so pop() yields results in original order
            for v in matches.into_iter().rev() {
                let mut vars = env.clone();
                vars.merge(v);
                let resolved = self.resolve_on_match_set_items();
                match self.runtime.set(resolved, &vars) {
                    Ok(()) => self.pending.push(vars),
                    Err(e) => {
                        self.is_error = true;
                        self.pending.clear();
                        let result = Err(e);
                        self.runtime.inspect_result(self.idx, &result);
                        return Some(result);
                    }
                }
            }
        }
    }
}
