//! Aggregate operator — groups input rows and computes aggregate functions.
//!
//! Implements Cypher `RETURN ... , count(...)` / `WITH ... , sum(...)` etc.
//! This is a *blocking* operator: it consumes the entire child iterator on
//! the first `next()` call, groups rows by key expressions, and then yields
//! one result per group.
//!
//! ```text
//!  child iter ──► consume all rows
//!                     │
//!          ┌──────────┴──────────┐
//!          │  group by key hash  │   HashMap<u64, (key_env, acc_env)>
//!          └──────────┬──────────┘
//!                     │
//!          for each group: finalize agg expressions
//!                     │
//!                 yield Env ──► parent
//! ```
//!
//! When no key expressions exist (e.g. bare `RETURN count(*)`), a single
//! default group is pre-inserted so aggregation still produces one row.

use super::OpIter;
use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{env::Env, functions::FnType, runtime::Runtime, value::Value};
use orx_tree::{Dyn, DynNode, DynTree, NodeIdx, NodeRef};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use thin_vec::ThinVec;

pub struct AggregateOp<'a> {
    runtime: &'a Runtime,
    iter: Option<Box<OpIter<'a>>>,
    default_acc: Option<Env>,
    errors: std::vec::IntoIter<String>,
    cache: std::collections::hash_map::IntoIter<u64, (Env, Env)>,
    keys: &'a [(Variable, QueryExpr<Variable>)],
    agg: &'a [(Variable, QueryExpr<Variable>)],
    idx: NodeIdx<Dyn<IR>>,
}

impl<'a> AggregateOp<'a> {
    pub fn new(
        runtime: &'a Runtime,
        iter: Box<OpIter<'a>>,
        keys: &'a [(Variable, QueryExpr<Variable>)],
        agg: &'a [(Variable, QueryExpr<Variable>)],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        // Initialize default accumulator with zeros
        let mut default_acc = Env::default();
        for (_var, t) in agg {
            Self::set_agg_expr_zero(&t.root(), &mut default_acc);
        }

        Self {
            runtime,
            iter: Some(iter),
            default_acc: Some(default_acc),
            errors: Vec::new().into_iter(),
            cache: HashMap::new().into_iter(),
            keys,
            agg,
            idx,
        }
    }

    fn consume_input(&mut self) {
        let iter = self.iter.take().unwrap();
        let default_acc = self.default_acc.take().unwrap();

        // Cache: hash of key -> (key_env, accumulator_env)
        let mut cache: HashMap<u64, (Env, Env)> = HashMap::new();
        let mut errors: Vec<String> = Vec::new();

        // If keys is empty, pre-insert a default group
        if self.keys.is_empty() {
            let key = Env::default();
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            let k = hasher.finish();
            cache.insert(k, (key, default_acc.clone()));
        }

        // Consume all input items
        for item in iter {
            // If item is Err, store the error and skip aggregation
            let vars = match item {
                Err(e) => {
                    errors.push(e);
                    continue;
                }
                Ok(vars) => vars,
            };

            // Compute key env by evaluating key expressions
            let key_env = match (|| {
                let mut key_env = Env::default();
                for (name, tree) in self.keys {
                    let value = self
                        .runtime
                        .run_expr(tree, tree.root().idx(), &vars, None)?;
                    key_env.insert(name, value);
                }
                Ok::<Env, String>(key_env)
            })() {
                Ok(k) => k,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };

            // Hash the key to get group_key
            let mut hasher = DefaultHasher::new();
            key_env.hash(&mut hasher);
            let group_key = hasher.finish();

            // Insert into cache if new group, then run aggregation
            let entry = cache
                .entry(group_key)
                .or_insert_with(|| (key_env, default_acc.clone()));

            let mut curr = vars;
            for (_, tree) in self.agg {
                if let Err(e) = Self::run_agg_expr(
                    self.runtime,
                    tree,
                    tree.root().idx(),
                    &mut curr,
                    &mut entry.1,
                    group_key,
                ) {
                    errors.push(e);
                    break;
                }
            }
        }

        self.errors = errors.into_iter();
        self.cache = cache.into_iter();
    }

    fn run_agg_expr(
        runtime: &Runtime,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        curr: &mut Env,
        acc: &mut Env,
        agg_group_key: u64,
    ) -> Result<(), String> {
        match ir.node(idx).data() {
            ExprIR::FuncInvocation(func) if func.is_aggregate() => {
                let num_children = ir.node(idx).num_children();

                // The last child is always the accumulator key variable
                // Minimum valid structure: [arg, accumulator_key]
                if num_children < 2 {
                    return Err(String::from(
                        "Aggregation function must have at least one argument",
                    ));
                }

                let ExprIR::Variable(key) = ir.node(idx).child(num_children - 1).data() else {
                    return Err(String::from(
                        "Aggregation function must end with a variable",
                    ));
                };

                // Take ownership of accumulator (moves value, no clone)
                let prev_value = acc.take(key).unwrap_or(Value::Null);

                // PHASE 1:  Evaluate all arguments
                let arg_results: Result<ThinVec<Value>, String> = (0..num_children - 1)
                    .map(|i| {
                        let child = ir.node(idx).child(i);
                        runtime.run_expr(ir, child.idx(), curr, Some(agg_group_key))
                    })
                    .collect();

                let mut args = match arg_results {
                    Ok(a) => a,
                    Err(e) => {
                        // Restore accumulator before returning error
                        acc.insert(key, prev_value);
                        return Err(e);
                    }
                };

                // PHASE 2: Handle DISTINCT unpacking (if present)
                if num_children == 2 && matches!(ir.node(idx).child(0).data(), ExprIR::Distinct) {
                    let arg = args.remove(0);
                    if let Value::List(values) = arg {
                        args = values;
                    } else {
                        // Restore accumulator before returning error
                        acc.insert(key, prev_value);
                        return Err(String::from("DISTINCT should return a list"));
                    }
                }

                // PHASE 3: Validate argument types
                if let Err(e) = func.validate_args_type(&args) {
                    // Restore accumulator before returning error
                    acc.insert(key, prev_value);
                    return Err(e);
                }

                // PHASE 4: Validate domain constraints
                // This catches things like percentile out of [0.0, 1.0]
                if let Err(e) = func.validate_args_domain(&args) {
                    // Restore accumulator before returning error
                    acc.insert(key, prev_value);
                    return Err(e);
                }

                // PHASE 5: Push the accumulator as the last argument (moved, not cloned!)
                args.push(prev_value);

                // PHASE 6: Call the aggregation function
                // At this point, all validation is complete
                let new_value = (func.func)(runtime, args)?;

                // Store result back in accumulator
                acc.insert(key, new_value);
            }
            _ => {
                for child in ir.node(idx).children() {
                    Self::run_agg_expr(runtime, ir, child.idx(), curr, acc, agg_group_key)?;
                }
            }
        }
        Ok(())
    }

    fn set_agg_expr_zero(
        ir: &DynNode<ExprIR<Variable>>,
        env: &mut Env,
    ) {
        match ir.data() {
            ExprIR::FuncInvocation(func) if func.is_aggregate() => {
                if let FnType::Aggregation(zero, _) = &func.fn_type {
                    let ExprIR::Variable(key) = ir.child(ir.num_children() - 1).data() else {
                        unreachable!();
                    };
                    env.insert(key, zero.clone());
                }
            }
            _ => {
                for child in ir.children() {
                    Self::set_agg_expr_zero(&child, env);
                }
            }
        }
    }
}

impl Iterator for AggregateOp<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Consume all input on first call
        if self.iter.is_some() {
            self.consume_input();
        }

        // Drain errors first
        if let Some(e) = self.errors.next() {
            let result = Err(e);
            self.runtime.inspect_result(self.idx, &result);
            return Some(result);
        }

        // Finalize next cache entry
        let (_hash, (key, mut acc)) = self.cache.next()?;
        let result: Result<Env, String> = (|| {
            // Copy original variable refs from keys into acc so that
            // aggregation expressions can reference parent-scope variables
            // (e.g., map projections like n{.*, x: COLLECT(...)}).
            for (name, tree) in self.keys {
                if let ExprIR::Variable(original_var) = tree.root().data()
                    && let Some(value) = key.get(name)
                {
                    acc.insert(original_var, value.clone());
                }
            }
            // Merge key into acc
            acc.merge(key);
            // Evaluate and store finalized agg results
            for (name, tree) in self.agg {
                acc.insert(
                    name,
                    self.runtime.run_expr(tree, tree.root().idx(), &acc, None)?,
                );
            }
            Ok(acc)
        })();
        self.runtime.inspect_result(self.idx, &result);
        Some(result)
    }
}
