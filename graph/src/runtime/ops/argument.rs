//! Argument operator — injects a single environment into a sub-plan.
//!
//! Used as the leaf of correlated sub-plans (Apply, SemiApply, Optional, etc.).
//! The parent operator sets the `env` field via `set_argument_env()` before
//! iteration begins; `ArgumentOp` then yields that single Env exactly once.
//!
//! ```text
//!  parent sets env ──► ArgumentOp ──► yields env once ──► child operators
//! ```

use crate::runtime::env::Env;

pub struct ArgumentOp {
    pub env: Option<Env>,
}

impl Default for ArgumentOp {
    fn default() -> Self {
        Self::new()
    }
}

impl ArgumentOp {
    #[must_use]
    pub const fn new() -> Self {
        Self { env: None }
    }
}

impl Iterator for ArgumentOp {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.env.take().map(Ok)
    }
}
