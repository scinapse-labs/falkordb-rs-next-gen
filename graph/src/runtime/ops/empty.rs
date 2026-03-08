//! Empty operator — yields no rows.
//!
//! Used as a placeholder child for operators that manage their own
//! children (e.g. `Union`), and as the return value for DDL
//! operations (`CreateIndex`, `DropIndex`) that produce no result rows.
//! Always returns `None` on the first call to `next()`.

use crate::runtime::env::Env;

#[derive(Default)]
pub struct EmptyOp;

impl Iterator for EmptyOp {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
