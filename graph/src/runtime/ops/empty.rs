//! Empty operator — yields no rows.
//!
//! Acts as a terminal leaf in execution plans that require no input,
//! such as standalone `CREATE` statements with no preceding `MATCH`.
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
