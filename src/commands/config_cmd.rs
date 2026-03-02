//! `GRAPH.CONFIG` command handler.
//!
//! Placeholder command implementation for graph-specific runtime configuration.
//!
//! This is intentionally minimal right now and serves as an extension point for
//! future graph-scoped config get/set semantics.

use redis_module::{Context, RedisResult, RedisString, RedisValue};

#[allow(clippy::unnecessary_wraps)]
pub fn graph_config(
    _ctx: &Context,
    _args: Vec<RedisString>,
) -> RedisResult {
    Ok(RedisValue::Integer(0))
}
