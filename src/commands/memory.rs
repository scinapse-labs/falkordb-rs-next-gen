//! `GRAPH.MEMORY` command handler.
//!
//! Returns memory usage, in bytes, for a specific graph key.
//!
//! The value comes from graph internals (`memory_usage`) and includes in-memory
//! graph structures currently tracked by the graph implementation.

use crate::{graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

pub fn graph_memory(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    if args.len() != 1 {
        return Err(RedisError::WrongArity);
    }

    let key = args.next_arg()?;
    let key = ctx.open_key(&key);

    let g = key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .expect("Graph does not exist");

    Ok(RedisValue::Integer(
        g.read().graph.read().borrow().memory_usage() as i64,
    ))
}
