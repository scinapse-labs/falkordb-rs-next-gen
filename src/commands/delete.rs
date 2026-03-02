//! `GRAPH.DELETE` command handler.
//!
//! Deletes an existing graph key from Redis.
//!
//! This command validates arity, checks that the key contains a graph native
//! type, and then delegates deletion to Redis key APIs.

use crate::{commands::EMPTY_KEY_ERR, graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString};
use std::sync::Arc;

pub fn graph_delete(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let key = args.next_arg()?;
    let key = ctx.open_key_writable(&key);
    if key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .is_some()
    {
        key.delete()
    } else {
        EMPTY_KEY_ERR
    }
}
