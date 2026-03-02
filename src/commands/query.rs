//! `GRAPH.QUERY` command handler.
//!
//! Executes read/write Cypher queries, auto-creates graphs when missing,
//! and supports compact output and memory-tracking flags.
//!
//! ## Execution flow
//! ```text
//! GRAPH.QUERY key query [--compact] [--track-memory]
//!        |
//!        +--> parse flags
//!        +--> open writable key
//!        +--> create graph if missing
//!        +--> delegate to graph_core::query_mut(..., write=true)
//!        +--> return NoReply (client unblocked asynchronously later)
//! ```
//!
//! The handler is intentionally thin: it validates command arguments and
//! chooses the target graph, while runtime execution is centralized in
//! `graph_core`.

use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, query_mut},
    redis_type::GRAPH_TYPE,
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::Arc;
#[cfg(feature = "fuzz")]
use std::{fs::File, io::Write};

#[cfg(feature = "fuzz")]
static mut file_id: i32 = 0;

#[allow(static_mut_refs)]
pub fn graph_query(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    let query = args.next_str()?;

    #[cfg(feature = "fuzz")]
    unsafe {
        let mut file = File::create(format!(
            "fuzz/corpus/fuzz_target_runtime/output{file_id}.txt"
        ))?;
        file.write_all(query.as_bytes())?;
        drop(file);
        file_id += 1;
    }

    let mut compact = false;
    let mut track_memory = false;
    while let Ok(arg) = args.next_str() {
        if arg == "--compact" {
            compact = true;
        } else if arg == "--track-memory" {
            track_memory = true;
        }
    }

    let key = ctx.open_key_writable(&key_str);

    if let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        query_mut(ctx, graph, query, compact, true, track_memory);
    } else {
        let graph = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        query_mut(ctx, &graph, query, compact, true, track_memory);
        key.set_value(&GRAPH_TYPE, graph)?;
    }

    RedisResult::Ok(RedisValue::NoReply)
}
