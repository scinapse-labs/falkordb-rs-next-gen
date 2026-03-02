//! Redis native type declaration for graph storage.
//!
//! Defines `GRAPH_TYPE` and its RDB load/save/free callbacks used to persist
//! and manage graph values in Redis keys.
//!
//! ## Value lifecycle
//! ```text
//! set_value(GRAPH_TYPE, Arc<RwLock<ThreadedGraph>>)
//!              |
//!              +--> key survives Redis operations
//!              |
//!              +--> on key delete/overwrite/expire:
//!                        Redis invokes `free` callback -> graph_free()
//! ```
//!
//! `rdb_load`/`rdb_save` are currently placeholders; long-term persistence
//! support is designed to plug in here.

use crate::graph_core::graph_free;
use redis_module::{
    REDISMODULE_TYPE_METHOD_VERSION, RedisModuleIO, RedisModuleTypeMethods, native_types::RedisType,
};
use std::{os::raw::c_void, ptr::null_mut};

#[unsafe(no_mangle)]
#[allow(clippy::missing_const_for_fn)]
unsafe extern "C" fn graph_rdb_load(
    _: *mut RedisModuleIO,
    _: i32,
) -> *mut c_void {
    null_mut()
}

#[unsafe(no_mangle)]
#[allow(clippy::missing_const_for_fn)]
unsafe extern "C" fn graph_rdb_save(
    _: *mut RedisModuleIO,
    _: *mut c_void,
) {
}

pub static GRAPH_TYPE: RedisType = RedisType::new(
    "graphdata",
    0,
    RedisModuleTypeMethods {
        version: REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(graph_rdb_load),
        rdb_save: Some(graph_rdb_save),
        aof_rewrite: None,
        free: Some(graph_free),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        free_effort: None,
        unlink: None,
        copy: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);
