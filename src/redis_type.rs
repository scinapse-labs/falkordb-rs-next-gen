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

use crate::graph_core::graph_free;
use graph::runtime::functions::{GraphFn, register_udf};
use graph::udf::get_udf_repo;
use redis_module::raw::{load_string_buffer, load_unsigned, save_string, save_unsigned};
use redis_module::{
    REDISMODULE_TYPE_METHOD_VERSION, RedisModuleIO, RedisModuleTypeMethods, native_types::RedisType,
};
use std::sync::Arc;
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

/// Save UDF libraries to RDB.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_save(
    rdb: *mut RedisModuleIO,
    _when: i32,
) {
    let repo = get_udf_repo();
    let libs = repo.serialize();
    save_unsigned(rdb, libs.len() as u64);
    for (name, code) in &libs {
        save_string(rdb, name);
        save_string(rdb, code);
    }
}

/// Load UDF libraries from RDB.
#[unsafe(no_mangle)]
unsafe extern "C" fn graph_aux_load(
    rdb: *mut RedisModuleIO,
    _encver: i32,
    _when: i32,
) -> i32 {
    let Ok(count) = load_unsigned(rdb) else {
        return 1; // REDISMODULE_ERR
    };

    let repo = get_udf_repo();
    let mut libs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = match load_string_buffer(rdb) {
            Ok(buf) => String::from_utf8_lossy(buf.as_ref()).to_string(),
            Err(_) => return 1,
        };
        let code = match load_string_buffer(rdb) {
            Ok(buf) => String::from_utf8_lossy(buf.as_ref()).to_string(),
            Err(_) => return 1,
        };
        libs.push((name, code));
    }

    // Load all libraries, registering their functions.
    // Clear existing UDFs first so stale functions from a previous snapshot
    // don't remain callable after loading the new payload.
    repo.flush();
    graph::runtime::functions::flush_udfs();
    match repo.deserialize(libs) {
        Ok(()) => {
            // Register bridge functions for each library's functions
            let all_libs = repo.get_all_libraries();
            for lib in &all_libs {
                for qname in &lib.function_names {
                    let graph_fn = Arc::new(GraphFn::new_udf(qname));
                    register_udf(qname, graph_fn);
                }
            }
            0 // REDISMODULE_OK
        }
        Err(_) => 1, // REDISMODULE_ERR
    }
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

        aux_load: Some(graph_aux_load),
        aux_save: None,
        aux_save2: Some(graph_aux_save),
        aux_save_triggers: 1, // REDISMODULE_AUX_BEFORE_RDB

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
