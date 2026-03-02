//! Global Redis module configuration values.
//!
//! Stores runtime-configurable values exposed through Redis configuration,
//! such as query plan cache size and CSV import folder.
//!
//! These values are wrapped in `RedisGILGuard` to match Redis module threading
//! constraints: reads/writes to config occur while respecting Redis global
//! synchronization semantics.

use lazy_static::lazy_static;
use redis_module::RedisGILGuard;

lazy_static! {
    pub static ref CONFIGURATION_IMPORT_FOLDER: RedisGILGuard<String> =
        RedisGILGuard::new("/var/lib/FalkorDB/import/".into());
    pub static ref CONFIGURATION_CACHE_SIZE: RedisGILGuard<i64> = RedisGILGuard::new(25.into());
}
