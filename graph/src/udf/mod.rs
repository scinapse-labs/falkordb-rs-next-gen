pub mod js_classes;
pub mod js_context;
pub mod js_globals;
pub mod repository;
pub mod type_convert;

use repository::UdfRepo;
use std::sync::OnceLock;

static UDF_REPO: OnceLock<UdfRepo> = OnceLock::new();

pub fn init_udf_repo() {
    let _ = UDF_REPO.set(UdfRepo::new());
}

pub fn get_udf_repo() -> &'static UdfRepo {
    UDF_REPO.get().expect("UDF repository not initialized")
}
