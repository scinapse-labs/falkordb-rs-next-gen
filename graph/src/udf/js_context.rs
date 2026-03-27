use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use rquickjs::{CatchResultExt, CaughtError, Context, Function, Persistent, Runtime as JsRuntime};
use thin_vec::ThinVec;

use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use crate::udf::get_udf_repo;
use crate::udf::js_classes::clear_current_graph;
use crate::udf::js_globals;
use crate::udf::type_convert;

/// Extract a human-readable error message from a CaughtError.
/// If `include_name` is true, prefix with error type (e.g., "SyntaxError: ...").
fn caught_error_message(
    err: &CaughtError<'_>,
    include_name: bool,
) -> String {
    match err {
        CaughtError::Error(e) => format!("{e}"),
        CaughtError::Exception(ex) => {
            let mut msg = ex.message().unwrap_or_default();
            // rquickjs ReferenceError message: "foo is not defined"
            // C QuickJS format: "'foo' is not defined"
            // Normalize to match the C format for compatibility.
            if let Some(stripped) = msg.strip_suffix(" is not defined")
                && !stripped.starts_with('\'')
            {
                msg = format!("'{stripped}' is not defined");
            }
            if include_name {
                // Try to get the error name (SyntaxError, TypeError, ReferenceError, etc.)
                let name = ex.as_object().get::<_, String>("name").ok();
                match name {
                    Some(n) if !n.is_empty() && n != "Error" => format!("{n}: {msg}"),
                    _ => msg,
                }
            } else {
                msg
            }
        }
        CaughtError::Value(val) => val.as_string().map_or_else(
            || format!("{val:?}"),
            |s| s.to_string().unwrap_or_else(|_| format!("{val:?}")),
        ),
    }
}

/// Atomic copies of JS config values, accessible without Redis GIL.
pub static JS_HEAP_SIZE: AtomicI64 = AtomicI64::new(256 * 1024 * 1024);
pub static JS_STACK_SIZE: AtomicI64 = AtomicI64::new(1024 * 1024);
pub static JS_TIMEOUT_MS: AtomicI64 = AtomicI64::new(0);

struct ThreadJsState {
    runtime: JsRuntime,
    context: Context,
    /// Cached function references: "lib.func" -> persistent JS function
    functions: HashMap<String, Persistent<Function<'static>>>,
    /// Version of the UdfRepo when this context was last rebuilt.
    version: u64,
}

thread_local! {
    static JS_STATE: RefCell<Option<ThreadJsState>> = const { RefCell::new(None) };
}

/// Validate a JS script by running it in a temporary context.
/// Returns the list of function names registered via `falkor.register()`.
pub fn validate_script(code: &str) -> Result<Vec<String>, String> {
    let rt = JsRuntime::new().map_err(|e| format!("Failed to create JS runtime: {e}"))?;
    let ctx = Context::full(&rt).map_err(|e| format!("Failed to create JS context: {e}"))?;

    ctx.with(|ctx| {
        let names = Rc::new(RefCell::new(Vec::new()));
        js_globals::setup_validate_globals(&ctx, names)?;

        ctx.eval::<(), _>(code)
            .catch(&ctx)
            .map_err(|e| caught_error_message(&e, true))?;

        js_globals::collect_validate_names(&ctx)
    })
}

/// Ensure the thread-local JS context is up-to-date with the global repository.
fn ensure_context_current() -> Result<(), String> {
    let repo = get_udf_repo();
    let current_version = repo.version();

    JS_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let needs_rebuild = (*state)
            .as_ref()
            .is_none_or(|s| s.version != current_version);

        if needs_rebuild {
            rebuild_context(&mut state, current_version)?;
        }
        Ok(())
    })
}

fn rebuild_context(
    state: &mut Option<ThreadJsState>,
    target_version: u64,
) -> Result<(), String> {
    // Drop old state in correct order: functions first, then context, then runtime.
    // Persistent references must be dropped while the runtime is still alive.
    if let Some(old) = state.take() {
        drop(old.functions);
        drop(old.context);
        drop(old.runtime);
    }

    let heap_size = JS_HEAP_SIZE.load(Ordering::Relaxed);
    let stack_size = JS_STACK_SIZE.load(Ordering::Relaxed);

    let rt = JsRuntime::new().map_err(|e| format!("Failed to create JS runtime: {e}"))?;
    rt.set_memory_limit(heap_size as usize);
    rt.set_max_stack_size(stack_size as usize);

    let ctx = Context::full(&rt).map_err(|e| format!("Failed to create JS context: {e}"))?;

    let functions = ctx.with(|ctx| {
        // Set up runtime globals (falkor.register stores to JS global object)
        js_globals::setup_runtime_globals(&ctx)?;

        // Evaluate all library scripts
        let repo = get_udf_repo();
        let libs = repo.get_all_libraries();
        for lib in &libs {
            // Set current library name so falkor.register() creates qualified keys
            ctx.eval::<(), _>(format!("globalThis.__falkor_current_lib = {:?};", lib.name))
                .map_err(|e| format!("Failed to set current lib: {e}"))?;

            ctx.eval::<(), _>(lib.code.as_str())
                .catch(&ctx)
                .map_err(|e| {
                    format!(
                        "Failed to load UDF library '{}': {}",
                        lib.name,
                        caught_error_message(&e, true)
                    )
                })?;
        }

        // Reset current lib name
        ctx.eval::<(), _>("globalThis.__falkor_current_lib = '';")
            .map_err(|e| format!("Failed to reset current lib: {e}"))?;

        // Collect function refs from JS global registry
        let raw_funcs = js_globals::collect_runtime_funcs(&ctx)?;

        // Map qualified names (lib.func) to their Persistent function references
        // Functions are now stored under qualified keys in JS registry
        let mut persistent_funcs = HashMap::new();
        for lib in &libs {
            for qname in &lib.function_names {
                if let Some(persistent) = raw_funcs.get(qname) {
                    persistent_funcs.insert(qname.to_lowercase(), persistent.clone());
                }
            }
        }

        Ok::<_, String>(persistent_funcs)
    })?;

    *state = Some(ThreadJsState {
        runtime: rt,
        context: ctx,
        functions,
        version: target_version,
    });

    Ok(())
}

/// Call a UDF by its qualified name (e.g., "mylib.myfunc").
/// This is called from the eval path when a UDF GraphFn is invoked.
pub fn call_udf_bridge(
    name: &str,
    rt: &Runtime,
    args: &ThinVec<Value>,
) -> Result<Value, String> {
    ensure_context_current()?;

    JS_STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().ok_or("JS context not initialized")?;

        let lower_name = name.to_lowercase();
        let persistent_fn = state
            .functions
            .get(&lower_name)
            .ok_or_else(|| format!("UDF function '{name}' not found in JS context"))?;

        // Set up timeout interrupt handler
        let timeout_ms = JS_TIMEOUT_MS.load(Ordering::Relaxed);
        if timeout_ms > 0 {
            let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
            state
                .runtime
                .set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)));
        }

        let result = state.context.with(|ctx| {
            let js_fn: Function = persistent_fn
                .clone()
                .restore(&ctx)
                .map_err(|e| format!("Failed to restore UDF function: {e}"))?;

            // Set graph reference for JS classes that need it
            crate::udf::js_classes::set_current_graph(rt.g.clone());

            // Convert arguments
            let js_args: Vec<rquickjs::Value> = args
                .iter()
                .map(|v| type_convert::value_to_js(&ctx, v, &rt.g))
                .collect::<Result<Vec<_>, _>>()?;

            // Call the function
            let result = js_fn
                .call::<(rquickjs::function::Rest<rquickjs::Value>,), rquickjs::Value>((
                    rquickjs::function::Rest(js_args),
                ))
                .catch(&ctx)
                .map_err(|e| {
                    let msg = caught_error_message(&e, false);
                    if msg.contains("interrupted") {
                        "UDF Exception: Query timed out".to_string()
                    } else if msg.contains("out of memory")
                        || msg.contains("InternalError: stack overflow")
                    {
                        "out of memory".to_string()
                    } else {
                        format!("UDF Exception: {msg}")
                    }
                })?;

            // Convert result back
            type_convert::js_to_value(result)
        });

        // Clear interrupt handler
        if timeout_ms > 0 {
            state
                .runtime
                .set_interrupt_handler(None::<Box<dyn FnMut() -> bool + Send>>);
        }

        // Clear graph reference
        clear_current_graph();

        result
    })
}
