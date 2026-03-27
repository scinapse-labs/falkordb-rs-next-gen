use graph::runtime::functions::{GraphFn, flush_udfs, register_udf, unregister_udf};
use graph::udf::get_udf_repo;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

pub fn graph_udf(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    let mut args_iter = args.into_iter().skip(1);
    let subcommand = args_iter.next_str()?;

    match subcommand.to_uppercase().as_str() {
        "LOAD" => udf_load(ctx, args_iter),
        "DELETE" => udf_delete(ctx, args_iter),
        "FLUSH" => udf_flush(ctx, args_iter),
        "LIST" => udf_list(ctx, args_iter),
        _ => Err(RedisError::String(format!(
            "Unknown UDF subcommand: {subcommand}"
        ))),
    }
}

fn udf_load(
    ctx: &Context,
    mut args: impl Iterator<Item = RedisString>,
) -> RedisResult {
    // Parse: GRAPH.UDF LOAD [REPLACE] <lib_name> <script>
    let first = args.next().ok_or(RedisError::Str(
        "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command",
    ))?;
    let first_str = first
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid argument"))?;

    let (replace, lib_name, script) = if first_str.eq_ignore_ascii_case("REPLACE") {
        let name = args.next().ok_or(RedisError::Str(
            "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command",
        ))?;
        let code = args.next().ok_or(RedisError::Str(
            "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command",
        ))?;
        (
            true,
            name.try_as_str()
                .map_err(|_| RedisError::Str("ERR invalid argument"))?
                .to_string(),
            code.try_as_str()
                .map_err(|_| RedisError::Str("ERR invalid argument"))?
                .to_string(),
        )
    } else {
        let code = args.next().ok_or(RedisError::Str(
            "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command",
        ))?;
        (
            false,
            first_str.to_string(),
            code.try_as_str()
                .map_err(|_| RedisError::Str("ERR invalid argument"))?
                .to_string(),
        )
    };

    // Check for any trailing invalid options
    if let Some(extra) = args.next() {
        let extra_str = extra.try_as_str().unwrap_or("");
        return Err(RedisError::String(format!(
            "Unknown option given: '{extra_str}'"
        )));
    }

    let repo = get_udf_repo();
    let function_names = repo
        .load(&lib_name, &script, replace)
        .map_err(RedisError::String)?;

    // Register each function in the dynamic function registry
    for qname in &function_names {
        let graph_fn = Arc::new(GraphFn::new_udf(qname));
        register_udf(qname, graph_fn);
    }

    ctx.replicate_verbatim();
    Ok(RedisValue::SimpleStringStatic("OK"))
}

fn udf_delete(
    ctx: &Context,
    mut args: impl Iterator<Item = RedisString>,
) -> RedisResult {
    let lib_name = args.next().ok_or(RedisError::Str(
        "ERR wrong number of arguments for 'GRAPH.UDF DELETE' command",
    ))?;
    let lib_name = lib_name
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid argument"))?;

    // Check for extra args
    if args.next().is_some() {
        return Err(RedisError::Str(
            "ERR wrong number of arguments for 'GRAPH.UDF DELETE' command",
        ));
    }

    let repo = get_udf_repo();
    let removed_names = repo.delete(lib_name).map_err(RedisError::String)?;

    for name in &removed_names {
        unregister_udf(name);
    }

    ctx.replicate_verbatim();
    Ok(RedisValue::SimpleStringStatic("OK"))
}

fn udf_flush(
    ctx: &Context,
    mut args: impl Iterator<Item = RedisString>,
) -> RedisResult {
    // Check for extra args
    if args.next().is_some() {
        return Err(RedisError::Str(
            "ERR wrong number of arguments for 'GRAPH.UDF FLUSH' command",
        ));
    }

    let repo = get_udf_repo();
    let removed_names = repo.flush();

    for name in &removed_names {
        unregister_udf(name);
    }
    flush_udfs();

    ctx.replicate_verbatim();
    Ok(RedisValue::SimpleStringStatic("OK"))
}

fn udf_list(
    _ctx: &Context,
    mut args: impl Iterator<Item = RedisString>,
) -> RedisResult {
    let mut filter: Option<String> = None;
    let mut with_code = false;

    // Parse optional arguments: [<lib_name>] [WITHCODE]
    // The Python client sends: GRAPH.UDF LIST [lib_name] [WITHCODE]
    while let Some(arg) = args.next() {
        let s = arg
            .try_as_str()
            .map_err(|_| RedisError::Str("ERR invalid argument"))?;
        match s.to_uppercase().as_str() {
            "WITHCODE" => with_code = true,
            _ => {
                if filter.is_some() {
                    return Err(RedisError::String(format!("Unknown option given: '{s}'")));
                }
                filter = Some(s.to_string());
            }
        }
    }

    let repo = get_udf_repo();
    let libs = repo.list(filter.as_deref(), with_code);

    let mut result = Vec::new();
    for lib in libs {
        let mut entry = vec![
            RedisValue::BulkString("library_name".into()),
            RedisValue::BulkString(lib.name),
            RedisValue::BulkString("functions".into()),
            RedisValue::Array(
                lib.function_names
                    .into_iter()
                    .map(RedisValue::BulkString)
                    .collect(),
            ),
        ];
        if let Some(code) = lib.code {
            entry.push(RedisValue::BulkString("library_code".into()));
            entry.push(RedisValue::BulkString(code.trim().to_string()));
        }
        result.push(RedisValue::Array(entry));
    }

    Ok(RedisValue::Array(result))
}
