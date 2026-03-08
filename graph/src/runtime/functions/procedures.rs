//! CALL procedures – `db.labels`, `db.indexes`, full-text helpers, etc.

#![allow(clippy::unnecessary_wraps)]

use super::{FnType, Functions, Type};
use crate::{
    index::indexer::{IndexInfo, IndexType},
    runtime::{ordermap::OrderMap, runtime::Runtime, value::Value},
};
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

pub fn register(funcs: &mut Functions) {
    // ── db.labels ──────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.labels",
        args: [],
        ret: Type::Any,
        procedure: ["label"],
        fn db_labels(runtime, _args) {
            Ok(Value::List(
                runtime
                    .g
                    .borrow()
                    .get_labels()
                    .iter()
                    .map(|l| {
                        let mut map = OrderMap::default();
                        map.insert(Arc::new(String::from("label")), Value::String(l.clone()));
                        Value::Map(map)
                    })
                    .collect(),
            ))
        }
    );

    // ── db.relationshiptypes ───────────────────────────────────────────
    cypher_fn!(funcs, "db.relationshiptypes",
        args: [],
        ret: Type::Any,
        procedure: ["relationshipType"],
        fn db_types(runtime, _args) {
            Ok(Value::List(
                runtime
                    .g
                    .borrow()
                    .get_types()
                    .iter()
                    .map(|t| {
                        let mut map = OrderMap::default();
                        map.insert(
                            Arc::new(String::from("relationshipType")),
                            Value::String(t.clone()),
                        );
                        Value::Map(map)
                    })
                    .collect(),
            ))
        }
    );

    // ── db.propertykeys ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.propertykeys",
        args: [],
        ret: Type::Any,
        procedure: ["propertyKey"],
        fn db_properties(runtime, _args) {
            Ok(Value::List(
                runtime
                    .g
                    .borrow()
                    .get_attrs()
                    .map(|p| {
                        let mut map = OrderMap::default();
                        map.insert(
                            Arc::new(String::from("propertyKey")),
                            Value::String(p.clone()),
                        );
                        Value::Map(map)
                    })
                    .collect(),
            ))
        }
    );

    // ── db.indexes ─────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.indexes",
        args: [],
        ret: Type::Any,
        procedure: ["label", "properties", "types", "options", "language", "stopwords", "entitytype", "status", "info"],
        fn db_indexes(runtime, _args) {
            Ok(Value::List(
                runtime
                    .g
                    .borrow()
                    .index_info()
                    .into_iter()
                    .map(
                        |IndexInfo {
                             label,
                             pending,
                             progress,
                             total,
                             fields,
                             language,
                             stopwords,
                         }| {
                            let mut map = OrderMap::default();
                            map.insert(Arc::new(String::from("label")), Value::String(label));
                            map.insert(
                                Arc::new(String::from("properties")),
                                Value::List(fields.keys().map(|f| Value::String(f.clone())).collect()),
                            );
                            let mut types_map = OrderMap::default();
                            for (attr, fields) in fields {
                                let mut types = thin_vec![];
                                for field in fields {
                                    match field.ty {
                                        IndexType::Range => {
                                            types.push(Value::String(Arc::new(String::from("RANGE"))));
                                        }
                                        IndexType::Fulltext => {
                                            types.push(Value::String(Arc::new(String::from("FULLTEXT"))));
                                        }
                                        IndexType::Vector => {
                                            types.push(Value::String(Arc::new(String::from("VECTOR"))));
                                        }
                                    }
                                }
                                types_map.insert(attr, Value::List(types));
                            }
                            map.insert(Arc::new(String::from("types")), Value::Map(types_map));
                            map.insert(Arc::new(String::from("options")), Value::Null);
                            map.insert(
                                Arc::new(String::from("language")),
                                language.map_or_else(|| Value::Null, Value::String),
                            );
                            map.insert(
                                Arc::new(String::from("stopwords")),
                                stopwords.map_or_else(
                                    || Value::Null,
                                    |sw| Value::List(sw.into_iter().map(Value::String).collect()),
                                ),
                            );
                            map.insert(
                                Arc::new(String::from("entitytype")),
                                Value::String(Arc::new(String::from("NODE"))),
                            );
                            map.insert(
                                Arc::new(String::from("status")),
                                if pending > 0 {
                                    Value::String(Arc::new(format!(
                                        "[Indexing] {progress}/{total}: UNDER CONSTRUCTION"
                                    )))
                                } else {
                                    Value::String(Arc::new(String::from("OPERATIONAL")))
                                },
                            );
                            map.insert(Arc::new(String::from("info")), Value::Null);

                            Value::Map(map)
                        },
                    )
                    .collect(),
            ))
        }
    );

    // ── db.idx.fulltext.createNodeIndex ────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.createNodeIndex",
        args: [Type::Map],
        ret: Type::Any,
        write procedure: [],
        fn db_fulltext_create_node_index(_runtime, _args) {
            Ok(Value::List(thin_vec![]))
        }
    );

    // ── db.idx.fulltext.queryNodes ─────────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.queryNodes",
        args: [Type::String, Type::String],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn db_fulltext_query_nodes(_, _) {
            Err(String::from("db.idx.fulltext.queryNodes() is not supported in this version"))
        }
    );
}
