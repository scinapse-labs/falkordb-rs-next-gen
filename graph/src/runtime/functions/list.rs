//! List operations.
//!
//! Functions that inspect or transform `List` values.  `Null` inputs
//! propagate as `Null` outputs.
//!
//! ```text
//!  Cypher                              Function                  Returns
//! ──────────────────────────────────────────────────────────────────────────
//!  size(x)                             size()                    Int
//!  head(list)                          head()                    first elem or Null
//!  last(list)                          last()                    last elem or Null
//!  tail(list)                          tail()                    list[1..]
//!  reverse(x)                          reverse()                 reversed list/str
//!  list.remove(list, idx [,count])     list_remove()             list without elements
//!  list.sort(list [,ascending?])       list_sort()               sorted list
//!  list.insert(list, idx, val [,dup?]) list_insert()             list with val inserted
//!  list.insertListElements(l1,l2,i,..) list_insert_list_elements() merged list
//!  list.dedup(list)                    list_dedup()              deduplicated list
//! ```
//!
//! The `list.*` functions follow the FalkorDB extension namespace.
//! `list.insert` and `list.insertListElements` accept an optional
//! `allowDuplicate` boolean (default `true`).  Negative indices are
//! normalised relative to the list length.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_possible_wrap)]

use super::{FnType, Functions, Type};
use crate::runtime::{
    runtime::Runtime,
    value::{CompareValue, Value},
};
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "size",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::String,
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn size(_, args) {
            match args.into_iter().next() {
                Some(Value::String(s)) => Ok(Value::Int(s.chars().count() as i64)),
                Some(Value::List(v)) => Ok(Value::Int(v.len() as i64)),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "head",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::Any,
        fn head(_, args) {
            match args.into_iter().next() {
                Some(Value::List(v)) => {
                    if v.is_empty() {
                        Ok(Value::Null)
                    } else {
                        Ok(v[0].clone())
                    }
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "last",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::Any,
        fn last(_, args) {
            match args.into_iter().next() {
                Some(Value::List(v)) => Ok(v.last().cloned().unwrap_or(Value::Null)),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "tail",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::Any,
        fn tail(_, args) {
            match args.into_iter().next() {
                Some(Value::List(v)) => {
                    if v.is_empty() {
                        Ok(Value::List(Arc::new(thin_vec![])))
                    } else {
                        Ok(Value::List(Arc::new(v[1..].iter().cloned().collect::<ThinVec<_>>())))
                    }
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "reverse",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::String,
            Type::Null,
        ])],
        ret: Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::String,
            Type::Null,
        ]),
        fn reverse(_, args) {
            match args.into_iter().next() {
                Some(Value::List(mut v)) => {
                    Arc::make_mut(&mut v).reverse();
                    Ok(Value::List(v))
                }
                Some(Value::String(s)) => Ok(Value::String(Arc::new(s.chars().rev().collect()))),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    // list.remove(list, index, count?)
    // Removes `count` (default 1) elements starting at `index`.
    cypher_fn!(funcs, "list.remove",
        args: [
            Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Int,
            Type::Optional(Box::new(Type::Int)),
        ],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn list_remove(_, args) {
            let mut it = args.into_iter();
            let list = it.next();
            let index = it.next();
            let count = it.next();

            match list {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::List(vs)) => {
                    let Some(Value::Int(idx)) = index else { return Ok(Value::Null) };
                    let count = match count {
                        Some(Value::Int(c)) => c,
                        None => 1,
                        _ => return Ok(Value::Null),
                    };
                    let len = vs.len() as i64;
                    // Normalize negative index
                    let normalized = if idx < 0 { len + idx } else { idx };
                    // Out of range or non-positive count: return original
                    if normalized < 0 || normalized >= len || count <= 0 {
                        return Ok(Value::List(vs));
                    }
                    let start = normalized as usize;
                    let end = ((normalized + count) as usize).min(vs.len());
                    let mut result = ThinVec::with_capacity(vs.len() - (end - start));
                    result.extend_from_slice(&vs[..start]);
                    result.extend_from_slice(&vs[end..]);
                    Ok(Value::List(Arc::new(result)))
                }
                _ => unreachable!(),
            }
        }
    );

    // list.sort(list, ascending?)
    // Sorts elements. Default ascending=true.
    cypher_fn!(funcs, "list.sort",
        args: [
            Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Optional(Box::new(Type::Bool)),
        ],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn list_sort(_, args) {
            let mut it = args.into_iter();
            let list = it.next();
            let ascending = it.next();

            match list {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::List(vs)) => {
                    let asc = match ascending {
                        Some(Value::Bool(b)) => b,
                        None => true,
                        _ => return Ok(Value::Null),
                    };
                    let mut sorted: ThinVec<Value> = vs.iter().cloned().collect();
                    sorted.sort_by(|a, b| a.compare_value(b).0);
                    if !asc {
                        sorted.reverse();
                    }
                    Ok(Value::List(Arc::new(sorted)))
                }
                _ => unreachable!(),
            }
        }
    );

    // list.insert(list, index, value, allowDuplicate?)
    // Inserts a single value at index. Default allowDuplicate=true.
    cypher_fn!(funcs, "list.insert",
        args: [
            Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Int,
            Type::Any,
            Type::Optional(Box::new(Type::Bool)),
        ],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn list_insert(_, args) {
            let mut it = args.into_iter();
            let list = it.next();
            let index = it.next();
            let value = it.next();
            let allow_dup = it.next();

            match list {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::List(vs)) => {
                    let Some(Value::Int(idx)) = index else { return Ok(Value::Null) };
                    let val = match value {
                        Some(Value::Null) | None => return Ok(Value::List(vs)),
                        Some(v) => v,
                    };
                    let allow_dup = match allow_dup {
                        Some(Value::Bool(b)) => b,
                        None => true,
                        _ => return Ok(Value::Null),
                    };
                    let len = vs.len() as i64;
                    // Normalize negative index: -1 means after last
                    let normalized = if idx < 0 { len + idx + 1 } else { idx };
                    // Out of range: return original
                    if normalized < 0 || normalized > len {
                        return Ok(Value::List(vs));
                    }
                    if !allow_dup && vs.contains(&val) {
                        return Ok(Value::List(vs));
                    }
                    let pos = normalized as usize;
                    let mut result = ThinVec::with_capacity(vs.len() + 1);
                    result.extend_from_slice(&vs[..pos]);
                    result.push(val);
                    result.extend_from_slice(&vs[pos..]);
                    Ok(Value::List(Arc::new(result)))
                }
                _ => unreachable!(),
            }
        }
    );

    // list.insertListElements(list, list2, index, allowDuplicate?)
    // Inserts all elements of list2 at index. Default allowDuplicate=true.
    cypher_fn!(funcs, "list.insertListElements",
        args: [
            Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Int,
            Type::Optional(Box::new(Type::Bool)),
        ],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn list_insert_list_elements(_, args) {
            let mut it = args.into_iter();
            let list = it.next();
            let list2 = it.next();
            let index = it.next();
            let allow_dup = it.next();

            match list {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::List(vs)) => {
                    let vals = match list2 {
                        Some(Value::Null) | None => return Ok(Value::List(vs)),
                        Some(Value::List(v)) => v,
                        _ => unreachable!(),
                    };
                    let Some(Value::Int(idx)) = index else { return Ok(Value::Null) };
                    let allow_dup = match allow_dup {
                        Some(Value::Bool(b)) => b,
                        None => true,
                        _ => return Ok(Value::Null),
                    };
                    let len = vs.len() as i64;
                    let normalized = if idx < 0 { len + idx + 1 } else { idx };
                    if normalized < 0 || normalized > len {
                        return Ok(Value::List(vs));
                    }
                    let pos = normalized as usize;
                    // Filter for duplicates if needed
                    let to_insert: ThinVec<Value> = if allow_dup {
                        vals.iter().cloned().collect()
                    } else {
                        let mut seen: ThinVec<Value> = vs.iter().cloned().collect();
                        let mut filtered = ThinVec::new();
                        for v in vals.iter() {
                            if !seen.contains(v) {
                                seen.push(v.clone());
                                filtered.push(v.clone());
                            }
                        }
                        filtered
                    };
                    let mut result = ThinVec::with_capacity(vs.len() + to_insert.len());
                    result.extend_from_slice(&vs[..pos]);
                    result.extend(to_insert);
                    result.extend_from_slice(&vs[pos..]);
                    Ok(Value::List(Arc::new(result)))
                }
                _ => unreachable!(),
            }
        }
    );

    // list.dedup(list)
    // Removes duplicate elements, preserving order.
    cypher_fn!(funcs, "list.dedup",
        args: [Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn list_dedup(_, args) {
            match args.into_iter().next() {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::List(vs)) => {
                    let mut seen = ThinVec::<Value>::new();
                    for v in vs.iter() {
                        if !seen.contains(v) {
                            seen.push(v.clone());
                        }
                    }
                    Ok(Value::List(Arc::new(seen)))
                }
                _ => unreachable!(),
            }
        }
    );
}
