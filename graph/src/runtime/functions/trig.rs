//! Trigonometric functions.
//!
//! Standard trigonometric operations following IEEE 754 semantics.
//! Inputs are `Int` or `Float`; all outputs are `Float`.  `Null`
//! propagates unchanged.
//!
//! ```text
//!  Cypher          Function     Notes
//! ──────────────────────────────────────────────
//!  sin(x)          sin()        radians in
//!  cos(x)          cos()        radians in
//!  tan(x)          tan()        radians in
//!  cot(x)          cot()        cos/sin
//!  asin(x)         asin()       radians out
//!  acos(x)         acos()       radians out
//!  atan(x)         atan()       radians out
//!  atan2(y, x)     atan2()      two-argument
//!  degrees(x)      degrees()    radians -> degrees
//!  radians(x)      radians()    degrees -> radians
//!  pi()            pi()         constant (3.14159...)
//!  haversin(x)     haversin()   (1 - cos(x)) / 2
//! ```

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};
use thin_vec::ThinVec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "sin",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn sin(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).sin())),
                Some(Value::Float(f)) => Ok(Value::Float(f.sin())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "cos",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn cos(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).cos())),
                Some(Value::Float(f)) => Ok(Value::Float(f.cos())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "tan",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn tan(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).tan())),
                Some(Value::Float(f)) => Ok(Value::Float(f.tan())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "cot",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn cot(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => {
                    let val = n as f64;
                    Ok(Value::Float(val.cos() / val.sin()))
                }
                Some(Value::Float(f)) => Ok(Value::Float(f.cos() / f.sin())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "asin",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn asin(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).asin())),
                Some(Value::Float(f)) => Ok(Value::Float(f.asin())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "acos",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn acos(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).acos())),
                Some(Value::Float(f)) => Ok(Value::Float(f.acos())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "atan",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn atan(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).atan())),
                Some(Value::Float(f)) => Ok(Value::Float(f.atan())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "atan2",
        args: [
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn atan2(_, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(Value::Int(y)), Some(Value::Int(x))) => Ok(Value::Float((y as f64).atan2(x as f64))),
                (Some(Value::Float(y)), Some(Value::Float(x))) => Ok(Value::Float(y.atan2(x))),
                (Some(Value::Int(y)), Some(Value::Float(x))) => Ok(Value::Float((y as f64).atan2(x))),
                (Some(Value::Float(y)), Some(Value::Int(x))) => Ok(Value::Float(y.atan2(x as f64))),
                (Some(Value::Null), Some(_)) | (Some(_), Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "degrees",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn degrees(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).to_degrees())),
                Some(Value::Float(f)) => Ok(Value::Float(f.to_degrees())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "radians",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn radians(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).to_radians())),
                Some(Value::Float(f)) => Ok(Value::Float(f.to_radians())),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "pi",
        args: [],
        ret: Type::Float,
        fn pi(_, args) {
            debug_assert!(args.is_empty());
            Ok(Value::Float(std::f64::consts::PI))
        }
    );

    cypher_fn!(funcs, "haversin",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn haversin(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => {
                    let val = n as f64;
                    Ok(Value::Float((1.0 - val.cos()) / 2.0))
                }
                Some(Value::Float(f)) => Ok(Value::Float((1.0 - f.cos()) / 2.0)),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );
}
