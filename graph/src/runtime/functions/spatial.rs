//! Spatial and vector functions.
//!
//! Functions for working with geographic points and dense float vectors.
//!
//! ```text
//!  Cypher               Function    Returns     Notes
//! ──────────────────────────────────────────────────────────────
//!  vecf32(list)          vecf32()   VecF32      dense f32 vector from numeric list
//!  point({lat, lon})     point()    Point       validates lat/lon ranges
//!  distance(p1, p2)      distance() Float       Haversine great-circle distance (m)
//! ```
//!
//! `point()` reads `latitude` and `longitude` fields from a Map and
//! constructs a validated `Point`.  `distance()` delegates to
//! `Point::distance()` which computes the Haversine formula.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use super::{FnType, Functions, Type};
use crate::runtime::{
    runtime::Runtime,
    value::{Point, Value},
};
use thin_vec::ThinVec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "vecf32",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::VecF32, Type::Null]),
        fn vecf32(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::List(vec)) => {
                    for v in &vec {
                        if !matches!(v, Value::Int(_) | Value::Float(_)) {
                            return Err("vectorf32 expects an array of numbers".to_string());
                        }
                    }
                    Ok(Value::VecF32(
                        vec.into_iter().map(|v| v.get_numeric() as f32).collect(),
                    ))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "point",
        args: [Type::Union(vec![Type::Map, Type::Null])],
        ret: Type::Union(vec![Type::Point, Type::Null]),
        fn point(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let latitude = map
                        .get_str("latitude")
                        .ok_or_else(|| String::from("point() requires 'latitude' field"))?;
                    let latitude = match latitude {
                        Value::Float(f) => *f as f32,
                        Value::Int(i) => *i as f32,
                        _ => {
                            return Err(format!(
                                "Type mismatch: 'latitude' must be a number, got {}",
                                latitude.name()
                            ));
                        }
                    };
                    let longitude = map
                        .get_str("longitude")
                        .ok_or_else(|| String::from("point() requires 'longitude' field"))?;
                    let longitude = match longitude {
                        Value::Float(f) => *f as f32,
                        Value::Int(i) => *i as f32,
                        _ => {
                            return Err(format!(
                                "Type mismatch: 'longitude' must be a number, got {}",
                                longitude.name()
                            ));
                        }
                    };
                    let point = Point::new(latitude, longitude);
                    point.validate()?;
                    Ok(Value::Point(point))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "distance",
        args: [
            Type::Union(vec![Type::Point, Type::Null]),
            Type::Union(vec![Type::Point, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn distance(_, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(Value::Point(p1)), Some(Value::Point(p2))) => Ok(Value::Float(p1.distance(&p2))),
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );
}
