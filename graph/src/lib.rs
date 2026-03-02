#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::type_complexity)]
#![allow(clippy::module_inception)]

//! # FalkorDB Graph Engine
//!
//! This crate contains the core graph database engine for FalkorDB.
//! It provides Cypher query parsing, planning, optimization, and execution
//! over a sparse matrix-based graph representation using GraphBLAS.
//!
//! ## Query Processing Pipeline
//!
//! ```text
//! Cypher Query String
//!        │
//!        ▼
//! ┌─────────────┐
//! │   parser    │  Parse query into AST (hand-written recursive descent)
//! └─────────────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  planner    │  Bind, plan, and optimize the execution plan (IR)
//! └─────────────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  runtime    │  Execute plan against the graph
//! └─────────────┘
//! ```
//!
//! ## Module Overview
//!
//! - [`parser`]: Cypher parser, AST definitions, and string escape utilities
//! - [`planner`]: Semantic binding, logical plan generation, and optimization
//! - [`graph`]: Graph data structures (sparse matrices, vectors, MVCC)
//! - [`index`]: Index types and management with RediSearch
//! - [`redisearch`]: RediSearch FFI bindings
//! - [`runtime`]: Query execution engine and built-in functions
//! - [`threadpool`]: Thread pool for parallel query execution

pub mod entity_type;
pub mod graph;
pub mod index;
pub mod parser;
pub mod planner;
pub mod redisearch;
pub mod runtime;
pub mod threadpool;
