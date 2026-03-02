//! Cypher query parsing.
//!
//! This module contains the Cypher parser, AST definitions, and supporting
//! utilities for converting query strings into abstract syntax trees.
//!
//! ## Key Components
//!
//! - [`ast`]: Abstract Syntax Tree node definitions
//! - [`cypher`]: Hand-written recursive descent Cypher parser
//! - [`string_escape`]: Cypher string escape sequence handling

pub mod ast;
pub mod cypher;
pub mod string_escape;
