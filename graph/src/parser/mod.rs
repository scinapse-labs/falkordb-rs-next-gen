//! Cypher query parsing.
//!
//! This module contains the Cypher parser, AST definitions, and supporting
//! utilities for converting query strings into abstract syntax trees.
//!
//! ## Key Components
//!
//! - [`ast`]: Abstract Syntax Tree node definitions
//! - [`lexer`]: Cypher lexical analysis and tokenization
//! - [`cypher`]: Hand-written recursive descent Cypher parser
//! - [`r#macro`]: parser helper macros
//! - [`string_escape`]: Cypher string escape sequence handling

pub mod ast;
#[macro_use]
pub mod r#macro;
pub mod cypher;
pub mod lexer;
pub mod string_escape;
