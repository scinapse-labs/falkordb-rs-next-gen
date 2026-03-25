//! Cypher parser helper macros.
//!
//! This module defines internal parser macros used by
//! [`crate::parser::cypher::Parser`] to reduce repetitive token-matching and
//! expression-stack handling logic.
//!
//! Macros in this module:
//! - `match_token!`: mandatory token/keyword match with contextual error message
//! - `optional_match_token!`: optional token/keyword match
//! - `parse_expr_return!`: expression-stack return helper
//! - `parse_operators!`: precedence-aware binary operator folding helper

macro_rules! match_token {
    ($lexer:expr, $token:ident) => {
        match $lexer.current()? {
            Token::$token => {
                $lexer.next();
            }
            _ => {
                return Err($lexer.format_error(&format!(
                    // Return the display in error
                    "Invalid input '{}': expected {}",
                    $lexer.current_str(),
                    Token::$token
                )));
            }
        }
    };
    ($lexer:expr => $token:ident) => {
        match $lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::$token),
                ..
            } => {
                $lexer.next();
            }
            _ => {
                return Err($lexer.format_error(&format!(
                    "Invalid input '{}': expected {:?}",
                    $lexer.current_str(),
                    Keyword::$token
                )));
            }
        }
    };
    () => {};
}

macro_rules! optional_match_token {
    ($lexer:expr, $token:ident) => {
        match $lexer.current()? {
            Token::$token => {
                $lexer.next();
                true
            }
            _ => false,
        }
    };
    ($lexer:expr => $token:ident) => {
        match $lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::$token),
                ..
            } => {
                $lexer.next();
                true
            }
            _ => false,
        }
    };
    () => {};
}

#[macro_export]
macro_rules! parse_expr_return {
    ($stack:ident, $res:ident) => {
        match &mut $stack.last_mut() {
            Some((_, Some(expr))) => {
                expr.root_mut().push_child_tree($res);
            }
            Some((_, expr)) => {
                *expr = Some($res);
            }
            _ => return Ok($res),
        }
    };
}

#[macro_export]
macro_rules! parse_operators {
    ($self:ident, $stack:ident, $res:ident, $current:ident, $token:pat => $expr:ident) => {
        if let $token = $self.lexer.current()? {
            $self.lexer.next();
            let res = if matches!($res.root().data(), ExprIR::$expr) {
                $res
            } else {
                tree!(ExprIR::$expr, $res)
            };
            $stack.push(($current, Some(res)));
            $stack.push(($current + 1, None));
        } else {
            match &mut $stack.last_mut() {
                Some((_, Some(expr))) => {
                    expr.root_mut().push_child_tree($res);
                }
                Some((_, expr)) => {
                    *expr = Some($res);
                }
                _ => return Ok($res),
            }
        }
    };
    ($self:ident, $stack:ident, $res:ident, $current:ident, $($token:pat => $expr:ident),*) => {
        let mut res = $res;
        $(if let $token = $self.lexer.current()? {
            $self.lexer.next();
            if matches!(res.root().data(), ExprIR::$expr) {
            } else {
                res = tree!(ExprIR::$expr, res);
            };
            $stack.push(($current, Some(res)));
            $stack.push(($current + 1, None));
            continue;
        })*

        match &mut $stack.last_mut() {
            Some((_, Some(expr))) => {
                expr.root_mut().push_child_tree(res);
            }
            Some((_, expr)) => {
                *expr = Some(res);
            }
            _ => return Ok(res),
        }
    };
}
