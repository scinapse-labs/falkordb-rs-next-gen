use crate::runtime::value::Value;
use std::collections::HashMap;
use std::hash::Hash;

/// Wrapper around usize to create a hashable, comparable expression ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(usize);

impl From<usize> for ExprId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

/// Context for stateful expression evaluation
/// Maintains state for functions like `prev()` that need to remember values
/// across multiple invocations within a single query execution
#[derive(Debug, Default)]
pub struct ExpressionContext {
    /// State storage keyed by expression node ID
    /// Each stateful function expression gets a unique ID
    stateful_state: HashMap<ExprId, Value>,
}

impl ExpressionContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            stateful_state: HashMap::new(),
        }
    }

    /// Get the current state for a stateful function expression
    #[must_use]
    pub fn get_state(
        &self,
        expr_id: ExprId,
    ) -> Option<&Value> {
        self.stateful_state.get(&expr_id)
    }

    /// Update the state for a stateful function expression
    pub fn set_state(
        &mut self,
        expr_id: ExprId,
        value: Value,
    ) {
        self.stateful_state.insert(expr_id, value);
    }

    /// Clear all state (called when starting a new query)
    pub fn clear(&mut self) {
        self.stateful_state.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_context_basic() {
        let mut ctx = ExpressionContext::new();

        // Initially no state
        assert!(ctx.get_state(ExprId(0)).is_none());

        // Set state
        ctx.set_state(ExprId(0), Value::Int(42));
        assert_eq!(ctx.get_state(ExprId(0)), Some(&Value::Int(42)));

        // Update state
        ctx.set_state(ExprId(0), Value::Int(100));
        assert_eq!(ctx.get_state(ExprId(0)), Some(&Value::Int(100)));

        // Clear all state
        ctx.clear();
        assert!(ctx.get_state(ExprId(0)).is_none());
    }

    #[test]
    fn test_multiple_expressions() {
        let mut ctx = ExpressionContext::new();

        ctx.set_state(ExprId(0), Value::Int(1));
        ctx.set_state(ExprId(1), Value::Int(2));
        ctx.set_state(ExprId(2), Value::Int(3));

        assert_eq!(ctx.get_state(ExprId(0)), Some(&Value::Int(1)));
        assert_eq!(ctx.get_state(ExprId(1)), Some(&Value::Int(2)));
        assert_eq!(ctx.get_state(ExprId(2)), Some(&Value::Int(3)));
    }
}
