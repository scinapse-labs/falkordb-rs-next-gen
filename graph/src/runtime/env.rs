use std::hash::Hash;

use crate::parser::ast::Variable;
use crate::runtime::bitset::BitSet;
use crate::runtime::value::Value;

#[derive(Default)]
pub struct Env {
    values: Vec<Value>,
    bound: BitSet,
}

impl Env {
    pub fn insert(
        &mut self,
        key: &Variable,
        value: Value,
    ) {
        while self.values.len() <= key.id as _ {
            self.values.push(Value::Null);
        }
        self.values[key.id as usize] = value;
        self.bound.set(key.id as usize);
    }

    /// Returns true if the variable was explicitly inserted (even if set to Null).
    /// Returns false for padding Null slots that were never explicitly set.
    #[must_use]
    pub fn is_bound(
        &self,
        key: &Variable,
    ) -> bool {
        self.bound.test(key.id as usize)
    }

    #[must_use]
    pub fn get(
        &self,
        key: &Variable,
    ) -> Option<&Value> {
        self.values.get(key.id as usize)
    }

    /// Takes ownership of a value from the environment, replacing it with `Null`.
    ///
    /// This method is designed for aggregation optimizations where we need to transfer
    /// ownership of large accumulated values (like lists with millions of items) without
    /// cloning them. By replacing the environment entry with `Null`, we ensure the value
    /// can be moved (not cloned) to the aggregation function.
    ///
    /// # Returns
    /// - `Some(value)` if the key exists and contains a non-Null value
    /// - `None` if the key doesn't exist or already contains `Null`
    ///
    /// # Usage
    /// Prefer this over `get()` when:
    /// - You need exclusive ownership of a value
    /// - The value is expensive to clone (e.g., large collections)
    /// - The environment slot won't be read again before being overwritten
    pub fn take(
        &mut self,
        key: &Variable,
    ) -> Option<Value> {
        self.values.get_mut(key.id as usize).and_then(|value| {
            match std::mem::replace(value, Value::Null) {
                Value::Null => None,
                v => Some(v),
            }
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn merge(
        &mut self,
        other: Self,
    ) {
        while self.values.len() < other.values.len() {
            self.values.push(Value::Null);
        }
        for (key, value) in other.values.into_iter().enumerate() {
            if value == Value::Null {
                continue;
            }
            self.values[key] = value;
        }
        self.bound.union(&other.bound);
    }
}

impl AsRef<Vec<Value>> for Env {
    fn as_ref(&self) -> &Vec<Value> {
        &self.values
    }
}

impl Hash for Env {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        for (key, value) in self.values.iter().enumerate() {
            if *value == Value::Null {
                continue;
            }
            key.hash(state);
            value.hash(state);
        }
    }
}

impl Clone for Env {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            bound: self.bound.clone(),
        }
    }
}
