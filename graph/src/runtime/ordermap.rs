//! Insertion-ordered map preserving key order.
//!
//! This module provides [`OrderMap`], a map that maintains insertion order
//! for iteration. Used for:
//!
//! - Property maps on nodes/relationships (consistent key ordering)
//! - Query result maps (deterministic output)
//! - Any context where iteration order must match insertion order
//!
//! ## Implementation
//!
//! Uses a `ThinVec<(K, V)>` internally with O(n) lookup. This is efficient
//! for small maps (typical property counts) while preserving order.

use std::{borrow::Borrow, hash::Hash, ops::Index};

use thin_vec::{ThinVec, thin_vec};

/// A map that preserves insertion order during iteration.
///
/// Keys are compared by equality (`PartialEq`). For small maps (< ~20 keys),
/// this is faster than hash-based lookup due to cache locality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderMap<K, V> {
    vec: ThinVec<(K, V)>,
}

impl<K, V> Default for OrderMap<K, V> {
    fn default() -> Self {
        Self {
            vec: ThinVec::new(),
        }
    }
}

impl<K: PartialEq, V> OrderMap<K, V> {
    #[must_use]
    pub fn from_vec(vec: ThinVec<(K, V)>) -> Self {
        let mut res = Self { vec: thin_vec![] };
        for (k, v) in vec {
            res.insert(k, v);
        }
        res
    }

    pub fn reserve_exact(
        &mut self,
        additional: usize,
    ) {
        self.vec.reserve_exact(additional);
    }

    pub fn insert(
        &mut self,
        key: K,
        value: V,
    ) -> Option<V> {
        for (k, v) in &mut self.vec {
            if *k == key {
                let old = std::mem::replace(v, value);
                return Some(old);
            }
        }
        self.vec.push((key, value));
        None
    }

    pub fn remove(
        &mut self,
        key: &K,
    ) -> Option<V> {
        if let Some(pos) = self.vec.iter().position(|(k, _)| k == key) {
            Some(self.vec.remove(pos).1)
        } else {
            None
        }
    }

    pub fn get(
        &self,
        key: &K,
    ) -> Option<&V> {
        for (k, v) in &self.vec {
            if k.borrow() == key {
                return Some(v);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.vec.iter().map(|(k, v)| (k, v))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.vec.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.vec.iter().map(|(_, v)| v)
    }
}

// Specialized implementation for Arc<String> keys
// Allows lookup with &str, &String, or &Arc<String> - zero allocations!
impl<V> OrderMap<std::sync::Arc<String>, V> {
    pub fn get_str<Q>(
        &self,
        key: &Q,
    ) -> Option<&V>
    where
        Q: AsRef<str> + ?Sized,
    {
        let key_str = key.as_ref();
        for (k, v) in &self.vec {
            if k.as_ref() == key_str {
                return Some(v);
            }
        }
        None
    }
}

impl<K: Hash, V: Hash> Hash for OrderMap<K, V> {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        self.vec.hash(state);
    }
}

impl<K: PartialEq, V: PartialEq> FromIterator<(K, V)> for OrderMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<K: PartialEq, V: PartialEq> Index<&K> for OrderMap<K, V> {
    type Output = V;

    fn index(
        &self,
        index: &K,
    ) -> &Self::Output {
        self.get(index).expect("no entry found for key")
    }
}

impl<K, V> IntoIterator for OrderMap<K, V> {
    type Item = (K, V);
    type IntoIter = thin_vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}
