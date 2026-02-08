//! Copy-on-Write wrapper for efficient MVCC versioning.
//!
//! This module provides [`Cow`] (Copy-on-Write), a wrapper that defers
//! duplication of data until mutation is needed. This enables efficient
//! MVCC by sharing immutable data between versions.
//!
//! ## How It Works
//!
//! ```text
//! Version 1: Cow { inner: Matrix, dup: false }
//!                     │
//!                     └─ (actual matrix data)
//!
//! new_version():
//! Version 2: Cow { inner: Matrix, dup: true }  ← shares same data
//!                     │
//!                     └─ (points to same matrix)
//!
//! deref_mut() on Version 2:
//! Version 2: Cow { inner: Matrix', dup: false }  ← now has own copy
//! ```
//!
//! ## Good Practice: Lazy Duplication
//!
//! By deferring duplication until write, read-only transactions have
//! zero copying overhead. Only the first mutation triggers a deep copy.

use std::ops::{Deref, DerefMut};

use crate::graph::matrix::Dup;

/// Copy-on-Write wrapper that defers duplication until mutation.
///
/// Wraps any type implementing `Dup` and `Clone`. The inner value is
/// shared until `deref_mut` is called, at which point it's duplicated.
#[derive(Clone)]
pub struct Cow<T: Dup<T> + Clone> {
    inner: T,
    /// If true, the next mutable access will duplicate the inner value
    dup: bool,
}

impl<T: Dup<T> + Clone> Cow<T> {
    pub const fn new(inner: T) -> Self {
        Self { inner, dup: false }
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dup: true,
        }
    }
}

impl<T: Dup<T> + Clone> Deref for Cow<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Dup<T> + Clone> DerefMut for Cow<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.dup {
            self.inner = self.inner.dup();
            self.dup = false;
        }
        &mut self.inner
    }
}
