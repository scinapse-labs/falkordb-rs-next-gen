use std::ops::{Deref, DerefMut};

use crate::graph::matrix::Dup;

#[derive(Clone)]
pub struct Cow<T: Dup<T> + Clone> {
    inner: T,
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
