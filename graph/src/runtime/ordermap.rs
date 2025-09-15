use std::{hash::Hash, ops::Index};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderMap<K, V> {
    vec: Vec<(K, V)>,
}

impl<K: PartialEq, V: PartialEq> OrderMap<K, V> {
    #[must_use]
    pub const fn from_vec(vec: Vec<(K, V)>) -> Self {
        Self { vec }
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
            if k == key {
                return Some(v);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.vec.iter().map(|(k, v)| (k, v))
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.vec.iter().map(|(k, _)| k)
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
