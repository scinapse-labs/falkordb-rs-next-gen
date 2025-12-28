use std::{hash::Hash, ops::Index};

use thin_vec::{ThinVec, thin_vec};

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
            if k == key {
                return Some(v);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.vec.iter().map(|(k, v)| (k, v))
    }

    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.vec.into_iter()
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
