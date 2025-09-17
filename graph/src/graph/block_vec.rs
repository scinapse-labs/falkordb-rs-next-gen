use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::graph::{cow::Cow, matrix::Dup};

#[derive(Clone)]
struct Block<T> {
    vec: Arc<AtomicRefCell<Vec<Option<T>>>>,
}

impl<T: Clone> Block<T> {
    pub fn exists(
        &self,
        idx: usize,
    ) -> bool {
        self.vec.borrow().get(idx).is_some() && self.vec.borrow()[idx].is_some()
    }

    pub fn get(
        &self,
        idx: usize,
    ) -> Option<T> {
        self.vec.borrow().get(idx)?.clone()
    }

    pub fn remove(
        &mut self,
        idx: usize,
    ) -> Option<T> {
        self.vec.borrow_mut().get_mut(idx)?.take()
    }

    pub fn push_new(&mut self) {
        self.vec.borrow_mut().push(None);
    }

    pub fn len(&self) -> usize {
        self.vec.borrow().len()
    }

    pub fn push(
        &mut self,
        value: T,
    ) {
        self.vec.borrow_mut().push(Some(value));
    }

    pub fn insert(
        &mut self,
        idx: usize,
        value: T,
    ) -> bool {
        let has_value = self.vec.borrow()[idx].is_some();
        self.vec.borrow_mut()[idx] = Some(value);
        has_value
    }

    fn new(block_cap: usize) -> Self {
        Self {
            vec: Arc::new(AtomicRefCell::new(Vec::with_capacity(block_cap))),
        }
    }
}

impl<T: Default + Clone> Dup<Self> for Block<T> {
    fn dup(&self) -> Self {
        Self {
            vec: Arc::new(AtomicRefCell::new(self.vec.borrow().clone())),
        }
    }
}

pub struct BlockVec<T: Default + Clone> {
    segments: Vec<Cow<Block<T>>>,
    block_cap: usize,
}

impl<T: Default + Clone> BlockVec<T> {
    #[must_use]
    pub const fn new(block_cap: usize) -> Self {
        Self {
            segments: Vec::new(),
            block_cap,
        }
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        Self {
            segments: self.segments.iter().map(Cow::new_version).collect(),
            block_cap: self.block_cap,
        }
    }

    #[must_use]
    pub fn exists(
        &self,
        key: u64,
    ) -> bool {
        self.segments
            .get((key as usize) / self.block_cap)
            .is_some_and(|block| block.exists((key as usize) % self.block_cap))
    }

    #[must_use]
    pub fn get(
        &self,
        key: u64,
    ) -> Option<T> {
        self.segments
            .get((key as usize) / self.block_cap)?
            .get((key as usize) % self.block_cap)
    }

    pub fn insert(
        &mut self,
        key: u64,
        value: T,
    ) -> bool {
        while (key as usize) / self.block_cap >= self.segments.len() {
            self.segments.push(Cow::new(Block::new(self.block_cap)));
        }
        let block = &mut self.segments[(key as usize) / self.block_cap];
        while (key as usize) % self.block_cap >= block.len() {
            block.push_new();
        }
        block.insert((key as usize) % self.block_cap, value)
    }

    pub fn remove(
        &mut self,
        key: u64,
    ) -> Option<T> {
        self.segments
            .get_mut((key as usize) / self.block_cap)?
            .remove((key as usize) % self.block_cap)
    }
}
