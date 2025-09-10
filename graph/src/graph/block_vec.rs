use crate::graph::{cow::Cow, matrix::Dup};

#[derive(Clone)]
struct Block<T> {
    vec: Vec<Option<T>>,
}

impl<T> Block<T> {
    pub fn get(
        &self,
        idx: usize,
    ) -> Option<&T> {
        self.vec.get(idx)?.as_ref()
    }

    pub fn get_mut(
        &mut self,
        idx: usize,
    ) -> Option<&mut Option<T>> {
        self.vec.get_mut(idx)
    }

    pub fn idx_mut(
        &mut self,
        idx: usize,
    ) -> &mut Option<T> {
        &mut self.vec[idx]
    }

    pub fn remove(
        &mut self,
        idx: usize,
    ) -> Option<T> {
        self.vec.get_mut(idx)?.take()
    }

    pub fn push_new(&mut self) {
        self.vec.push(None);
    }

    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn push(
        &mut self,
        value: T,
    ) {
        self.vec.push(Some(value));
    }

    fn new(block_cap: usize) -> Self {
        Self {
            vec: Vec::with_capacity(block_cap),
        }
    }
}

impl<T: Default + Clone> Dup<Self> for Block<T> {
    fn dup(&self) -> Self {
        Self {
            vec: self.vec.clone(),
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
            segments: self.segments.clone(),
            block_cap: self.block_cap,
        }
    }

    #[must_use]
    pub fn get(
        &self,
        key: u64,
    ) -> Option<&T> {
        self.segments
            .get((key as usize) / self.block_cap)?
            .get((key as usize) % self.block_cap)
    }

    pub fn get_mut(
        &mut self,
        key: u64,
    ) -> Option<&mut T> {
        self.segments
            .get_mut((key as usize) / self.block_cap)?
            .get_mut((key as usize) % self.block_cap)?
            .as_mut()
    }

    pub fn insert(
        &mut self,
        key: u64,
    ) -> &mut Option<T> {
        while (key as usize) / self.block_cap >= self.segments.len() {
            self.segments.push(Cow::new(Block::new(self.block_cap)));
        }
        let block = &mut self.segments[(key as usize) / self.block_cap];
        while (key as usize) % self.block_cap >= block.len() {
            block.push_new();
        }
        block.idx_mut((key as usize) % self.block_cap)
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
