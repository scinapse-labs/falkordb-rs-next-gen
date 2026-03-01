#[derive(Default, Clone)]
pub struct BitSet(Vec<u64>);

impl BitSet {
    pub fn set(
        &mut self,
        bit: usize,
    ) {
        let word = bit / 64;
        if word >= self.0.len() {
            self.0.resize(word + 1, 0);
        }
        self.0[word] |= 1u64 << (bit % 64);
    }

    pub fn test(
        &self,
        bit: usize,
    ) -> bool {
        let word = bit / 64;
        word < self.0.len() && self.0[word] & (1u64 << (bit % 64)) != 0
    }

    pub fn union(
        &mut self,
        other: &Self,
    ) {
        if other.0.len() > self.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= b;
        }
    }
}
