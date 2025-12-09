use crate::graph::{
    GraphBLAS::GxB_Print_Level,
    cow::Cow,
    matrix::{
        self, Descriptor, Dup, Get, MaskedElementWiseAdd, MaskedElementWiseMultiply, Matrix, New,
        Remove, Set, Size, Transpose,
    },
};

pub struct VersionedMatrix {
    m: Cow<Matrix>,
    dp: Cow<Matrix>,
    dm: Cow<Matrix>,
}

unsafe impl Send for VersionedMatrix {}
unsafe impl Sync for VersionedMatrix {}

impl VersionedMatrix {
    pub fn flush(&mut self) {
        self.wait();
        if self.dp.nvals() >= 10000 {
            self.m.element_wise_add(None, None, Some(&self.dp), None);
            self.dp.clear();
        }
        if self.dm.nvals() >= 10000 {
            self.m.remove_all(&self.dm);
            self.dm.clear();
        }
    }

    pub fn wait(&self) {
        debug_assert!(!self.m.pending());
        self.dp.wait();
        self.dm.wait();
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage() + self.dp.memory_usage() + self.dm.memory_usage()
    }
}

impl Size for VersionedMatrix {
    fn nrows(&self) -> u64 {
        self.m.nrows()
    }

    fn ncols(&self) -> u64 {
        self.m.ncols()
    }

    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.wait();
        self.m.resize(nrows, ncols);
        self.dp.resize(nrows, ncols);
        self.dm.resize(nrows, ncols);
    }

    fn nvals(&self) -> u64 {
        self.wait();
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }
}

impl New for VersionedMatrix {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::new(nrows, ncols)),
            dp: Cow::new(Matrix::new(nrows, ncols)),
            dm: Cow::new(Matrix::new(nrows, ncols)),
        }
    }
}

impl Dup<Self> for VersionedMatrix {
    fn dup(&self) -> Self {
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
        }
    }
}

impl VersionedMatrix {
    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        self.wait();
        Iter::new(self, min_row, max_row)
    }

    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        // TODO: remove
        self.wait();
        let mut m = self.m.dup();
        m.remove_all(&self.dm);
        m.element_wise_add(None, None, Some(&self.dp), None);
        m
    }

    pub fn print(
        &self,
        level: GxB_Print_Level,
    ) {
        self.m.print(level);
        self.dp.print(level);
        self.dm.print(level);
    }
}

impl Remove for VersionedMatrix {
    fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.set(i, j, true);
        } else {
            self.dp.remove(i, j);
        }
    }

    fn remove_all(
        &mut self,
        b: &Matrix,
    ) {
        self.wait();
        self.dp.remove_all(b);
        self.dm.element_wise_add(Some(&self.m), None, Some(b), None);
    }
}

pub trait ElementWiseAdd {
    fn element_wise_add(
        &mut self,
        b: &Self,
    );
}

impl ElementWiseAdd for VersionedMatrix {
    fn element_wise_add(
        &mut self,
        b: &Self,
    ) {
        // TODO: fix
        self.wait();
        self.dp.element_wise_add(None, None, Some(&b.m), None);
        self.dp.element_wise_add(None, None, Some(&b.dp), None);
        self.dm.element_wise_add(None, None, Some(&b.dm), None);
    }
}

pub trait ElementWiseMultiply {
    fn element_wise_multiply(
        &mut self,
        b: &Self,
    );
}

impl ElementWiseMultiply for VersionedMatrix {
    fn element_wise_multiply(
        &mut self,
        b: &Self,
    ) {
        // TODO: fix
        debug_assert_eq!(self.dp.nvals(), 0);
        debug_assert_eq!(self.dm.nvals(), 0);
        self.wait();
        self.dp.element_wise_multiply(
            Some(&self.dm),
            Some(&self.m),
            Some(&b.dp),
            Some(Descriptor::C),
        );
        // self.dm.element_wise_multiply(None, &self.dm, &b.dp, None);
    }
}

// impl MxM<bool> for VersionedMatrix<bool> {
//     fn lmxm(
//         &mut self,
//         b: &Self,
//     ) {

//     }

//     fn rmxm(
//         &mut self,
//         b: &Self,
//     ) {

//     }
// }

impl Get for VersionedMatrix {
    fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool> {
        self.wait();
        self.m.get(i, j).map_or_else(
            || self.dp.get(i, j),
            |value| {
                if self.dm.get(i, j).is_some() {
                    None
                } else {
                    Some(value)
                }
            },
        )
    }
}

impl Set for VersionedMatrix {
    fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        debug_assert!(!self.m.pending());
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.remove(i, j);
        } else {
            debug_assert!(self.dm.get(i, j).is_none());
            self.dp.set(i, j, value);
        }
    }
}

pub trait SetAll {
    fn set_all(
        &mut self,
        b: &Matrix,
    );
}

impl SetAll for VersionedMatrix {
    fn set_all(
        &mut self,
        b: &Matrix,
    ) {
        self.wait();
        self.dp
            .element_wise_add(Some(&self.m), None, Some(b), Some(Descriptor::C));
        self.dm.remove_all(b);
    }
}

impl Transpose for VersionedMatrix
where
    Self: New,
{
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    fn transpose(&self) -> Self {
        Self {
            m: Cow::new(self.m.transpose()),
            dp: Cow::new(self.dp.transpose()),
            dm: Cow::new(self.dm.transpose()),
        }
    }
}

pub struct Iter {
    mit: matrix::Iter,
    dpit: matrix::Iter,
    dm: Cow<Matrix>,
}

unsafe impl Send for Iter {}
unsafe impl Sync for Iter {}

impl Iter {
    /// Creates a new iterator for traversing all elements in a matrix.
    ///
    /// # Parameters
    /// - `m`: The matrix to iterate over.
    /// - `min_row`: The minimum row index to start iterating from.
    /// - `max_row`: The maximum row index to stop iterating at.
    #[must_use]
    pub fn new(
        m: &VersionedMatrix,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        Self {
            mit: m.m.iter(min_row, max_row),
            dpit: m.dp.iter(min_row, max_row),
            dm: m.dm.clone(),
        }
    }
}

impl Iterator for Iter {
    type Item = (u64, u64);

    /// Advances the iterator and returns the next element in the matrix.
    ///
    /// # Returns
    /// - `Some((u64, u64))`: The next element in the matrix.
    /// - `None`: The iterator is depleted.
    fn next(&mut self) -> Option<Self::Item> {
        for (i, j) in &mut self.mit {
            if self.dm.get(i, j).is_none() {
                return Some((i, j));
            }
        }
        self.dpit.next()
    }
}
