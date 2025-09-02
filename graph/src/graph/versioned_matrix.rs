use std::ops::{Deref, DerefMut};

use crate::graph::matrix::{
    self, Dup, ElementWiseAdd, Get, Matrix, New, Remove, Set, Size, Transpose,
};

#[derive(Clone)]
struct Cow<T: Dup<T> + Clone> {
    inner: T,
    dup: bool,
}

impl<T: Dup<T> + Clone> Cow<T> {
    const fn new(inner: T) -> Self {
        Self { inner, dup: false }
    }

    fn new_version(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dup: true,
        }
    }
}

impl Deref for Cow<Matrix> {
    type Target = Matrix;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Cow<Matrix> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.dup {
            self.inner = self.inner.dup();
            self.dup = false;
        }
        &mut self.inner
    }
}

#[derive(Clone)]
pub struct VersionedMatrix {
    m: Matrix,
    dp: Cow<Matrix>,
    dm: Cow<Matrix>,
}

unsafe impl Send for VersionedMatrix {}
unsafe impl Sync for VersionedMatrix {}

impl VersionedMatrix {
    pub fn wait(&self) {
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
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
        self.m.resize(nrows, ncols);
        self.dp.resize(nrows, ncols);
        self.dm.resize(nrows, ncols);
    }

    fn nvals(&self) -> u64 {
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }
}

impl New for VersionedMatrix {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Matrix::new(nrows, ncols),
            dp: Cow::new(Matrix::new(nrows, ncols)),
            dm: Cow::new(Matrix::new(nrows, ncols)),
        }
    }
}

impl Dup<Self> for VersionedMatrix {
    fn dup(&self) -> Self {
        Self {
            m: self.m.clone(),
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
        Iter::new(self, min_row, max_row)
    }

    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        // TODO: fix
        let mut m = self.m.dup();
        m.element_wise_add(&self.dp);
        m.remove_all(&self.dm);
        m
    }

    pub fn print(&self) {
        self.m.print();
        self.dp.print();
        self.dm.print();
    }
}

impl Remove for VersionedMatrix {
    fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        if self.m.get(i, j).is_some() {
            self.dm.set(i, j, true);
        } else {
            self.dp.remove(i, j);
        }
    }

    fn remove_all(
        &mut self,
        b: &Self,
    ) {
        todo!()
    }
}

// impl ElementWiseAdd<bool> for VersionedMatrix<bool> {
//     fn element_wise_add(
//         &mut self,
//         b: &Self,
//     ) {
//         // TODO: fixß
//         self.dp.element_wise_add(&b.m);
//         self.dp.element_wise_add(&b.dp);
//         self.dm.element_wise_add(&b.dm);
//     }
// }

// impl ElementWiseMultiply<bool> for VersionedMatrix<bool> {
//     fn element_wise_multiply(
//         &mut self,
//         b: &Self,
//     ) {
//         self.m.element_wise_multiply(&b.m);
//         self.dp.element_wise_multiply(&b.dp);
//         self.dm.element_wise_multiply(&b.dm);
//     }
// }

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
        if self.m.get(i, j).is_some() {
            self.dm.remove(i, j);
        } else {
            self.dp.set(i, j, value);
        }
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
            m: self.m.transpose(),
            dp: Cow::new(self.dp.transpose()),
            dm: Cow::new(self.dm.transpose()),
        }
    }
}

pub struct Iter {
    mit: matrix::Iter,
    dp: matrix::Iter,
}

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
            dp: m.dp.iter(min_row, max_row),
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
        // TODO: fix
        self.mit.next().or_else(|| self.dp.next())
    }
}
