#![allow(clippy::doc_markdown)]

use std::{
    mem::MaybeUninit,
    os::raw::c_void,
    ptr::null_mut,
    sync::{Arc, Mutex},
};

use crate::graph::GraphBLAS::{
    GrB_BOOL, GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R, GrB_DESC_RC,
    GrB_DESC_RCT0, GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC, GrB_DESC_RSCT0,
    GrB_DESC_RSCT0T1, GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1, GrB_DESC_RST1, GrB_DESC_RT0,
    GrB_DESC_RT0T1, GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC, GrB_DESC_SCT0, GrB_DESC_SCT0T1,
    GrB_DESC_SCT1, GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1, GrB_DESC_T0, GrB_DESC_T0T1,
    GrB_DESC_T1, GrB_Descriptor, GrB_Info, GrB_Matrix, GrB_Matrix_dup,
    GrB_Matrix_eWiseAdd_Semiring, GrB_Matrix_eWiseMult_Semiring, GrB_Matrix_extractElement_BOOL,
    GrB_Matrix_free, GrB_Matrix_get_INT32, GrB_Matrix_ncols, GrB_Matrix_new, GrB_Matrix_nrows,
    GrB_Matrix_nvals, GrB_Matrix_removeElement, GrB_Matrix_resize, GrB_Matrix_setElement_BOOL,
    GrB_Matrix_wait, GrB_Mode, GrB_WaitMode, GrB_finalize, GrB_mxm, GrB_transpose,
    GxB_ANY_PAIR_BOOL, GxB_Iterator, GxB_Iterator_free, GxB_Iterator_new,
    GxB_Matrix_Iterator_attach, GxB_Matrix_Iterator_getIndex, GxB_Matrix_Iterator_next,
    GxB_Matrix_fprint, GxB_Matrix_memoryUsage, GxB_Option_Field, GxB_Print_Level, GxB_init,
    GxB_rowIterator_getRowIndex, GxB_rowIterator_nextRow, GxB_rowIterator_seekRow,
};

/// Initializes the GraphBLAS library in non-blocking mode.
#[allow(clippy::similar_names)]
pub fn init(
    user_malloc_function: Option<unsafe extern "C" fn(arg1: usize) -> *mut c_void>,
    user_calloc_function: Option<unsafe extern "C" fn(arg1: usize, arg2: usize) -> *mut c_void>,
    user_realloc_function: Option<
        unsafe extern "C" fn(arg1: *mut c_void, arg2: usize) -> *mut c_void,
    >,
    user_free_function: Option<unsafe extern "C" fn(arg1: *mut c_void)>,
) {
    unsafe {
        GxB_init(
            GrB_Mode::GrB_NONBLOCKING as _,
            user_malloc_function,
            user_calloc_function,
            user_realloc_function,
            user_free_function,
        );
    }
}

/// Finalizes the GraphBLAS library, releasing all resources.
pub fn shutdown() {
    unsafe {
        GrB_finalize();
    }
}

/// A trait for querying and modifying the size of a matrix.
pub trait Size {
    /// Returns the number of rows in the matrix.
    fn nrows(&self) -> u64;

    /// Returns the number of columns in the matrix.
    fn ncols(&self) -> u64;

    /// Resizes the matrix to the specified number of rows and columns.
    ///
    /// # Parameters
    /// - `nrows`: The new number of rows.
    /// - `ncols`: The new number of columns.
    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    );

    /// Returns the number of non-zero values in the matrix.
    fn nvals(&self) -> u64;
}

/// A trait for retrieving elements from a matrix.
pub trait Get {
    /// Retrieves the element at the specified row and column.
    /// Returns `None` if the element does not exist.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    ///
    /// # Returns
    /// - `Some(T)`: The element at the specified position.
    /// - `None`: The element does not exist.
    fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool>;
}

/// A trait for setting elements in a matrix.
pub trait Set {
    /// Sets the element at the specified row and column to the given value.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    /// - `value`: The value to set.
    fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    );
}

/// A trait for removing elements from a matrix.
pub trait Remove {
    /// Removes the element at the specified row and column.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    fn remove(
        &mut self,
        i: u64,
        j: u64,
    );

    fn remove_all(
        &mut self,
        b: &Self,
    );
}

pub trait Transpose {
    /// Transposes the matrix.
    #[must_use]
    fn transpose(&self) -> Self;
}

pub trait ElementWiseAdd {
    fn element_wise_add(
        &mut self,
        b: &Self,
    );
}

impl ElementWiseAdd for Matrix {
    fn element_wise_add(
        &mut self,
        b: &Self,
    ) {
        unsafe {
            let info = GrB_Matrix_eWiseAdd_Semiring(
                *self.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *self.m,
                *b.m,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

pub trait MaskedElementWiseMultiply {
    fn element_wise_multiply(
        &mut self,
        mask: Option<&Matrix>,
        a: Option<&Self>,
        b: Option<&Self>,
        descriptor: Option<Descriptor>,
    );
}

pub enum Descriptor {
    T0,
    T1,
    T0T1,
    C,
    CT0,
    CT1,
    CT0T1,
    S,
    ST0,
    ST1,
    ST0T1,
    SC,
    SCT0,
    SCT1,
    SCT0T1,
    R,
    RT0,
    RT1,
    RT0T1,
    RC,
    RCT0,
    RCT1,
    RCT0T1,
    RS,
    RST0,
    RST1,
    RST0T1,
    RSC,
    RSCT0,
    RSCT1,
    RSCT0T1,
}

impl From<Descriptor> for GrB_Descriptor {
    fn from(descriptor: Descriptor) -> Self {
        unsafe {
            match descriptor {
                Descriptor::T0 => GrB_DESC_T0,
                Descriptor::T1 => GrB_DESC_T1,
                Descriptor::T0T1 => GrB_DESC_T0T1,
                Descriptor::C => GrB_DESC_C,
                Descriptor::CT0 => GrB_DESC_CT0,
                Descriptor::CT1 => GrB_DESC_CT1,
                Descriptor::CT0T1 => GrB_DESC_CT0T1,
                Descriptor::S => GrB_DESC_S,
                Descriptor::ST0 => GrB_DESC_ST0,
                Descriptor::ST1 => GrB_DESC_ST1,
                Descriptor::ST0T1 => GrB_DESC_ST0T1,
                Descriptor::SC => GrB_DESC_SC,
                Descriptor::SCT0 => GrB_DESC_SCT0,
                Descriptor::SCT1 => GrB_DESC_SCT1,
                Descriptor::SCT0T1 => GrB_DESC_SCT0T1,
                Descriptor::R => GrB_DESC_R,
                Descriptor::RT0 => GrB_DESC_RT0,
                Descriptor::RT1 => GrB_DESC_RT1,
                Descriptor::RT0T1 => GrB_DESC_RT0T1,
                Descriptor::RC => GrB_DESC_RC,
                Descriptor::RCT0 => GrB_DESC_RCT0,
                Descriptor::RCT1 => GrB_DESC_RCT1,
                Descriptor::RCT0T1 => GrB_DESC_RCT0T1,
                Descriptor::RS => GrB_DESC_RS,
                Descriptor::RST0 => GrB_DESC_RST0,
                Descriptor::RST1 => GrB_DESC_RST1,
                Descriptor::RST0T1 => GrB_DESC_RST0T1,
                Descriptor::RSC => GrB_DESC_RSC,
                Descriptor::RSCT0 => GrB_DESC_RSCT0,
                Descriptor::RSCT1 => GrB_DESC_RSCT1,
                Descriptor::RSCT0T1 => GrB_DESC_RSCT0T1,
            }
        }
    }
}

impl MaskedElementWiseMultiply for Matrix {
    fn element_wise_multiply(
        &mut self,
        mask: Option<&Self>,
        a: Option<&Self>,
        b: Option<&Self>,
        descriptor: Option<Descriptor>,
    ) {
        unsafe {
            let info = GrB_Matrix_eWiseMult_Semiring(
                *self.m,
                mask.map_or(null_mut(), |m| *m.m),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                a.map_or(*self.m, |a| *a.m),
                b.map_or(*self.m, |b| *b.m),
                descriptor.map_or(null_mut(), |d| d.into()),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

pub trait MxM {
    /// Multiplies two matrices and stores the result in the current matrix.
    ///
    /// # Parameters
    /// - `b`: The matrix to multiply with.
    fn lmxm(
        &mut self,
        b: &Self,
    );

    fn rmxm(
        &mut self,
        b: &Self,
    );
}

impl MxM for Matrix {
    fn lmxm(
        &mut self,
        b: &Self,
    ) {
        unsafe {
            let info = GrB_mxm(
                *self.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *self.m,
                *b.m,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    fn rmxm(
        &mut self,
        b: &Self,
    ) {
        unsafe {
            let info = GrB_mxm(
                *self.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *b.m,
                *self.m,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

/// A wrapper around a GraphBLAS boolean matrix.
#[derive(Clone)]
pub struct Matrix {
    /// The underlying GraphBLAS matrix.
    m: Arc<GrB_Matrix>,
    wait_lock: Arc<Mutex<()>>,
}

unsafe impl Send for Matrix {}
unsafe impl Sync for Matrix {}

impl Drop for Matrix {
    fn drop(&mut self) {
        if let Some(m) = Arc::get_mut(&mut self.m) {
            unsafe {
                let info = GrB_Matrix_free(m);
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
        }
    }
}

impl Matrix {
    #[must_use]
    pub fn pending(&self) -> bool {
        unsafe {
            let mut pending = MaybeUninit::uninit();
            let info = GrB_Matrix_get_INT32(
                *self.m,
                pending.as_mut_ptr(),
                GxB_Option_Field::GxB_WILL_WAIT as _,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            pending.assume_init() == 1
        }
    }

    pub fn wait(&self) {
        if self.pending() {
            let _lock = self.wait_lock.lock();
            if self.pending() {
                unsafe {
                    let info = GrB_Matrix_wait(*self.m, GrB_WaitMode::GrB_MATERIALIZE as _);
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                }
            }
        }
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        unsafe {
            let mut usage = 0usize;
            let info = GxB_Matrix_memoryUsage(&raw mut usage, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            usage
        }
    }
}

impl Size for Matrix {
    fn nrows(&self) -> u64 {
        unsafe {
            let mut nrows = 0u64;
            let info = GrB_Matrix_nrows(&raw mut nrows, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nrows
        }
    }

    fn ncols(&self) -> u64 {
        unsafe {
            let mut ncols = 0u64;
            let info = GrB_Matrix_ncols(&raw mut ncols, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            ncols
        }
    }

    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        unsafe {
            let info = GrB_Matrix_resize(*self.m, nrows, ncols);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    fn nvals(&self) -> u64 {
        unsafe {
            let mut nvals = 0u64;
            let info = GrB_Matrix_nvals(&raw mut nvals, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nvals
        }
    }
}

pub trait New {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self;
}

impl New for Matrix {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        unsafe {
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_BOOL, nrows, ncols);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            Self {
                m: Arc::new(m.assume_init()),
                wait_lock: Arc::new(Mutex::new(())),
            }
        }
    }
}

pub trait Dup<T> {
    fn dup(&self) -> T;
}

impl Dup<Self> for Matrix {
    fn dup(&self) -> Self {
        Self {
            m: Arc::new(unsafe {
                let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
                let info = GrB_Matrix_dup(m.as_mut_ptr(), *self.m);
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                m.assume_init()
            }),
            wait_lock: Arc::new(Mutex::new(())),
        }
    }
}

impl Matrix {
    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        Iter::new(self, min_row, max_row)
    }

    pub fn print(&self) {
        unsafe {
            let info = GrB_Matrix_wait(*self.m, GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let info = GxB_Matrix_fprint(
                *self.m,
                null_mut(),
                GxB_Print_Level::GxB_COMPLETE as _,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Remove for Matrix {
    fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        unsafe {
            let info = GrB_Matrix_removeElement(*self.m, i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    fn remove_all(
        &mut self,
        b: &Self,
    ) {
        unsafe {
            let info = GrB_transpose(*self.m, *b.m, null_mut(), *self.m, GrB_DESC_RCT0);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Get for Matrix {
    /// Retrieves the boolean value at the specified position in the matrix.
    /// Returns `None` if the element does not exist.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    ///
    /// # Returns
    /// - `Some(bool)`: The boolean value at the specified position.
    /// - `None`: The element does not exist.
    fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool> {
        unsafe {
            let mut m: MaybeUninit<bool> = MaybeUninit::uninit();
            let info = GrB_Matrix_extractElement_BOOL(m.as_mut_ptr(), *self.m, i, j);
            if info == GrB_Info::GrB_SUCCESS {
                Some(m.assume_init())
            } else {
                None
            }
        }
    }
}

impl Set for Matrix {
    fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        unsafe {
            let info = GrB_Matrix_setElement_BOOL(*self.m, value, i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Transpose for Matrix
where
    Self: New,
{
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    fn transpose(&self) -> Self {
        let transpose = Self::new(self.ncols(), self.nrows());
        unsafe {
            let info = GrB_transpose(*transpose.m, null_mut(), null_mut(), *self.m, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        transpose
    }
}

pub struct Iter {
    m: Arc<GrB_Matrix>,
    /// The underlying GraphBLAS iterator.
    inner: GxB_Iterator,
    /// Indicates whether the iterator is depleted.
    depleted: bool,
    /// The maximum row index for the iterator.
    max_row: u64,
}

impl Drop for Iter {
    /// Frees the GraphBLAS iterator when the `Iter` is dropped.
    fn drop(&mut self) {
        unsafe {
            if let Some(m) = Arc::get_mut(&mut self.m) {
                let info = GrB_Matrix_free(m);
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
            GxB_Iterator_free(&raw mut self.inner);
        }
    }
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
        m: &Matrix,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        unsafe {
            let mut iter = MaybeUninit::uninit();
            let info = GxB_Iterator_new(iter.as_mut_ptr());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let iter = iter.assume_init();
            let info = GxB_Matrix_Iterator_attach(iter, *m.m, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut info = GxB_rowIterator_seekRow(iter, min_row);
            debug_assert!(
                info == GrB_Info::GrB_SUCCESS
                    || info == GrB_Info::GrB_NO_VALUE
                    || info == GrB_Info::GxB_EXHAUSTED
            );
            if info == GrB_Info::GrB_NO_VALUE {
                while info == GrB_Info::GrB_NO_VALUE && GxB_rowIterator_getRowIndex(iter) < max_row
                {
                    info = GxB_rowIterator_nextRow(iter);
                }
            }
            Self {
                m: m.m.clone(),
                inner: iter,
                depleted: info != GrB_Info::GrB_SUCCESS
                    || GxB_rowIterator_getRowIndex(iter) > max_row,
                max_row,
            }
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
        if self.depleted {
            return None;
        }
        unsafe {
            let mut row = 0u64;
            let mut col = 0u64;
            GxB_Matrix_Iterator_getIndex(self.inner, &raw mut row, &raw mut col);
            self.depleted = GxB_Matrix_Iterator_next(self.inner) != GrB_Info::GrB_SUCCESS
                || GxB_rowIterator_getRowIndex(self.inner) > self.max_row;
            Some((row, col))
        }
    }
}
