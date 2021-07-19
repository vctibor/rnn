#![feature(const_generics)]
//#![feature(const_evaluatable_checked)]

use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Deref, DerefMut};

/// R*C matrix of type T.
struct Matrix<T, const R: usize, const C: usize> where T: Copy
{
    data: Rc<[[T; C]; R]>
}

impl<T, const R: usize, const C: usize> Deref for Matrix<T, R, C> where T: Copy {
    type Target = [[T; C]; R];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/*
impl<T, const R: usize, const C: usize> DerefMut for Matrix<T, R, C> where T: Copy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut()
    }
}
*/

impl<T, const R: usize, const C: usize> Clone for Matrix<T, R, C> where T: Copy {
    fn clone(&self) -> Matrix<T, R, C> {
        Matrix {
            data: self.data.clone()
        }
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> where T: Copy
{
    fn get(&self, r: usize, c: usize) -> T {
        assert!(r < R);
        assert!(c < C);
        self.data[r][c]
    }

    /*
    fn put(&mut self, r: usize, c: usize, val: T) {
        assert!(r < R);
        assert!(c < C);
        self.data[r][c] = val;
    }
    */
}