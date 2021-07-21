#![feature(const_generics)]
//#![feature(const_evaluatable_checked)]

use std::cell::RefCell;
use std::convert::TryInto;
use std::rc::Rc;
use std::ops::{Deref, DerefMut};


/// R*C matrix of type T.
struct Matrix<T, const R: usize, const C: usize> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug + std::cmp::Eq,
{
    data: Rc<[[T; C]; R]>
}

impl<T, const R: usize, const C: usize> Deref for Matrix<T, R, C> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug + std::cmp::Eq, {
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

impl<T, const R: usize, const C: usize> Clone for Matrix<T, R, C> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug + std::cmp::Eq, {
    fn clone(&self) -> Matrix<T, R, C> {
        Matrix {
            data: self.data.clone()
        }
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug + std::cmp::Eq,
{
    fn new(data: [[T; C]; R]) -> Matrix<T, R, C> {
        Matrix {
            data: Rc::new(data)
        }
    }

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

    fn subtract(&self, other: &Matrix<T, R, C>) -> Matrix<T, R, C> {
        let self_data = self.data.as_ref();
        let other_data = other.data.as_ref();

        let result: [[T; C]; R] = self_data.iter().zip(other_data).map(|(self_row, other_row)| {
            let row_result: [T; C] = self_row.iter().zip(other_row).map(|(self_column, other_column)| {
                *self_column - *other_column
            }).collect::<Vec<T>>().try_into().unwrap();
            row_result
        }).collect::<Vec<[T; C]>>().try_into().unwrap();

        Matrix::new(result)
    }
}

/*
impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug +  std::cmp::Eq, {
    fn eq(&self, other: &Self) -> bool {
        let self_data = self.data.as_ref();
        let other_data = other.data.as_ref();

        self_data.iter().zip(other_data).map(|(self_row, other_row)| {
            self_row.iter().zip(other_row).map(|(self_column, other_column)| {
                if *self_column != *other_column { return false; }
            });
        });

        true
    }
}

impl<T, const R: usize, const C: usize> Eq for Matrix<T, R, C> where T: std::marker::Copy + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> + std::ops::Add<Output=T> + std::fmt::Debug + std::cmp::Eq,
{

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_subtraction() {

        let mat1 = Matrix::new([
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0]
        ]);

        let mat2 = Matrix::new([
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0]
        ]);

        let expected = Matrix::new([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]);

        assert_eq!(expected, mat1.subtract(&mat2));
    }
}
*/