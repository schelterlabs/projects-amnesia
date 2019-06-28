#[cfg_attr(test, macro_use)]
extern crate ndarray;

pub mod itembased;
pub mod lsh;

trait IncrementalDecremental<T> {
    fn partial_fit(self: &mut Self, data: &[T]);
    fn forget(self: &mut Self, data: &T);
}