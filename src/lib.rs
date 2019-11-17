#[cfg_attr(test, macro_use)]
extern crate ndarray;

pub mod itembased;
pub mod lsh;
pub mod ridge;

pub mod differential;

pub trait IncrementalDecrementalModel<T, I, O> {
    fn partial_fit(self: &mut Self, data: &[T]);
    fn forget(self: &mut Self, data: &T);

    fn predict(self: &Self, data: &I) -> O;
}