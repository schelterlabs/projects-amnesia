#[cfg_attr(test, macro_use)]
extern crate ndarray;

pub mod itembased;
pub mod lsh;

trait IncrementalDecrementalModel<T, I, O> {
    fn partial_fit(self: &mut Self, items: &[T]);
    fn forget(self: &mut Self, item: &T);

    fn predict(self: &Self, item: &I) -> O;
}