#[cfg(test)]
#[macro_use]
extern crate ndarray;
#[cfg(not(test))]
extern crate ndarray;

pub mod itembased;
pub mod lsh;