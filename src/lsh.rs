extern crate rand;
extern crate ndarray;

use rand::distributions::Normal;
use ndarray_rand::RandomExt;
use ndarray::{Array, Dim};
use fnv::{FnvHashMap, FnvHashSet};

pub struct LshTable {
    omega: Array<f64, Dim<[usize; 2]>>,
    table: FnvHashMap<u32, FnvHashSet<u32>>,
}

impl LshTable {

    pub fn new(num_features: usize, num_components: usize) -> LshTable {

        let distribution = Normal::new(0.0, 1.0 / num_components as f64);
        let omega = Array::random((num_features, num_components), distribution);

        let table = FnvHashMap::with_capacity_and_hasher(0, Default::default());

        LshTable { omega, table }
    }

    fn key(&self, sample: &Array<f64, Dim<[usize; 1]>>) -> u32 {

        let projected = sample.dot(&self.omega);
        let mut key = 0u32;
        for (dimension, value) in projected.iter().enumerate() {
            if *value > 0.0 {
                key |= 1u32 << dimension;
            }
        }

        key
    }

    pub fn partial_fit(&mut self, samples: &[(u32, Array<f64, Dim<[usize; 1]>>)]) {
        // TODO stacking and single MM would improve performance
        for (sample_index, sample) in samples.iter() {

            let key = self.key(sample);
            self.table.entry(key)
                .or_insert(FnvHashSet::with_capacity_and_hasher(0, Default::default()))
                .insert(*sample_index);
        }
    }

}
