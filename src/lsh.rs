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

        assert!(num_components <= 32, "huhu");

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

    pub fn similar_sample_indexes(&self, sample: &Array<f64, Dim<[usize; 1]>>) -> FnvHashSet<u32> {
        let key = self.key(sample);

        //TODO clone necessary?
        match self.table.get(&key) {
            Some(set) => set.clone(),
            None => FnvHashSet::with_capacity_and_hasher(0, Default::default())
        }
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

    pub fn forget(&mut self, sample_index: u32, sample: &Array<f64, Dim<[usize; 1]>>) {
        let key = self.key(sample);
        self.table.get_mut(&key).unwrap().remove(&sample_index);
    }
}

#[cfg(test)]
mod tests {

    use crate::lsh::LshTable;

    #[test]
    fn toy_example() {

        let num_features = 5;
        let num_components = 3;


        let mut table = LshTable::new(num_features, num_components);

        let samples = vec![
            (1, array![1.0, 2.0, 3.0, 4.0, 5.0]),
            (2, array![1.0, 2.0, 3.0, 4.0, 5.0]),
            (3, array![1.0, 2.0, 3.0, 4.0, 5.0]),
            (4, array![-1.0, -1.0, -1.0, -1.0, -1.0])
        ];

        table.partial_fit(&samples);

        let similar_indexes = table.similar_sample_indexes(&array![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!(similar_indexes.contains(&1u32));
        assert!(similar_indexes.contains(&2u32));
        assert!(similar_indexes.contains(&3u32));
        assert!(!similar_indexes.contains(&4u32));

        table.forget(3, &array![1.0, 2.0, 3.0, 4.0, 5.0]);

        let similar_indexes = table.similar_sample_indexes(&array![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!(similar_indexes.contains(&1u32));
        assert!(similar_indexes.contains(&2u32));
        assert!(!similar_indexes.contains(&3u32));
        assert!(!similar_indexes.contains(&4u32));
    }
}