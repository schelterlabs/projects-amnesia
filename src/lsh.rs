extern crate rand;
extern crate ndarray;

use rand::distributions::Normal;
use ndarray_rand::RandomExt;
use ndarray::{Array, Dim};
use fnv::{FnvHashMap, FnvHashSet};
use std::hash::Hash;
use std::hash::Hasher;

use crate::IncrementalDecremental;

#[derive(Debug, PartialEq, Clone)]
pub struct Example {
    features: Array<f64, Dim<[usize; 1]>>,
}

impl Example {
    fn new(features: Array<f64, Dim<[usize; 1]>>,) -> Example {
        Example { features }
    }
}

impl Eq for Example { }

impl Hash for Example {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elem in self.features.iter() {
            state.write_u64(elem.to_bits());
        }
    }
}

pub struct LshTable {
    omega: Array<f64, Dim<[usize; 2]>>,
    table: FnvHashMap<u32, FnvHashSet<Example>>,
}



// TODO Activate blas feature for ndarray
impl LshTable {

    pub fn new(num_features: usize, num_components: usize) -> LshTable {

        assert!(num_components <= 32, "Maximum of 32 components supported currently.");

        let distribution = Normal::new(0.0, 1.0 / num_components as f64);
        let omega = Array::random((num_features, num_components), distribution);

        let table = FnvHashMap::with_capacity_and_hasher(0, Default::default());

        LshTable { omega, table }
    }

    fn key(&self, example: &Example) -> u32 {

        let projected = example.features.dot(&self.omega);
        let mut key = 0u32;
        for (dimension, value) in projected.iter().enumerate() {
            if *value > 0.0 {
                key |= 1u32 << dimension;
            }
        }

        key
    }

    pub fn similar_sample_indexes(&self, example: &Example) -> FnvHashSet<Example> {
        let key = self.key(example);

        //TODO clone necessary?
        match self.table.get(&key) {
            Some(set) => set.clone(),
            None => FnvHashSet::with_capacity_and_hasher(0, Default::default())
        }
    }
}

impl IncrementalDecremental<Example> for LshTable {

    fn partial_fit(&mut self, data: &[Example]) {
        // TODO stacking and single MM would improve performance
        for sample in data.iter() {

            let key = self.key(sample);
            self.table.entry(key)
                .or_insert(FnvHashSet::with_capacity_and_hasher(0, Default::default()))
                .insert((*sample).clone());
        }
    }

    fn forget(&mut self, sample: &Example) {
        let key = self.key(sample);
        self.table.get_mut(&key).unwrap().remove(sample);
    }
}

#[cfg(test)]
mod tests {

    extern crate fnv;

    use crate::IncrementalDecremental;
    use crate::lsh::LshTable;
    use crate::lsh::Example;

    #[test]
    fn compare_examples() {
        let example_1 = Example::new(array![1.0, 2.0, 3.0]);
        let example_2 = Example::new(array![1.0, 2.0, 3.0]);
        let example_3 = Example::new(array![3.0, 4.0, 5.0]);

        assert_eq!(example_1, example_2);
        assert_eq!(example_2, example_1);
        assert_ne!(example_1, example_3);
        assert_ne!(example_3, example_1);
        assert_ne!(example_2, example_3);
        assert_ne!(example_3, example_2);

        let mut set = fnv::FnvHashSet::with_capacity_and_hasher(0, Default::default());

        set.insert(example_1.clone());
        assert!(set.contains(&example_1));
        assert!(!set.contains(&example_3));
    }

    #[test]
    fn toy_example() {

        let num_features = 5;
        let num_components = 3;


        let mut table = LshTable::new(num_features, num_components);

        let samples = vec![
            Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0]),
            Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0]),
            Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0]),
            Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0]),
        ];

        table.partial_fit(&samples);

        let similar_indexes = table.similar_sample_indexes(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0]));

        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0])));
        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0])));
        assert!(!similar_indexes.contains(&Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0])));

        table.forget(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0]));

        let similar_indexes = table.similar_sample_indexes(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0]));

        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0])));
        assert!(!similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0])));
        assert!(!similar_indexes.contains(&Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0])));
    }
}