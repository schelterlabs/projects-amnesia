extern crate rand;
extern crate ndarray;

use rand::distributions::Normal;
use ndarray_rand::RandomExt;
use ndarray::{Array, Dim};
use fnv::{FnvHashMap, FnvHashSet};
use std::hash::Hash;
use std::hash::Hasher;
use ndarray_linalg::norm::Norm;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::IncrementalDecrementalModel;

type FeatureVector = Array<f64, Dim<[usize; 1]>>;

#[derive(Debug, PartialEq, Clone)]
pub struct Example {
    features: FeatureVector,
    label: u8,
}

impl Example {
    fn new(features: FeatureVector, label: u8) -> Example {
        Example { features, label }
    }
}

impl Eq for Example { }

impl Hash for Example {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elem in self.features.iter() {
            state.write_u64(elem.to_bits());
        }
        state.write_u8(self.label);
    }
}

// We could probably save some indirections here...
pub struct ApproximateKnn {
    tables: Vec<LshTable>,
    k: usize,
    num_classes: usize,
}

impl ApproximateKnn {

    pub fn new(
        num_tables: usize,
        num_features: usize,
        num_components: usize,
        k: usize,
        num_classes: usize)
        -> ApproximateKnn {

        let tables: Vec<LshTable> = (0..num_tables)
            .map(|_| LshTable::new(num_features, num_components, k, num_classes))
            .collect();

        ApproximateKnn { tables, k, num_classes }
    }
}


impl IncrementalDecrementalModel<Example, FeatureVector, Vec<f64>> for ApproximateKnn {

    fn partial_fit(&mut self, examples: &[Example]) {
        self.tables.iter_mut().for_each(|table| {
            table.partial_fit(&examples);
        })
    }

    fn forget(&mut self, example: &Example) {
        self.tables.iter_mut().for_each(|table| {
            table.forget(&example);
        })
    }

    fn predict(&self, example: &FeatureVector) -> Vec<f64> {
        let mut close_examples = FnvHashSet::with_capacity_and_hasher(0, Default::default());

        self.tables.iter().for_each(|table| {

            let close_in_table = table.close_to(example);

            match close_in_table {
                Some(close_examples_in_table) =>
                    close_examples_in_table.iter().for_each(|close_example| {
                        close_examples.insert(close_example);
                    }),
                None => (),
            }


        });

        let mut class_counts = vec![0f64; self.num_classes];

        // We'll use a heap to keep track of the current top-n scored items
        let mut top_examples = BinaryHeap::with_capacity(self.k);

        for close_example in close_examples.iter() {
            let distance = (*&example - &close_example.features).norm_l2();

            let scored = ScoredExample { label: close_example.label, distance };

            if top_examples.len() < self.k {
                top_examples.push(scored);
            } else {
                let mut top = top_examples.peek_mut().unwrap();
                if scored < *top {
                    *top = scored;
                }
            }
        }


        for scored in top_examples.iter() {
            class_counts[scored.label as usize] += 1.0;
        }

        let num_examples = top_examples.len();

        class_counts.iter_mut().for_each(|x| *x /= num_examples as f64);

        class_counts
    }
}


struct LshTable {
    omega: Array<f64, Dim<[usize; 2]>>,
    table: FnvHashMap<u32, FnvHashSet<Example>>,
    k: usize,
    num_classes: usize,
}


impl LshTable {

    pub fn new(
        num_features: usize,
        num_components: usize,
        k: usize,
        num_classes: usize)
    -> LshTable {

        assert!(num_components <= 32, "Maximum of 32 components supported currently.");

        let distribution = Normal::new(0.0, 1.0 / num_components as f64);
        let omega = Array::random((num_features, num_components), distribution);

        let table = FnvHashMap::with_capacity_and_hasher(0, Default::default());

        LshTable { omega, table, k, num_classes }
    }

    // There's probably a not too complicated SIMD accelerated version of this
    fn key(&self, features: &FeatureVector) -> u32 {

        let projected = features.dot(&self.omega);
        let mut key = 0u32;
        for (dimension, value) in projected.iter().enumerate() {
            if *value > 0.0 {
                key |= 1u32 << dimension;
            }
        }

        key
    }

    fn close_to(&self, features: &FeatureVector) -> Option<&FnvHashSet<Example>> {
        let key = self.key(features);
        self.table.get(&key)
    }
}

impl IncrementalDecrementalModel<Example, FeatureVector, Vec<f64>> for LshTable {

    fn partial_fit(&mut self, data: &[Example]) {
        // TODO stacking and single MM would improve performance
        for sample in data.iter() {

            // TODO check bounds for label
            let key = self.key(&sample.features);
            self.table.entry(key)
                .or_insert(FnvHashSet::with_capacity_and_hasher(0, Default::default()))
                .insert((*sample).clone());
        }
    }

    fn forget(&mut self, sample: &Example) {
        let key = self.key(&sample.features);
        self.table.get_mut(&key).unwrap().remove(sample);
    }

    fn predict(&self, features: &FeatureVector) -> Vec<f64> {
        let key = self.key(features);

        match self.table.get(&key) {
            Some(bucket) => {

                let mut class_counts = vec![0f64; self.num_classes];

                // We'll use a heap to keep track of the current top-n scored items
                let mut top_examples = BinaryHeap::with_capacity(self.k);

                for close_example in bucket.iter() {
                    let distance = (*&features - &close_example.features).norm_l2();

                    let scored = ScoredExample { label: close_example.label, distance };

                    if top_examples.len() < self.k {
                        top_examples.push(scored);
                    } else {
                        let mut top = top_examples.peek_mut().unwrap();
                        if scored < *top {
                            *top = scored;
                        }
                    }
                }


                for scored in top_examples.iter() {
                    class_counts[scored.label as usize] += 1.0;
                }

                let num_examples = top_examples.len();

                class_counts.iter_mut().for_each(|x| *x /= num_examples as f64);

                class_counts
            },

            None => vec![0f64; self.num_classes]
        }
    }
}


/// Result type used to find the top-k closest examples per bucket via a binary heap
#[derive(PartialEq, Debug)]
struct ScoredExample {
    label: u8,
    distance: f64,
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(scored_item_a: &ScoredExample, scored_item_b: &ScoredExample) -> Ordering {
    match scored_item_a.distance.partial_cmp(&scored_item_b.distance) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        None => Ordering::Equal
    }
}

impl Eq for ScoredExample {}

impl Ord for ScoredExample {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for ScoredExample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}


#[cfg(test)]
mod tests {

    extern crate fnv;

    use crate::IncrementalDecrementalModel;
    use crate::lsh::LshTable;
    use crate::lsh::Example;

    #[test]
    fn compare_examples() {
        let example_1 = Example::new(array![1.0, 2.0, 3.0], 1);
        let example_2 = Example::new(array![1.0, 2.0, 3.0], 1);
        let example_3 = Example::new(array![3.0, 4.0, 5.0], 1);

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
        let k = 3;

        let mut table = LshTable::new(num_features, num_components, k, 2);

        let samples = vec![
            Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 0),
            Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 0),
            Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0], 0),
            Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0], 1),
        ];

        table.partial_fit(&samples);

        let prediction = table.predict(&array![1.0, 2.0, 3.0, 4.0, 5.0]);

        let prob_mass: f64 = prediction.iter().sum();
        println!("{:?}", prob_mass);

        assert!((1.0_f64 - prob_mass).abs() < 0.00001_f64);

        /*
        let similar_indexes = table.predict(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 1));

        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 1)));
        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0], 1)));
        assert!(!similar_indexes.contains(&Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0], 2)));

        table.forget(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0], 1));

        let similar_indexes = table.predict(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 1));

        assert!(similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 4.0, 5.0], 1)));
        assert!(!similar_indexes.contains(&Example::new(array![1.0, 2.0, 3.0, 3.99, 5.0], 1)));
        assert!(!similar_indexes.contains(&Example::new(array![-1.0, -1.0, -1.0, -1.0, -1.0], 2)));
        */
    }
}