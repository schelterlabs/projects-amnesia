extern crate timely;
extern crate differential_dataflow;
extern crate blas;

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;
use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::operators::Join;
use differential_dataflow::Hashable;
use timely::dataflow::operators::Probe;

use std::hash::Hasher;
use std::cmp::Ordering;

use blas::*;
use self::differential_dataflow::operators::arrange::ArrangeByKey;

#[derive(Abomonation, Debug, Clone)]
pub struct Sample {
    pub id: u64,
    pub features: Vec<f64>,
    pub label: u8,
}

impl Sample {
    pub fn new(id: u64, features: Vec<f64>, label: u8) -> Sample {
        Sample { id, features, label }
    }
}

impl PartialEq for Sample {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Sample {}


impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for Sample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hashable for Sample {
    type Output = u64;
    fn hashed(&self) -> u64 {
        let mut h: ::fnv::FnvHasher = Default::default();
        h.write_u64(self.id);
        h.finish()
    }
}

#[derive(Abomonation, Debug, Clone)]
pub struct ProjectionMatrix {
    pub table_index: usize,
    pub weights: Vec<f64>,
}

impl ProjectionMatrix {
    pub fn new(
        table_index: usize,
        weights: Vec<f64>)
    -> ProjectionMatrix {
        ProjectionMatrix { table_index, weights }
    }
}

impl PartialEq for ProjectionMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.table_index == other.table_index
    }
}

impl Eq for ProjectionMatrix {}


impl PartialOrd for ProjectionMatrix {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.table_index.cmp(&other.table_index))
    }
}

impl Ord for ProjectionMatrix {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hashable for ProjectionMatrix {
    type Output = u64;
    fn hashed(&self) -> u64 {
        let mut h: ::fnv::FnvHasher = Default::default();
        h.write_usize(self.table_index);
        h.finish()
    }
}


pub fn lsh<T>(
    worker: &mut Worker<Allocator>,
    examples_input: &mut InputSession<T, Sample, isize>,
    tables_input: &mut InputSession<T, ProjectionMatrix, isize>)
-> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    let mut probe = timely::dataflow::operators::probe::Handle::new();

    worker.dataflow(|scope| {

        let initial_projection_matrices = tables_input.to_collection(scope);
        let examples = examples_input.to_collection(scope);

        let projection_matrices = initial_projection_matrices
            .map(|matrix| ((), matrix));

        // Couldn't find an operator for cartesian products, so we mimic this with a join
        let indexed_examples = examples
            .map(|example| ((), example))
            .join_map(&projection_matrices, |_, example, matrix| {

                let num_features = example.features.len();
                let num_hash_dimensions = matrix.weights.len() / num_features;

                let mut projection: Vec<f64> = vec![0.0; num_hash_dimensions as usize];

                // Random projection of features
                unsafe {
                    dgemv(
                        b'T',
                        num_features as i32,
                        num_hash_dimensions as i32,
                        1.0,
                        &matrix.weights,
                        num_features as i32,
                        &example.features,
                        1,
                        0.0,
                        &mut projection,
                        1);
                }

                // Signs of the result of the random projection give us the bucket key
                let mut key = 0u32;
                for (dimension, value) in projection.iter().enumerate() {
                    if *value > 0.0 {
                        key |= 1u32 << dimension as u32;
                    }
                }

                ((matrix.table_index, key), example.id)
            });

        // Buckets per table, could be used for search afterwards
        let arranged_indexed_examples = indexed_examples.arrange_by_key();

        arranged_indexed_examples.stream.probe_with(&mut probe);
    });

    probe
}