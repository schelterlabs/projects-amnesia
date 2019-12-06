extern crate timely;
extern crate differential_dataflow;

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;
use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::Hashable;
use differential_dataflow::operators::Reduce;

use std::hash::Hasher;
use std::cmp::Ordering;


#[derive(Abomonation, Debug, Clone)]
pub struct CategoricalSample {
    pub id: u64,
    pub features: Vec<u32>,
    pub label: u8,
}

impl CategoricalSample {
    pub fn new(id: u64, features: Vec<u32>, label: u8) -> CategoricalSample {
        CategoricalSample { id, features, label }
    }
}

impl PartialEq for CategoricalSample {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CategoricalSample {}


impl PartialOrd for CategoricalSample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for CategoricalSample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hashable for CategoricalSample {
    type Output = u64;
    fn hashed(&self) -> u64 {
        let mut h: ::fnv::FnvHasher = Default::default();
        h.write_u64(self.id);
        h.finish()
    }
}

// Very simple version of naive bayes in DD
pub fn mnb<T>(
    worker: &mut Worker<Allocator>,
    examples_input: &mut InputSession<T, CategoricalSample, isize>
) -> (ProbeHandle<T>, ProbeHandle<T>)
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    worker.dataflow(|scope| {
        let examples = examples_input.to_collection(scope);

        let features_per_label = examples
            .flat_map(|example: CategoricalSample| {
                let label = example.label;
                // We only have binary features in our example datasets
                example.features.into_iter()
                    .map(move |feature_index| ((feature_index, label), 1_u32))
            });

        // Counts per feature and label
        let feature_per_label_counts = features_per_label
            .reduce(|_feature_index_and_label, records, output| {
                let sum: u32 = records.iter().map(|record| *record.0).sum();
                output.push((sum, 1));
            });

        // Counts per label
        let label_counts = feature_per_label_counts
            .map(|((_feature_index, label), count)| (label, count))
            .reduce(|_label, records, output| {
                let sum: u32 = records.iter().map(|record| record.0).sum();
                output.push((sum, 1));
            });

        (label_counts.probe(), feature_per_label_counts.probe())
    })
}