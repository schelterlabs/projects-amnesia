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
use differential_dataflow::operators::CountTotal;

use std::hash::Hasher;
use std::cmp::Ordering;
use self::differential_dataflow::operators::arrange::ArrangeByKey;

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
                example.features.into_iter().map(move |feature_index| ((feature_index, label), ()))
            });

        let arranged_features_per_label = features_per_label.arrange_by_key();

        let feature_per_label_counts_probe = arranged_features_per_label
            .count_total()
            .probe();

        let label_counts_probe = features_per_label
            .map(|(_feature_index, label)| label)
            .count_total()
            .probe();

        (label_counts_probe, feature_per_label_counts_probe)
    })
}