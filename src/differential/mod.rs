extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;
use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;
use differential_dataflow::lattice::Lattice;

use differential_dataflow::operators::{Join,CountTotal,Count};
use differential_dataflow::operators::arrange::ArrangeByKey;
use differential_dataflow::operators::join::JoinCore;

use std::cmp::Ordering;

#[derive(Abomonation, Debug, Clone)]
pub struct Sample {
    pub id: u64,
    pub features: [f64; 10],
    pub label: u8,
}

impl Sample {
    pub fn new(id: u64, features: [f64; 10], label: u8) -> Sample {
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
        self.id.cmp(&other.id)
    }
}

pub fn lsh<T>(
    worker: &mut Worker<Allocator>,
    examples_input: &mut InputSession<T, Sample, isize>)
-> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    let probe = worker.dataflow(|scope| {
        let examples = examples_input.to_collection(scope);
        examples
            //.map(|sample| (sample.id, sample))
            //.count()
            .filter(|s| s.id < 100)
            .inspect(|x| println!("{:?}", x))
            .probe()
    });

    probe
}

pub fn itembased_cf<T>(
    worker: &mut Worker<Allocator>,
    interactions_input: &mut InputSession<T, (u32, u32), isize>)
-> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    let probe = worker.dataflow(|scope| {

        let interactions = interactions_input.to_collection(scope);

        // Find all users with less than 500 interactions
        let users_with_enough_interactions = interactions
            .map(|(user, _item)| user)
            .count_total()
            .filter(move |(_user, count): &(u32, isize)| *count < 500)
            .map(|(user, _count)| user);

        // Remove users with too many interactions
        let remaining_interactions = interactions
            .semijoin(&users_with_enough_interactions);

        let num_interactions_per_item = remaining_interactions
            .map(|(_user, item)| item)
            .count_total();

        let arranged_remaining_interactions = remaining_interactions.arrange_by_key();

        // Compute the number of cooccurrences of each item pair
        let cooccurrences = arranged_remaining_interactions
            .join_core(&arranged_remaining_interactions, |_user, &item_a, &item_b| {
                if item_a > item_b { Some((item_a, item_b)) } else { None }
            })
            .count();

        let arranged_num_interactions_per_item = num_interactions_per_item.arrange_by_key();

        // Compute the jaccard similarity between item pairs (= number of users that interacted
        // with both items / number of users that interacted with at least one of the items)
        let jaccard_similarities = cooccurrences
            // Find the number of interactions for item_a
            .map(|((item_a, item_b), num_cooc)| (item_a, (item_b, num_cooc)))
            .join_core(
                &arranged_num_interactions_per_item,
                |&item_a, &(item_b, num_cooc), &occ_a| Some((item_b, (item_a, num_cooc, occ_a)))
            )
            // Find the number of interactions for item_b
            .join_core(
                &arranged_num_interactions_per_item,
                |&item_b, &(item_a, num_cooc, occ_a), &occ_b| {
                    Some(((item_a, item_b), (num_cooc, occ_a, occ_b)))
                },
            )
            // Compute Jaccard similarity, has to be done in a map due to the lack of a
            // total order for f64 (which seems to break the consolidation in join)
            .map(|((item_a, item_b), (num_cooc, occ_a, occ_b))| {
                let jaccard = num_cooc as f64 / (occ_a + occ_b - num_cooc) as f64;
                ((item_a, item_b), jaccard)
            });

        // We threshold the similarity matrix
        let thresholded_similarities = jaccard_similarities
            .filter(|(_item_pair, jaccard)| *jaccard > 0.05);

        thresholded_similarities.probe()
    });

    probe
}