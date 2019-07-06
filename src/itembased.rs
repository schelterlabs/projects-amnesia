extern crate fnv;

use std::cmp::Ordering;
use fnv::{FnvHashMap, FnvHashSet};
use std::collections::BinaryHeap;

use crate::IncrementalDecrementalModel;

#[derive(Debug)]
pub struct ItembasedCF {
    k: usize,
    c: Vec<FnvHashMap<u32, u32>>,
    s: Vec<FnvHashMap<u32, f32>>,
    n: Vec<u32>,
}

impl ItembasedCF {

    pub fn new(num_items: usize, k: usize) -> ItembasedCF {
        let c = vec![FnvHashMap::with_capacity_and_hasher(0, Default::default()); num_items];
        let s = vec![FnvHashMap::with_capacity_and_hasher(0, Default::default()); num_items];
        let n = vec![0; num_items];

        ItembasedCF { k, c, s, n }
    }
}

impl IncrementalDecrementalModel<Vec<u32>, u32, FnvHashSet<u32>> for ItembasedCF {

    fn partial_fit(&mut self, interactions: &[Vec<u32>]) {

        let mut items_to_rescore = FnvHashSet::with_capacity_and_hasher(0, Default::default());

        // Update cooccurrence matrix
        for user_history in interactions.iter() {
            for item_a in user_history.iter() {
                // Remember item for rescoring later
                items_to_rescore.insert(*item_a);
                self.n[*item_a as usize] += 1;
                for item_b in user_history.iter() {
                    if item_a > item_b {
                        // Update cooccurrence matrix
                        *self.c[*item_a as usize].entry(*item_b).or_insert(0) += 1;
                        *self.c[*item_b as usize].entry(*item_a).or_insert(0) += 1;
                    }
                }
            }
        }

        // Update similarity matrix
        for item_a in items_to_rescore.iter() {
            let n_a = self.n[*item_a as usize] as f32;
            for (item_b, count) in self.c[*item_a as usize].iter() {
                let n_b = self.n[*item_b as usize] as f32;
                let similarity = (*count as f32) / (n_a + n_b - *count as f32);
                self.s[*item_a as usize].insert(*item_b, similarity);
                self.s[*item_b as usize].insert(*item_a, similarity);
            }
        }
    }

    fn forget(&mut self, user_history: &Vec<u32>) {

        // Update cooccurrence matrix
        for item_a in user_history.iter() {
            self.n[*item_a as usize] -= 1;
            for item_b in user_history.iter() {
                if item_a > item_b {
                    // Update cooccurrence matrix
                    *self.c[*item_a as usize].get_mut(item_b).unwrap() -= 1;
                    *self.c[*item_b as usize].get_mut(item_a).unwrap() -= 1;
                }
            }
        }

        // Update similarity matrix
        for item_a in user_history.iter() {
            let n_a = self.n[*item_a as usize] as f32;
            for (item_b, count) in self.c[*item_a as usize].iter() {
                let n_b = self.n[*item_b as usize] as f32;
                let similarity = (*count as f32) / (n_a + n_b - *count as f32);
                self.s[*item_a as usize].insert(*item_b, similarity);
                self.s[*item_b as usize].insert(*item_a, similarity);
            }
        }
    }

    fn predict(&self, item: &u32) -> FnvHashSet<u32> {
        // We'll use a heap to keep track of the current top-n scored items
        let mut top_items = BinaryHeap::with_capacity(self.k);

        for (other_item, similarity) in self.s[*item as usize].iter() {
            let scored_item = ScoredItem { item: *other_item, score: *similarity };

            if top_items.len() < self.k {
                top_items.push(scored_item);
            } else {
                let mut top = top_items.peek_mut().unwrap();
                if scored_item < *top {
                    *top = scored_item;
                }
            }
        }

        let top_k_items: FnvHashSet<u32> = top_items
            .drain()
            .map(|scored_item| scored_item.item)
            .collect();

        top_k_items
    }
}



/// Result type used to find the top-k anomalous items per item via a binary heap
#[derive(PartialEq, Debug)]
struct ScoredItem {
    pub item: u32,
    pub score: f32,
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(scored_item_a: &ScoredItem, scored_item_b: &ScoredItem) -> Ordering {
    match scored_item_a.score.partial_cmp(&scored_item_b.score) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        None => Ordering::Equal
    }
}

impl Eq for ScoredItem {}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}


#[cfg(test)]
mod tests {

    use crate::IncrementalDecrementalModel;
    use crate::itembased::ItembasedCF;

    #[test]
    fn toy_example() {
        let interactions: Vec<Vec<u32>> = vec![
            vec![0, 1, 2],
            vec![0, 2],
            vec![1, 2]
        ];

        let mut itembased_cf = ItembasedCF::new(3, 2);

        itembased_cf.partial_fit(&interactions);
        itembased_cf.forget(&vec![0, 2]);


        let interactions2: Vec<Vec<u32>> = vec![
            vec![0, 1, 2],
            vec![1, 2]
        ];

        let mut itembased_cf2 = ItembasedCF::new(3, 2);
        itembased_cf2.partial_fit(&interactions2);

        assert_eq!(itembased_cf.c, itembased_cf2.c);
        assert_eq!(itembased_cf.s, itembased_cf2.s);
        assert_eq!(itembased_cf.n, itembased_cf2.n);
    }
}