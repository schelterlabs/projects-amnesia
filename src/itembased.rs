extern crate fnv;

use fnv::{FnvHashMap, FnvHashSet};
use crate::IncrementalDecremental;

#[derive(Debug)]
pub struct ItembasedCF {
    c: Vec<FnvHashMap<u32, u32>>,
    s: Vec<FnvHashMap<u32, f32>>,
    n: Vec<u32>,
}

impl ItembasedCF {

    pub fn new(num_items: usize) -> ItembasedCF {
        let c = vec![FnvHashMap::with_capacity_and_hasher(0, Default::default()); num_items];
        let s = vec![FnvHashMap::with_capacity_and_hasher(0, Default::default()); num_items];
        let n = vec![0; num_items];

        ItembasedCF { c, s, n }
    }
}

impl IncrementalDecremental<Vec<u32>> for ItembasedCF {

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

}

#[cfg(test)]
mod tests {

    use crate::IncrementalDecremental;
    use crate::itembased::ItembasedCF;

    #[test]
    fn toy_example() {
        let interactions: Vec<Vec<u32>> = vec![
            vec![0, 1, 2],
            vec![0, 2],
            vec![1, 2]
        ];

        let mut itembased_cf = ItembasedCF::new(3);

        itembased_cf.partial_fit(&interactions);
        itembased_cf.forget(&vec![0, 2]);


        let interactions2: Vec<Vec<u32>> = vec![
            vec![0, 1, 2],
            vec![1, 2]
        ];

        let mut itembased_cf2 = ItembasedCF::new(3);
        itembased_cf2.partial_fit(&interactions2);

        assert_eq!(itembased_cf.c, itembased_cf2.c);
        assert_eq!(itembased_cf.s, itembased_cf2.s);
        assert_eq!(itembased_cf.n, itembased_cf2.n);
    }
}