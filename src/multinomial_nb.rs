use fnv::FnvHashMap;
use crate::IncrementalDecrementalModel;

pub struct MultinomialNaiveBayes {
    num_labels: u8,
    num_features: u32,
    feature_counts_per_label: Vec<FnvHashMap<u32, u32>>,
    counts_per_label: Vec<u32>,
}

impl MultinomialNaiveBayes {

    pub fn new(num_labels: u8, num_features: u32) -> Self {
        let mut feature_counts_per_label = Vec::with_capacity(num_labels as usize);
        for _ in 0..num_labels {
            let counts = FnvHashMap::with_capacity_and_hasher(
                num_features as usize, Default::default());
            feature_counts_per_label.push(counts);
        }

        let counts_per_label = vec![0; num_labels as usize];

        MultinomialNaiveBayes {
            num_labels,
            num_features,
            feature_counts_per_label,
            counts_per_label
        }
    }
}

#[derive(Clone)]
pub struct MNBFeatures {
    pub features: FnvHashMap<u32, u32>,
}

impl MNBFeatures {
    pub fn new(features: FnvHashMap<u32, u32>) -> Self {
        MNBFeatures { features }
    }
}

impl IncrementalDecrementalModel<(MNBFeatures, u8), MNBFeatures, u8> for MultinomialNaiveBayes {

    fn partial_fit(self: &mut Self, data: &[(MNBFeatures, u8)]) {
        for example in data.iter() {
            let (features, label) = example;
            let label_index = *label as usize;
            for (feature_index, count) in features.features.iter() {
                *self.feature_counts_per_label[label_index]
                    .entry(*feature_index).or_insert(0) += *count;
                self.counts_per_label[label_index] += *count;
            }
        }
    }

    fn forget(self: &mut Self, data: &(MNBFeatures, u8)) {
        let (features, label) = data;
        let label_index = *label as usize;
        for (feature_index, count) in features.features.iter() {
            *self.feature_counts_per_label[label_index].get_mut(feature_index).unwrap() -= *count;
            self.counts_per_label[label_index] -= *count;
        }
    }

    fn predict(self: &Self, data: &MNBFeatures) -> u8 {

        let mut predicted_label: Option<u8> = None;
        let mut predicted_label_log_prob: Option<f64> = None;

        for label in 0..self.num_labels {
            let mut log_prob = 0.0;
            let label_index = label as usize;
            let nc = self.counts_per_label[label_index];
            for (feature_index, count) in data.features.iter() {
                let nci = self.feature_counts_per_label[label_index].get(feature_index).unwrap();
                log_prob += *count as f64 * ((nci + 1) as f64 / (nc + self.num_features) as f64).ln()
            }

            match predicted_label_log_prob {
                None => {
                    predicted_label = Some(label);
                    predicted_label_log_prob = Some(log_prob);
                }

                Some(current_log_prob) if current_log_prob < log_prob => {
                    predicted_label = Some(label);
                    predicted_label_log_prob = Some(log_prob);
                }

                _ => {}
            }
        }

        predicted_label.unwrap()
    }
}