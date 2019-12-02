use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

use crate::lsh::Example;
use ndarray::Array1;


use crate::mnb::MNBFeatures;
use fnv::FnvHashMap;

pub fn read_libsvm_file(dataset_file: &str, num_features: usize) -> Vec<Example> {

    let mut examples = Vec::new();

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for line in file.lines() {
        let l = line.unwrap();

        let mut features = Array1::<f64>::zeros(num_features);
        let mut label: u8 = 0;

        let tokens = l.split(' ');
        for (index, t) in tokens.enumerate() {
            if index == 0 {
                label = t.parse::<u8>().expect("Unable to parse class label!");
            } else {
                if !t.is_empty() {
                    let tt: Vec<_> = t.split(':').collect();
                    let offset = tt[0].parse::<usize>().expect("Unable to parse offset!") - 1;
                    let value = tt[1].parse::<f64>().expect("Unable to parse value!");

                    features[offset] = value;
                }
            }
        }

        examples.push(Example::new(features, label));
    }

    examples
}

pub fn read_libsvm_file_for_mnb(dataset_file: &str, adjust_labels: bool) -> Vec<(MNBFeatures, u8)> {

    let mut examples = Vec::new();

    let mut label_correction = 0;
    if adjust_labels {
        label_correction += 1;
    }

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for line in file.lines() {
        let l = line.unwrap();

        let mut features: FnvHashMap<u32, u32> = FnvHashMap::default();
        let mut label: u8 = 0;

        let tokens = l.split(' ');
        for (index, t) in tokens.enumerate() {
            if index == 0 {
                label = t.parse::<u8>().expect("Unable to parse class label!") - label_correction;
            } else {
                if !t.is_empty() {
                    let tt: Vec<_> = t.split(':').collect();
                    let offset = tt[0].parse::<usize>().expect("Unable to parse offset!") - 1;
                    let value = tt[1].parse::<f64>().expect("Unable to parse value!");

                    features.insert(offset as u32, value as u32);
                }
            }
        }

        examples.push((MNBFeatures::new(features), label));
    }

    examples
}