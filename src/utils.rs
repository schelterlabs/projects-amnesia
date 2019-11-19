use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

use crate::lsh::Example;
use ndarray::Array1;

use crate::differential::lsh::Sample;

pub fn read_libsvm_file_for_differential(dataset_file: &str, num_features: usize,) -> Vec<Sample> {

    let mut samples = Vec::new();

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for (index, line) in file.lines().enumerate() {
        let l = line.unwrap();

        let mut features: Vec<f64> = vec![0.0; num_features];
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

        samples.push(Sample::new(index as u64, features, label));
    }

    samples
}

pub fn read_libsvm_file(dataset_file: &str, num_features: usize,) -> Vec<Example> {

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