use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

use crate::differential::lsh::Sample;
use crate::differential::mnb::CategoricalSample;

pub fn read_interactions(dataset_file: &'static str, num_users: usize) -> Vec<(u32, Vec<u32>)> {

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(dataset_file)
        .expect("Unable to read input file");

    let mut interactions: Vec<(u32, Vec<u32>)> = (0 .. num_users)
        .map(|index| (index as u32, Vec::new())).collect();

    reader.deserialize()
        .for_each(|result| {
            if result.is_ok() {
                let (user, item): (u32, u32) = result.unwrap();
                let user_idx = user - 1;
                let item_idx = item - 1;

                if interactions[user_idx as usize].1.len() < 500 {
                    interactions[user_idx as usize].1.push(item_idx);
                }
            }
        });

    interactions
}


pub fn read_libsvm_file_for_differential(dataset_file: &str, num_features: usize) -> Vec<Sample> {

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

pub fn read_libsvm_file_as_categorical(
    dataset_file: &str,
    adjust_labels: bool
) -> Vec<CategoricalSample> {

    let mut label_correction = 0;
    if adjust_labels {
        label_correction += 1;
    }

    let mut samples = Vec::new();

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for (index, line) in file.lines().enumerate() {
        let l = line.unwrap();

        let mut features: Vec<u32> = Vec::new();
        let mut label: u8 = 0;

        let tokens = l.split(' ');
        for (index, t) in tokens.enumerate() {
            if index == 0 {
                label = t.parse::<u8>().expect("Unable to parse class label!") - label_correction;
            } else {
                if !t.is_empty() {
                    let tt: Vec<_> = t.split(':').collect();
                    let offset = tt[0].parse::<u32>().expect("Unable to parse offset!") - 1;

                    features.push(offset);
                }
            }
        }

        samples.push(CategoricalSample::new(index as u64, features, label));
    }

    samples
}