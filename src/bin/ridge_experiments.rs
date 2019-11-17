extern crate amnesia;
extern crate rgsl;
extern crate rand;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::time::Instant;

use rgsl::{MatrixF64, VectorF64};

use amnesia::IncrementalDecrementalModel;
use amnesia::ridge::{RidgeRegression, Example};

use rand::Rng;

fn read_libsvm_file(dataset_file: &str, num_features: usize) -> Vec<Example> {

    let mut examples = Vec::new();

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for line in file.lines() {
        let l = line.unwrap();

        let mut features = vec![0_f64; num_features];
        let mut target: f64 = 0_f64;

        let tokens = l.split(' ');
        for (index, t) in tokens.enumerate() {
            if index == 0 {
                target = t.parse::<f64>().expect("Unable to parse target!");
            } else {
                if !t.is_empty() {
                    let tt: Vec<_> = t.split(':').collect();
                    let offset = tt[0].parse::<usize>().expect("Unable to parse offset!") - 1;
                    let value = tt[1].parse::<f64>().expect("Unable to parse value!");

                    features[offset] = value;
                }
            }
        }

        let feature_vector = VectorF64::from_slice(&features).unwrap();

        examples.push(Example::new(feature_vector, target));
    }

    examples
}

fn read_libsvm_file_matrix_vector(
    dataset_file: &str,
    num_examples: usize,
    num_features: usize
) -> (MatrixF64, VectorF64) {

    let mut examples = MatrixF64::new(num_examples, num_features).unwrap();
    let mut targets = VectorF64::new(num_examples).unwrap();

    let handle = File::open(dataset_file).expect("Unable to read file!");

    let file = BufReader::new(&handle);
    for (row, line) in file.lines().enumerate() {
        let l = line.unwrap();

        let tokens = l.split(' ');
        for (index, t) in tokens.enumerate() {
            if index == 0 {
                let target_value = t.parse::<f64>().expect("Unable to parse target!");
                targets.set(index, target_value);
            } else {
                if !t.is_empty() {
                    let tt: Vec<_> = t.split(':').collect();
                    let offset = tt[0].parse::<usize>().expect("Unable to parse offset!") - 1;
                    let value = tt[1].parse::<f64>().expect("Unable to parse value!");

                    examples.set(row, offset, value);
                }
            }
        }
    }

    (examples, targets)
}

fn examples_to_matrix(examples: &[Example], num_features: usize) -> (MatrixF64, VectorF64) {

    let num_examples = examples.len();

    let mut x = MatrixF64::new(num_examples, num_features).unwrap();
    let mut y = VectorF64::new(num_examples).unwrap();

    for (index, example) in examples.iter().enumerate() {
        y.set(index, example.target);
        for offset in 0..num_features {
            x.set(index, offset, example.features.get(offset));
        }
    }

    (x, y)
}


fn main() {

    let num_examples_to_forget = 20;

    run_experiment("datasets/housing_scale.libsvm", 506, 13, num_examples_to_forget);
    run_experiment("datasets/cadata.libsvm",  20_640, 8, num_examples_to_forget);
    run_experiment("datasets/YearPredictionMSD.libsvm",  463_715, 90, num_examples_to_forget);
}

fn run_experiment(
    dataset_file: &str,
    num_examples: usize,
    num_features: usize,
    num_examples_to_forget: usize)
{

    let mut examples = read_libsvm_file(dataset_file, num_features);

    let (x, y) = read_libsvm_file_matrix_vector(dataset_file, num_examples, num_features);

    let mut ridge = RidgeRegression::new(x, y, 0.001);

    let mut rng = rand::thread_rng();
    for _ in 0 .. num_examples_to_forget {
        let index = rng.gen_range(0, examples.len());

        let example_to_forget = examples.remove(index);

        let start = Instant::now();
        ridge.forget(&example_to_forget);
        let forgetting_duration = start.elapsed();

        let (x_for_retrain, y_for_retrain) = examples_to_matrix(&examples, num_features);

        let start = Instant::now();
        let _retrained_ridge = RidgeRegression::new(x_for_retrain, y_for_retrain, 0.001);
        let retrain_duration = start.elapsed();

        println!("{},{},{}", dataset_file, forgetting_duration.as_micros(),
            retrain_duration.as_micros());
    }
}