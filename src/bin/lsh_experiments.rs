extern crate amnesia;
extern crate ndarray;
extern crate rand;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::time::Instant;

use ndarray::Array1;
use amnesia::lsh::Example;
use amnesia::lsh::ApproximateKnn;
use amnesia::IncrementalDecrementalModel;

use rand::Rng;

fn main() {

    let num_examples_to_forget = 20;

    run_experiment("datasets/mushrooms.libsvm", 112, num_examples_to_forget);
    //run_experiment("datasets/phishing.libsvm", 68, num_examples_to_forget);
    //run_experiment("datasets/covtype.libsvm", 54, num_examples_to_forget);

}


fn read_libsvm_file(dataset_file: &str, num_features: usize,) -> Vec<Example> {

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

fn run_experiment(
    dataset_file: &str,
    num_features: usize,
    num_examples_to_forget: usize
) {

    let examples = read_libsvm_file(dataset_file, num_features);

    let mut knn = ApproximateKnn::new(20, num_features, 32, 10, 2);

    println!("Training full model");
    let start = Instant::now();
    knn.partial_fit(&examples);
    println!("Training took {} ms", start.elapsed().as_millis());

    let mut examples_without = examples.clone();

    for _ in 0 .. num_examples_to_forget {
        let mut rng = rand::thread_rng();
        let example = rng.gen_range(0, examples_without.len());

        let to_forget = examples_without.remove(example);

        let start = Instant::now();
        knn.forget(&to_forget);
        let forgetting_duration = start.elapsed();

        let mut knn_without_example = ApproximateKnn::new(20, num_features, 32, 10, 2);
        let start = Instant::now();
        knn_without_example.partial_fit(&examples_without);
        let retraining_duration = start.elapsed();

        println!("{}\t{}\t{}", dataset_file, forgetting_duration.as_micros(),
            retraining_duration.as_micros());
    }
}