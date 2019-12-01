extern crate amnesia;
extern crate ndarray;
extern crate rand;

use std::time::Instant;
use rand::Rng;

use amnesia::lsh::ApproximateKnn;
use amnesia::IncrementalDecrementalModel;


fn main() {

    let num_examples_to_forget = 20;

    run_experiment("datasets/mushrooms.libsvm", 112, num_examples_to_forget);
    run_experiment("datasets/phishing.libsvm", 68, num_examples_to_forget);
    run_experiment("datasets/covtype.libsvm", 54, num_examples_to_forget);
}




fn run_experiment(
    dataset_file: &str,
    num_features: usize,
    num_examples_to_forget: usize
) {

    let examples = amnesia::io_utils::read_libsvm_file(dataset_file, num_features);

    let mut knn = ApproximateKnn::new(20, num_features, 32, 10, 2);

    println!("Training full model");
    let start = Instant::now();
    knn.partial_fit(&examples);
    println!("Training took {} ms", start.elapsed().as_millis());

    let mut examples_without = examples.clone();

    let mut rng = rand::thread_rng();
    for _ in 0 .. num_examples_to_forget {

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