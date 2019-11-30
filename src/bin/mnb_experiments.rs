extern crate amnesia;
extern crate rand;

use std::time::Instant;
use rand::Rng;
use amnesia::multinomial_nb::MultinomialNaiveBayes;
use amnesia::IncrementalDecrementalModel;

fn main() {
    let num_examples_to_forget = 20;
    run_experiment("datasets/mushrooms.libsvm", 112, num_examples_to_forget, 2, true);
    run_experiment("datasets/phishing.libsvm", 68, num_examples_to_forget, 2, false);
    run_experiment("datasets/covtype.libsvm", 54, num_examples_to_forget, 7, true);
}

fn run_experiment(
    dataset_file: &str,
    num_features: u32,
    num_examples_to_forget: usize,
    num_labels: u8,
    adjust_labels: bool)
{
    let examples = amnesia::utils::read_libsvm_file_for_mnb(&dataset_file, adjust_labels);

    let mut mnb = MultinomialNaiveBayes::new(num_labels, num_features);

    println!("Training full model");
    let start = Instant::now();
    mnb.partial_fit(&examples);
    println!("Training took {} ys", start.elapsed().as_micros());

    let mut examples_without = examples.clone();

    let mut rng = rand::thread_rng();
    for _ in 0 .. num_examples_to_forget {

        let example = rng.gen_range(0, examples_without.len());

        let to_forget = examples_without.remove(example);

        let start = Instant::now();
        mnb.forget(&to_forget);
        let forgetting_duration = start.elapsed();

        let mut mnb_without_example = MultinomialNaiveBayes::new(num_labels, num_features);
        let start = Instant::now();
        mnb_without_example.partial_fit(&examples_without);
        let retraining_duration = start.elapsed();

        println!("{},{},{}", dataset_file, forgetting_duration.as_micros(),
            retraining_duration.as_micros());
    }

}