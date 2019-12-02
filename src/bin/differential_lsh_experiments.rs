extern crate abomonation;
extern crate abomonation_derive;
extern crate amnesia;
extern crate timely;
extern crate differential_dataflow;
extern crate ndarray;
extern crate rand;

use rand::distributions::Normal;
use differential_dataflow::input::InputSession;
use amnesia::differential::lsh::ProjectionMatrix;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;

use std::time::Instant;

fn main() {

    let num_samples_to_forget: usize = std::env::args().nth(2)
        .expect("num_samples_to_forget not specified").parse()
        .expect("Unable to parse num_samples_to_forget");

    run_experiment("datasets/mushrooms.libsvm", 112, num_samples_to_forget);
    run_experiment("datasets/phishing.libsvm", 68, num_samples_to_forget);
    run_experiment("datasets/covtype.libsvm", 54, num_samples_to_forget);
}

fn run_experiment(dataset_file: &'static str, num_features: usize, num_samples_to_forget: usize) {

    timely::execute_from_args(std::env::args(), move |worker| {

        let num_tables = 20;

        let num_hash_dimensions = 32;
        let mut examples_input = InputSession::new();
        let mut tables_input = InputSession::new();

        let probe = amnesia::differential::lsh::lsh(worker, &mut examples_input, &mut tables_input);

        let mut rng = thread_rng();
        let distribution = Normal::new(0.0, 1.0 / num_hash_dimensions as f64);

        let mut tables: Vec<ProjectionMatrix> = Vec::with_capacity(num_tables);

        for table_index in 0..num_tables {
            let mut random_matrix: Vec<f64> =
                Vec::with_capacity(num_hash_dimensions * num_features);

            for _ in 0..(num_hash_dimensions * num_features) {
                let sampled: f64 = rng.sample(distribution);
                random_matrix.push(sampled);
            }
            let projection_matrix = ProjectionMatrix::new(table_index, random_matrix);
            tables.push(projection_matrix);
        }

        let mut samples = amnesia::differential::io_utils::read_libsvm_file_for_differential(
            dataset_file, num_features);

        for projection_matrix in tables.iter() {
            if projection_matrix.table_index as usize % worker.peers() == worker.index() {
                tables_input.insert(projection_matrix.clone());
            }
        }

        for sample in samples.iter() {
            if sample.id as usize % worker.peers() == worker.index() {
                examples_input.insert(sample.clone());
            }
        }

        examples_input.advance_to(1);
        examples_input.flush();
        tables_input.advance_to(1);
        tables_input.flush();

        worker.step_while(|| {
            probe.less_than(examples_input.time()) && probe.less_than(tables_input.time())
        });

        let mut rng = rand::thread_rng();
        let (samples_to_forget, _) = samples.partial_shuffle(&mut rng, num_samples_to_forget);

        let start = Instant::now();

        for sample in samples_to_forget.iter() {
            if sample.id as usize % worker.peers() == worker.index() {
                examples_input.remove(sample.clone());
            }
        }

        examples_input.advance_to(2);
        examples_input.flush();
        tables_input.advance_to(2);
        tables_input.flush();

        worker.step_while(|| {
            probe.less_than(examples_input.time()) && probe.less_than(tables_input.time())
        });

        let forgetting_duration = start.elapsed();

        if worker.index() == 0 {
            println!("knn,{},{}", dataset_file, forgetting_duration.as_micros());
        }

    }).unwrap();
}
