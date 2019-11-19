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

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let dataset_file = "datasets/mushrooms.libsvm";
        let num_features = 112;
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

        let samples = amnesia::utils::read_libsvm_file_for_differential(dataset_file, num_features);

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

    }).unwrap();
}
