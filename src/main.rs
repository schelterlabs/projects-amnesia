extern crate abomonation;
extern crate abomonation_derive;
extern crate amnesia;
extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;
use amnesia::differential::lsh::{Sample, ProjectionMatrix};

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let mut examples_input = InputSession::new();
        let mut tables_input = InputSession::new();

        let probe = amnesia::differential::lsh::lsh(worker, &mut examples_input, &mut tables_input);

        let tables: Vec<ProjectionMatrix> = vec![
            ProjectionMatrix::new(1, 1234),
            ProjectionMatrix::new(2, 5678),
        ];

        let samples: Vec<Sample> = vec![
            Sample::new(1, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0),
            Sample::new(2, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 1),
        ];

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
