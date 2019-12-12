extern crate amnesia;
extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;
use std::time::Instant;
use rand::seq::SliceRandom;
#[allow(deprecated)]
use rand::{SeedableRng,XorShiftRng};

fn main() {

    let num_samples_to_forget: usize = std::env::args().nth(2)
        .expect("num_samples_to_forget not specified").parse()
        .expect("Unable to parse num_samples_to_forget");

    run_experiment("datasets/movielens1m.tsv", 6040, num_samples_to_forget);
    run_experiment("datasets/jester.tsv", 50692, num_samples_to_forget);
    run_experiment("datasets/ciaodvd.tsv", 21019, num_samples_to_forget);
}

fn run_experiment(
    dataset_file: &'static str,
    num_users: usize,
    num_users_to_forget: usize
) {
    timely::execute_from_args(std::env::args(), move |worker| {

        let mut interactions_input = InputSession::new();

        let probe = amnesia::differential::itembased::itembased_cf(worker, &mut interactions_input);

        let mut interactions =
            amnesia::differential::io_utils::read_interactions(dataset_file, num_users);

        for (user, history) in interactions.iter() {
            if *user as usize % worker.peers() == worker.index() {
                for item in history.iter() {
                    interactions_input.insert((*user, *item));
                }
            }
        }

        interactions_input.advance_to(1);
        interactions_input.flush();

        worker.step_while(|| probe.less_than(interactions_input.time()));

        #[allow(deprecated)]
        let mut rng = XorShiftRng::from_seed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16]);

        let (interactions_to_remove, _) =
            interactions.partial_shuffle(&mut rng, num_users_to_forget);

        let start = Instant::now();

        for (user, history) in interactions_to_remove.iter() {
            if *user as usize % worker.peers() == worker.index() {
                for item in history.iter() {
                    interactions_input.remove((*user, *item));
                }
            }
        }

        interactions_input.advance_to(2);
        interactions_input.flush();

        worker.step_while(|| probe.less_than(interactions_input.time()));

        let forgetting_duration = start.elapsed();

        if worker.index() == 0 {
            println!("itembased,{},{}", dataset_file, forgetting_duration.as_micros());
        }
    }).unwrap();
}
