extern crate amnesia;
extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let mut interactions_input = InputSession::new();

        let probe = amnesia::differential::itembased::itembased_cf(worker, &mut interactions_input);

        let interactions: Vec<(u32, u32)> = vec![
            (0, 0), (0, 1), (0, 2),
            (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 3)
        ];

        for (user, item) in interactions.iter() {
            if *user as usize % worker.peers() == worker.index() {
                interactions_input.insert((*user, *item));
            }
        }

        interactions_input.advance_to(1);
        interactions_input.flush();

        worker.step_while(|| probe.less_than(interactions_input.time()));

        let interactions_to_remove: Vec<(u32, u32)> = vec![(1, 1), (1, 2)];

        for (user, item) in interactions_to_remove.iter() {
            if *user as usize % worker.peers() == worker.index() {
                interactions_input.remove((*user, *item));
            }
        }

        interactions_input.advance_to(2);
        interactions_input.flush();

        worker.step_while(|| probe.less_than(interactions_input.time()));


    }).unwrap();
}
