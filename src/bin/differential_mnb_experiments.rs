extern crate timely;
extern crate differential_dataflow;
extern crate amnesia;

use rand::seq::SliceRandom;

use std::time::Instant;

use differential_dataflow::input::InputSession;

fn main() {

    let num_samples_to_forget = 20;

    run_experiment("datasets/mushrooms.libsvm", num_samples_to_forget, true);
    run_experiment("datasets/phishing.libsvm", num_samples_to_forget, false);
    run_experiment("datasets/covtype.libsvm", num_samples_to_forget, true);
}

fn run_experiment(
    dataset_file: &'static str,
    num_samples_to_forget: usize,
    adjust_labels: bool
) {
    timely::execute_from_args(std::env::args(), move |worker| {

        let mut samples_input = InputSession::new();

        let mut samples = amnesia::differential::io_utils::read_libsvm_file_as_categorical(
            dataset_file, adjust_labels);

        let (label_counts_probe, feature_per_label_counts_probe) =
            amnesia::differential::mnb::mnb(worker, &mut samples_input);

        for sample in samples.iter() {
            if sample.id as usize % worker.peers() == worker.index() {
                samples_input.insert(sample.clone());
            }
        }

        samples_input.advance_to(1);
        samples_input.flush();

        worker.step_while(|| {
            label_counts_probe.less_than(samples_input.time()) &&
                feature_per_label_counts_probe.less_than(samples_input.time())
        });

        let mut rng = rand::thread_rng();
        let (samples_to_forget, _) = samples.partial_shuffle(&mut rng, num_samples_to_forget);

        let start = Instant::now();

        for sample in samples_to_forget.iter() {
            if sample.id as usize % worker.peers() == worker.index() {
                samples_input.remove(sample.clone());
            }
        }

        samples_input.advance_to(2);
        samples_input.flush();

        worker.step_while(|| {
            label_counts_probe.less_than(samples_input.time()) &&
                feature_per_label_counts_probe.less_than(samples_input.time())
        });

        let forgetting_duration = start.elapsed();

        println!("{},{},{}", dataset_file, worker.index(), forgetting_duration.as_micros());

    }).unwrap();

}