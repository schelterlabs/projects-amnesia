extern crate amnesia;
extern crate csv;
extern crate rand;

use std::time::Instant;

use amnesia::IncrementalDecrementalModel;
use amnesia::itembased::ItembasedCF;

use rand::Rng;

fn main() {
    let num_users_to_forget = 20;

    run_experiment("datasets/movielens1m.tsv", 6040, 3706, num_users_to_forget);
    run_experiment("datasets/jester.tsv", 50692, 140, num_users_to_forget);
    run_experiment("datasets/ciaodvd.tsv", 21019, 71633, num_users_to_forget);

}

fn run_experiment(
    dataset_file: &str,
    num_users: usize,
    num_items: usize,
    num_users_to_forget: usize
) {

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(dataset_file)
        .expect("Unable to read input file");

    let mut interactions: Vec<Vec<u32>> = (0 .. num_users).map(|_| Vec::new()).collect();

    reader.deserialize()
        .for_each(|result| {
            if result.is_ok() {
                let (user, item): (u32, u32) = result.unwrap();
                let user_idx = user - 1;
                let item_idx = item - 1;

                if interactions[user_idx as usize].len() < 500 {
                    interactions[user_idx as usize].push(item_idx);
                }
            }
        });

    let mut itembased_cf = ItembasedCF::new(num_items, 10);

    println!("Training full model");
    let start = Instant::now();
    itembased_cf.partial_fit(&interactions);
    println!("Training took {} ms", start.elapsed().as_millis());

    let mut interactions_without_users = interactions.clone();

    for _ in 0 .. num_users_to_forget {
        let mut rng = rand::thread_rng();
        let user = rng.gen_range(0, num_users);

        let user_interactions = &interactions[user];

        let start = Instant::now();
        itembased_cf.forget(user_interactions);
        let forgetting_duration = start.elapsed();

        interactions_without_users[user].clear();

        let mut itembased_cf_without_user = ItembasedCF::new(num_items, 10);

        let start = Instant::now();
        itembased_cf_without_user.partial_fit(&interactions_without_users);
        let retraining_duration = start.elapsed();

        println!("{},{},{},{},{}", dataset_file, user, user_interactions.len(),
            forgetting_duration.as_millis(), retraining_duration.as_millis());
    }


}
