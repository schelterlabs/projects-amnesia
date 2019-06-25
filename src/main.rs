extern crate amnesia;
extern crate rand;
extern crate ndarray;

use rand::distributions::Uniform;
use ndarray::{Array,Dim};
use ndarray_rand::RandomExt;

use amnesia::lsh::LshTable;

fn main() {

    let num_samples = 1000;
    let num_features = 100;
    let num_components = 32;

    let samples: Vec<(u32, Array<f64, Dim<[usize; 1]>>)> = (0..num_samples)
        .map(|sample_index| {
            let data = Array::random(num_features, Uniform::new(-1.0, 1.0));
            (sample_index, data)
        })
        .collect();

    let mut table = LshTable::new(num_features, num_components);

    table.partial_fit(&samples);
}
