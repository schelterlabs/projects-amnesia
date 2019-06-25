extern crate amnesia;

use amnesia::itembased::ItembasedCF;

fn main() {

    let interactions: Vec<Vec<u32>> = vec![
        vec![0, 1, 2],
        vec![0, 2],
        vec![1, 2]
    ];

    let mut itembased_cf = ItembasedCF::new(3);

    itembased_cf.fit_partial(&interactions);
    itembased_cf.forget(&vec![0, 2]);


    let interactions2: Vec<Vec<u32>> = vec![
        vec![0, 1, 2],
        vec![1, 2]
    ];

    let mut itembased_cf2 = ItembasedCF::new(3);
    itembased_cf2.fit_partial(&interactions2);

    println!("{:?}", itembased_cf);
    println!("{:?}", itembased_cf2);
}
