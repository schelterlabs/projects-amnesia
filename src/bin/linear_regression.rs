extern crate rgsl;

use rgsl::MatrixF64;
use rgsl::VectorF64;
use rgsl::linear_algebra::{QR_decomp, QR_lssolve, QR_solve};

use rgsl::blas::level3::dgemm;
use rgsl::blas::level2::dgemv;

use rgsl::cblas::Transpose::{Trans, NoTrans};

fn main() {

    let mut a = MatrixF64::new(5, 2).expect("Failed to create matrix");
    let mut b = VectorF64::new(5).expect("Failed to create vector");

    /*
0.130010 -0.223675  0.475747
1 -0.504190 -0.223675 -0.084074
2  0.502476 -0.223675  0.228626
3 -0.735723 -1.537767 -0.867025
4  1.257476  1.090417  1.595389 */


    a.set(0, 0, 0.130010 ); a.set(0, 1, -0.223675);
    a.set(1, 0, -0.504190); a.set(1, 1, -0.223675);
    a.set(2, 0, 0.502476);  a.set(2, 1, -0.223675);
    a.set(3, 0, -0.735723); a.set(3, 1, -1.537767);
    a.set(4, 0, 1.257476); a.set(4, 1, 1.090417);

    b.set(0, 0.475747);
    b.set(1, -0.084074);
    b.set(2, 0.228626);
    b.set(3, -0.867025);
    b.set(4, 1.595389);

    let mut ata = MatrixF64::new(2, 2).expect("Unable to create matrix");
    let mut atb = VectorF64::new(2).expect("Unable to create matrix");

    dgemm(Trans, NoTrans, 1.0, &a, &a, 0.0, &mut ata);
    dgemv(Trans, 1.0, &a, &b, 1.0, &mut atb);
    println!("b\n{:?}\n", b);
    println!("A'b\n{:?}\n", atb);




    println!("A\n{:?}\n", a);
//    println!("b\n{:?}\n", b);

    let mut tau = VectorF64::new(2).expect("Failed to create vector");

    QR_decomp(&mut a, &mut tau);

//    println!("Q\n{:?}\n", q);
//    println!("tau\n{:?}\n", tau);

    let mut x = VectorF64::new(2).expect("Failed to create vector");
    let mut residual = VectorF64::new(5).expect("Failed to create vector");

    QR_lssolve(&a, &tau, &b, &mut x, &mut residual);


    println!("x\n{:?}\n", x);

    let mut x2 = VectorF64::new(2).expect("Failed to create vector");

    let mut tau2 = VectorF64::new(2).expect("Failed to create vector");
    QR_decomp(&mut ata, &mut tau2);

    QR_solve(&ata, &tau2, &atb, &mut x2);

    println!("x\n{:?}\n", x2);


    //QR_decomp()
    //https://docs.rs/GSL/0.4.31/rgsl/linear_algebra/fn.QR_update.html*/
}