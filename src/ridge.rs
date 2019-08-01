extern crate rgsl;

use crate::IncrementalDecrementalModel;

use rgsl::MatrixF64;
use rgsl::VectorF64;

use rgsl::linear_algebra::{QR_decomp, QR_QRsolve, QR_update, QR_unpack};
use rgsl::blas::level1::{daxpy, ddot};
use rgsl::blas::level3::dgemm;
use rgsl::blas::level2::dgemv;
use rgsl::cblas::Transpose::{Trans,NoTrans};


#[derive(Debug)]
pub struct Example {
    pub features: VectorF64,
    pub target: f64,
}

impl Example {
    pub fn new(features: VectorF64, target: f64) -> Example {
        Example { features, target }
    }
}

impl Clone for Example {
    fn clone(&self) -> Self {
        let cloned_features = self.features.clone().expect("Unable to clone!");
        Example::new(cloned_features, self.target)
    }
}

#[derive(Debug)]
pub struct RidgeRegression {
    q: MatrixF64,
    r: MatrixF64,
    z: VectorF64,
    weights: VectorF64,
}

impl RidgeRegression {

    //TODO add lambda I
    pub fn new(x: MatrixF64, y: VectorF64) -> Self {

        let num_features = x.size2();

        let mut xtx = MatrixF64::new(num_features, num_features).expect("Unable to allocate X'X");
        let mut z = VectorF64::new(num_features).expect("Unable to allocate X'y");

        dgemm(Trans, NoTrans, 1.0, &x, &x, 0.0, &mut xtx);
        dgemv(Trans, 1.0, &x, &y, 1.0, &mut z);

        let mut tau = VectorF64::new(num_features).expect("Unable to allocate tau");

        QR_decomp(&mut xtx, &mut tau);

        let qr = xtx;

        let mut q = MatrixF64::new(num_features, num_features).expect("Unable to allocate Q");
        let mut r = MatrixF64::new(num_features, num_features).expect("Unable to allocate R");

        QR_unpack(&qr, &tau, &mut q, &mut r);

        let mut weights = VectorF64::new(num_features).expect("Unable to allocate tau");
        QR_QRsolve(&mut q, &mut r, &z, &mut weights);

        RidgeRegression { q, r, z, weights }
    }
}

impl IncrementalDecrementalModel<Example, VectorF64, f64> for RidgeRegression {

    fn partial_fit(&mut self, data: &[Example]) {
        //TODO could me more performant to batch updates and recompute QR once
        for example in data.iter() {
            let mut w = VectorF64::new(example.features.len()).expect("Unable to allocate w");

            dgemv(Trans, 1.0, &self.q, &example.features, 1.0, &mut w);
            QR_update(&mut self.q, &mut self.r, w, &example.features);
            daxpy(example.target, &example.features, &mut self.z);
        }

        QR_QRsolve(&mut self.q, &mut self.r, &self.z, &mut self.weights);
    }

    fn forget(&mut self, example: &Example) {
        let mut w = VectorF64::new(example.features.len()).expect("Unable to allocate w");

        dgemv(Trans, -1.0, &self.q, &example.features, 1.0, &mut w);
        QR_update(&mut self.q, &mut self.r, w, &example.features);
        daxpy(-1.0 * example.target, &example.features, &mut self.z);
        QR_QRsolve(&mut self.q, &mut self.r, &self.z, &mut self.weights);
    }

    fn predict(&self, features: &VectorF64) -> f64 {
        let mut y_hat = 0_f64;
        ddot(&features, &self.weights, &mut y_hat);
        y_hat
    }
}

#[cfg(test)]
mod tests {

    extern crate rgsl;

    use rgsl::{VectorF64, MatrixView, VectorView};

    use crate::ridge::RidgeRegression;
    use crate::ridge::Example;
    use crate::IncrementalDecrementalModel;

    #[test]
    fn forget_one() {

        let mut x = MatrixView::from_array(&mut [
                0.130010, -0.223675,
                -0.504190, -0.223675,
                0.502476, -0.223675,
                -0.735723, -1.537767,
                1.257476,  1.090417],
            5, 2);

        let mut y = VectorView::from_array(
            &mut [0.475747, -0.084074, 0.228626, -0.867025,  1.595389]
        );


        let mut ridge = RidgeRegression::new(x.matrix(), y.vector());

        let example_to_forget = Example::new(
            VectorView::from_array(&mut [1.257476,  1.090417]).vector(),
            1.595389
        );

        ridge.forget(&example_to_forget);

        println!("RIDGE FORGET\n{:?}", ridge);

        let mut x2 = MatrixView::from_array(&mut [
            0.130010, -0.223675,
            -0.504190, -0.223675,
            0.502476, -0.223675,
            -0.735723, -1.537767],
            4, 2);

        let mut y2 = VectorView::from_array(
            &mut [0.475747, -0.084074, 0.228626, -0.867025]
        );

        let mut ridge2 = RidgeRegression::new(x2.matrix(), y2.vector());

        println!("RIDGE RETRAIN\n{:?}", ridge2);

        //TODO check that ridge and ridge2 are close enough
    }


}
