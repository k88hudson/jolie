use crate::distributions::traits::*;

/// Numerically differentiates CDF and checks it approximates PDF.
pub fn assert_cdf_derivative_approx_pdf<D>(dist: &D, points: &[f64], h: f64, tol: f64)
where
    D: UnivariateContinuous<f64>,
{
    for &x in points {
        let numerical_pdf = (dist.cdf(x + h) - dist.cdf(x - h)) / (2.0 * h);
        let analytic_pdf = dist.pdf(&x);
        assert!(
            (numerical_pdf - analytic_pdf).abs() < tol,
            "CDF derivative != PDF at x={x}: numerical={numerical_pdf}, analytic={analytic_pdf}"
        );
    }
}
