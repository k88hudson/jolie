use num_traits::Float;
use rand::Rng;

use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::erf::{erfc, erfc_inv};
use crate::special::sampling::standard_normal;
use crate::unchecked::Unchecked;

const LN_2PI: f64 = 1.8378770664093453;
const LN_2PI_E: f64 = 2.8378770664093453;

/// Normal (Gaussian) distribution N(μ, σ²). `std_dev = 0` degenerates to a
/// point mass at μ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal<F: Float> {
    mean: F,
    std_dev: F,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalParams<F> {
    pub mean: F,
    pub std_dev: F,
}

impl<F: Float> Normal<F> {
    /// - `mean` (μ): must be finite
    /// - `std_dev` (σ): must be finite and >= 0
    pub fn new(mean: F, std_dev: F) -> Result<Self, DistributionError> {
        if !mean.is_finite() || !std_dev.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "mean and std_dev must be finite",
            ));
        }
        if std_dev < F::zero() {
            return Err(DistributionError::InvalidParameter(
                "std_dev must be non-negative",
            ));
        }
        Ok(Self { mean, std_dev })
    }

    pub fn new_unchecked(_: Unchecked, mean: F, std_dev: F) -> Self {
        Self { mean, std_dev }
    }

    /// Named `mean_param` to avoid conflict with `HasMean::mean`.
    pub fn mean_param(&self) -> F {
        self.mean
    }

    pub fn std_dev(&self) -> F {
        self.std_dev
    }
}

impl<F: Float> Sampleable for Normal<F> {
    type Value = F;

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        self.mean + self.std_dev * F::from(standard_normal(rng)).unwrap()
    }
}

crate::distributions::traits::impl_rand_distribution!(Normal<F: Float> => F);

impl<F: Float> Distribution<F> for Normal<F> {
    fn log_pdf(&self, x: &F) -> F {
        if !x.is_finite() {
            return F::neg_infinity();
        }
        if self.std_dev == F::zero() {
            return if *x == self.mean {
                F::infinity()
            } else {
                F::neg_infinity()
            };
        }
        let z = (*x - self.mean) / self.std_dev;
        let half = F::from(0.5).unwrap();
        -half * F::from(LN_2PI).unwrap() - self.std_dev.ln() - half * z * z
    }
}

impl<F: Float> UnivariateContinuous<F> for Normal<F> {
    type Params = NormalParams<F>;

    fn cdf(&self, x: F) -> F {
        if self.std_dev == F::zero() {
            return if x >= self.mean { F::one() } else { F::zero() };
        }
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
        F::from(0.5).unwrap() * erfc(-(x - self.mean) / (self.std_dev * sqrt2))
    }

    fn inverse_cdf(&self, p: F) -> F {
        if self.std_dev == F::zero() {
            return self.mean;
        }
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
        let two = F::from(2.0).unwrap();
        self.mean - self.std_dev * sqrt2 * erfc_inv(two * p)
    }

    fn support(&self) -> (F, F) {
        (F::neg_infinity(), F::infinity())
    }

    fn params(&self) -> NormalParams<F> {
        NormalParams {
            mean: self.mean,
            std_dev: self.std_dev,
        }
    }

    fn from_params(params: NormalParams<F>) -> Result<Self, DistributionError> {
        Self::new(params.mean, params.std_dev)
    }
}

impl<F: Float> HasMean for Normal<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        Some(self.mean)
    }
}

impl<F: Float> HasVariance for Normal<F> {
    fn variance(&self) -> Option<F> {
        Some(self.std_dev * self.std_dev)
    }
}

impl<F: Float> HasEntropy for Normal<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        Some(F::from(0.5).unwrap() * F::from(LN_2PI_E).unwrap() + self.std_dev.ln())
    }
}

impl<F: Float> HasMode for Normal<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        Some(self.mean)
    }
}

impl<F: Float> HasSkewness for Normal<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        Some(F::zero())
    }
}

impl<F: Float> HasKurtosis for Normal<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        Some(F::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    const REFERENCE_JSON: &str = include_str!("test_reference.json");

    #[test]
    fn new_valid() {
        assert!(Normal::<f64>::new(0.0, 1.0).is_ok());
        assert!(Normal::<f64>::new(-10.0, 0.01).is_ok());
        assert!(Normal::<f64>::new(1000.0, 100.0).is_ok());
        assert!(Normal::<f32>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn new_rejects_non_finite_mean() {
        assert!(Normal::<f64>::new(f64::NAN, 1.0).is_err());
        assert!(Normal::<f64>::new(f64::INFINITY, 1.0).is_err());
        assert!(Normal::<f64>::new(f64::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite_std_dev() {
        assert!(Normal::<f64>::new(0.0, f64::NAN).is_err());
        assert!(Normal::<f64>::new(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn new_accepts_zero_std_dev() {
        assert!(Normal::<f64>::new(0.0, 0.0).is_ok());
    }

    #[test]
    fn new_rejects_negative_std_dev() {
        assert!(Normal::<f64>::new(0.0, -1.0).is_err());
    }

    // Degenerate σ = 0 (point mass at μ).

    #[test]
    fn degenerate_pdf() {
        let d = Normal::<f64>::new(0.0, 0.0).unwrap();
        assert_eq!(d.pdf(&0.0), f64::INFINITY);
        assert_eq!(d.log_pdf(&0.0), f64::INFINITY);
        assert_eq!(d.pdf(&0.5), 0.0);
        assert_eq!(d.pdf(&-0.5), 0.0);
        assert_eq!(d.log_pdf(&1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn degenerate_cdf() {
        let d = Normal::<f64>::new(2.0, 0.0).unwrap();
        assert_eq!(d.cdf(1.9), 0.0);
        assert_eq!(d.cdf(2.0), 1.0);
        assert_eq!(d.cdf(2.1), 1.0);
    }

    #[test]
    fn degenerate_inverse_cdf() {
        let d = Normal::<f64>::new(5.0, 0.0).unwrap();
        assert_eq!(d.inverse_cdf(0.1), 5.0);
        assert_eq!(d.inverse_cdf(0.5), 5.0);
        assert_eq!(d.inverse_cdf(0.99), 5.0);
    }

    #[test]
    fn degenerate_sampling() {
        let mut rng = test_rng();
        let d = Normal::<f64>::new(3.0, 0.0).unwrap();
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 3.0);
        }
    }

    #[test]
    fn degenerate_moments() {
        let d = Normal::<f64>::new(3.0, 0.0).unwrap();
        assert_eq!(d.mean().unwrap(), 3.0);
        assert_eq!(d.variance().unwrap(), 0.0);
        assert_eq!(d.mode().unwrap(), 3.0);
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = Normal::<f64>::new_unchecked(Unchecked, 0.0, -1.0);
        assert_eq!(d.mean_param(), 0.0);
        assert_eq!(d.std_dev(), -1.0);
    }

    #[test]
    fn accessors() {
        let d = Normal::<f64>::new(5.0, 2.0).unwrap();
        assert_eq!(d.mean_param(), 5.0);
        assert_eq!(d.std_dev(), 2.0);
    }

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        // 1e-10: erfc loses a little precision in the tails.
        run_continuous_reference_tests(
            |mean, std_dev| Normal::<f64>::new(mean, std_dev).unwrap(),
            &data,
            1e-10,
        );
    }

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[-2.0, -1.0, 0.0, 1.0, 2.0], 1e-8, 1e-6);

        let d = Normal::<f64>::new(5.0, 2.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[1.0, 3.0, 5.0, 7.0, 9.0], 1e-8, 1e-6);
    }

    #[test]
    fn internal_consistency() {
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        assert_continuous_consistency(&d, &[-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], 1e-14);

        let d = Normal::<f64>::new(5.0, 2.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], 1e-14);
    }

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        let bins: Vec<(f64, f64)> = vec![
            (-4.0, -2.0),
            (-2.0, -1.0),
            (-1.0, -0.5),
            (-0.5, 0.0),
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
        ];
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Normal::<f64>::new(5.0, 2.0).unwrap();
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();

        let expected_mean = d.mean().unwrap();
        let expected_var = d.variance().unwrap();
        let expected_kurt = d.kurtosis().unwrap();

        let sample_mean = samples.iter().sum::<f64>() / n as f64;
        let sample_var = samples
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        let mean_tol = clt_mean_tolerance(expected_var, n);
        let var_tol = clt_variance_tolerance(expected_var, expected_kurt, n);

        assert!(
            (sample_mean - expected_mean).abs() < mean_tol,
            "mean: got {sample_mean}, expected {expected_mean}, tol {mean_tol}"
        );
        assert!(
            (sample_var - expected_var).abs() < var_tol,
            "variance: got {sample_var}, expected {expected_var}, tol {var_tol}"
        );
    }

    #[test]
    fn non_finite_edge_cases() {
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        for _ in 0..10_000 {
            assert!(d.sample(&mut rng).is_finite());
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Normal::<f64>::new(0.0, 1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = Normal::<f64>::new(3.0, 2.0).unwrap();
        for &x in &[-3.0, 0.0, 1.0, 3.0, 5.0, 9.0] {
            let p = d.cdf(x);
            assert!(
                (d.inverse_cdf(p) - x).abs() < 1e-8,
                "inverse_cdf(cdf({x})) = {}, expected {x}",
                d.inverse_cdf(p)
            );
        }
    }

    #[test]
    fn symmetry() {
        let d = Normal::<f64>::new(3.0, 2.0).unwrap();
        let mu = d.mean_param();
        for &offset in &[0.5, 1.0, 2.0, 3.0] {
            let pdf_plus = d.pdf(&(mu + offset));
            let pdf_minus = d.pdf(&(mu - offset));
            assert!((pdf_plus - pdf_minus).abs() < 1e-14);

            let cdf_sum = d.cdf(mu + offset) + d.cdf(mu - offset);
            assert!((cdf_sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn from_params_round_trip() {
        let d = Normal::<f64>::new(1.0, 2.0).unwrap();
        let d2 = Normal::from_params(d.params()).unwrap();
        assert_eq!(d.mean_param(), d2.mean_param());
        assert_eq!(d.std_dev(), d2.std_dev());
    }
}
