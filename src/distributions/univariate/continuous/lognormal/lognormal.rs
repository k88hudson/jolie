use num_traits::Float;
use rand::Rng;

use crate::constants::LN_2PI;
use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::erf::{erfc, erfc_inv};
use crate::special::sampling::standard_normal;
use crate::unchecked::Unchecked;

/// Log-normal distribution LogNormal(μ, σ): if `X ~ LogNormal(μ, σ)` then
/// `ln X ~ N(μ, σ²)`. `sigma = 0` degenerates to a point mass at exp(μ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal<F: Float> {
    mu: F,
    sigma: F,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogNormalParams<F> {
    pub mu: F,
    pub sigma: F,
}

impl<F: Float> LogNormal<F> {
    /// - `mu` (μ): log-space location, must be finite
    /// - `sigma` (σ): log-space scale, must be finite and >= 0
    pub fn new(mu: F, sigma: F) -> Result<Self, DistributionError> {
        if !mu.is_finite() || !sigma.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "mu and sigma must be finite",
            ));
        }
        if sigma < F::zero() {
            return Err(DistributionError::InvalidParameter(
                "sigma must be non-negative",
            ));
        }
        Ok(Self { mu, sigma })
    }

    pub fn new_unchecked(_: Unchecked, mu: F, sigma: F) -> Self {
        Self { mu, sigma }
    }

    pub fn mu(&self) -> F {
        self.mu
    }

    pub fn sigma(&self) -> F {
        self.sigma
    }
}

impl<F: Float> Sampleable for LogNormal<F> {
    type Value = F;

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        (self.mu + self.sigma * F::from(standard_normal(rng)).unwrap()).exp()
    }
}

crate::distributions::traits::impl_rand_distribution!(LogNormal<F: Float> => F);

impl<F: Float> Distribution<F> for LogNormal<F> {
    fn log_pdf(&self, x: &F) -> F {
        if *x <= F::zero() {
            return F::neg_infinity();
        }
        if self.sigma == F::zero() {
            // Compare in original space; exp(μ).ln() != μ for some μ.
            return if *x == self.mu.exp() {
                F::infinity()
            } else {
                F::neg_infinity()
            };
        }
        // x = +inf yields -inf via IEEE; no explicit is_finite guard needed.
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        let half = F::from(0.5).unwrap();
        -half * F::from(LN_2PI).unwrap() - self.sigma.ln() - ln_x - half * z * z
    }
}

impl<F: Float> UnivariateContinuous<F> for LogNormal<F> {
    type Params = LogNormalParams<F>;

    fn cdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }
        if self.sigma == F::zero() {
            return if x >= self.mu.exp() {
                F::one()
            } else {
                F::zero()
            };
        }
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
        F::from(0.5).unwrap() * erfc(-(x.ln() - self.mu) / (self.sigma * sqrt2))
    }

    fn inverse_cdf(&self, p: F) -> F {
        if self.sigma == F::zero() {
            return self.mu.exp();
        }
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
        let two = F::from(2.0).unwrap();
        (self.mu - self.sigma * sqrt2 * erfc_inv(two * p)).exp()
    }

    fn support(&self) -> (F, F) {
        (F::zero(), F::infinity())
    }

    fn params(&self) -> LogNormalParams<F> {
        LogNormalParams {
            mu: self.mu,
            sigma: self.sigma,
        }
    }

    fn from_params(params: LogNormalParams<F>) -> Result<Self, DistributionError> {
        Self::new(params.mu, params.sigma)
    }
}

impl<F: Float> HasMean for LogNormal<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        let half = F::from(0.5).unwrap();
        Some((self.mu + half * self.sigma * self.sigma).exp())
    }
}

impl<F: Float> HasVariance for LogNormal<F> {
    fn variance(&self) -> Option<F> {
        let sigma2 = self.sigma * self.sigma;
        let two = F::from(2.0).unwrap();
        Some(sigma2.exp_m1() * (two * self.mu + sigma2).exp())
    }
}

impl<F: Float> HasEntropy for LogNormal<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        let half = F::from(0.5).unwrap();
        Some(half + self.sigma.ln() + self.mu + half * F::from(LN_2PI).unwrap())
    }
}

impl<F: Float> HasMode for LogNormal<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        Some((self.mu - self.sigma * self.sigma).exp())
    }
}

impl<F: Float> HasSkewness for LogNormal<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        let expm1_sigma2 = (self.sigma * self.sigma).exp_m1();
        let three = F::from(3.0).unwrap();
        Some((expm1_sigma2 + three) * expm1_sigma2.sqrt())
    }
}

impl<F: Float> HasKurtosis for LogNormal<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        let sigma2 = self.sigma * self.sigma;
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        // exp_m1 form avoids cancellation when σ² is small (all terms ≈ 1).
        let em2 = (sigma2 * two).exp_m1();
        let em3 = (sigma2 * three).exp_m1();
        let em4 = (sigma2 * F::from(4.0).unwrap()).exp_m1();
        Some(em4 + two * em3 + three * em2)
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
        assert!(LogNormal::<f64>::new(0.0, 1.0).is_ok());
        assert!(LogNormal::<f64>::new(-10.0, 0.01).is_ok());
        assert!(LogNormal::<f64>::new(1000.0, 100.0).is_ok());
        assert!(LogNormal::<f32>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn new_rejects_non_finite_mu() {
        assert!(LogNormal::<f64>::new(f64::NAN, 1.0).is_err());
        assert!(LogNormal::<f64>::new(f64::INFINITY, 1.0).is_err());
        assert!(LogNormal::<f64>::new(f64::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite_sigma() {
        assert!(LogNormal::<f64>::new(0.0, f64::NAN).is_err());
        assert!(LogNormal::<f64>::new(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn new_accepts_zero_sigma() {
        assert!(LogNormal::<f64>::new(0.0, 0.0).is_ok());
    }

    #[test]
    fn new_rejects_negative_sigma() {
        assert!(LogNormal::<f64>::new(0.0, -1.0).is_err());
    }

    // Degenerate σ = 0 (point mass at exp(μ)).

    #[test]
    fn degenerate_pdf() {
        let d = LogNormal::<f64>::new(0.0, 0.0).unwrap();
        assert_eq!(d.pdf(&1.0), f64::INFINITY);
        assert_eq!(d.log_pdf(&1.0), f64::INFINITY);
        assert_eq!(d.pdf(&0.5), 0.0);
        assert_eq!(d.pdf(&2.0), 0.0);
        assert_eq!(d.log_pdf(&0.5), f64::NEG_INFINITY);
        assert_eq!(d.log_pdf(&-1.0), f64::NEG_INFINITY);
        assert_eq!(d.log_pdf(&0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn degenerate_cdf() {
        let d = LogNormal::<f64>::new(0.0, 0.0).unwrap();
        assert_eq!(d.cdf(0.5), 0.0);
        assert_eq!(d.cdf(0.99), 0.0);
        assert_eq!(d.cdf(1.0), 1.0);
        assert_eq!(d.cdf(2.0), 1.0);
        assert_eq!(d.cdf(0.0), 0.0);
        assert_eq!(d.cdf(-1.0), 0.0);
    }

    #[test]
    fn degenerate_inverse_cdf() {
        let d = LogNormal::<f64>::new(0.25, 0.0).unwrap();
        let point = 0.25_f64.exp();
        assert_eq!(d.inverse_cdf(0.1), point);
        assert_eq!(d.inverse_cdf(0.5), point);
        assert_eq!(d.inverse_cdf(0.95), point);
    }

    #[test]
    fn degenerate_sampling() {
        let mut rng = test_rng();
        let d = LogNormal::<f64>::new(1.0, 0.0).unwrap();
        let point = 1.0_f64.exp();
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), point);
        }
    }

    #[test]
    fn degenerate_moments() {
        let d = LogNormal::<f64>::new(2.0, 0.0).unwrap();
        let point = 2.0_f64.exp();
        assert_eq!(d.mean().unwrap(), point);
        assert_eq!(d.variance().unwrap(), 0.0);
        assert_eq!(d.mode().unwrap(), point);
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = LogNormal::<f64>::new_unchecked(Unchecked, 0.0, -1.0);
        assert_eq!(d.mu(), 0.0);
        assert_eq!(d.sigma(), -1.0);
    }

    #[test]
    fn accessors() {
        let d = LogNormal::<f64>::new(5.0, 2.0).unwrap();
        assert_eq!(d.mu(), 5.0);
        assert_eq!(d.sigma(), 2.0);
    }

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        run_continuous_reference_tests(
            |mu, sigma| LogNormal::<f64>::new(mu, sigma).unwrap(),
            &data,
            1e-12,
        );
    }

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.5, 1.0, 2.0, 3.0, 5.0], 1e-8, 1e-6);

        let d = LogNormal::<f64>::new(1.0, 0.5).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[1.0, 2.0, 3.0, 5.0, 8.0], 1e-8, 1e-6);
    }

    #[test]
    fn internal_consistency() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert_continuous_consistency(&d, &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 1e-14);

        let d = LogNormal::<f64>::new(3.0, 2.0).unwrap();
        assert_continuous_consistency(&d, &[0.1, 1.0, 5.0, 20.0, 100.0, 500.0], 1e-14);
    }

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        let bins: Vec<(f64, f64)> = vec![
            (0.0, 0.2),
            (0.2, 0.5),
            (0.5, 1.0),
            (1.0, 2.0),
            (2.0, 5.0),
            (5.0, 20.0),
        ];
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = LogNormal::<f64>::new(1.0, 0.5).unwrap();
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
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    // log_pdf has no is_finite guard; pin the IEEE limits at +inf.
    #[test]
    fn density_at_infinity() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert_eq!(d.pdf(&f64::INFINITY), 0.0);
        assert_eq!(d.log_pdf(&f64::INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(
                v > 0.0 && v.is_finite(),
                "sample {v} must be positive and finite"
            );
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v > 0.0 && v.is_finite());
        }
    }

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = LogNormal::<f64>::new(1.0, 0.5).unwrap();
        for &x in &[0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let p = d.cdf(x);
            assert!(
                (d.inverse_cdf(p) - x).abs() < 1e-8,
                "inverse_cdf(cdf({x})) = {}, expected {x}",
                d.inverse_cdf(p)
            );
        }
    }

    #[test]
    fn median_is_exp_mu() {
        let d = LogNormal::<f64>::new(2.0, 1.0).unwrap();
        let median = d.inverse_cdf(0.5);
        let expected = 2.0_f64.exp();
        assert!((median - expected).abs() < 1e-10, "median = {median}");
    }

    #[test]
    fn from_params_round_trip() {
        let d = LogNormal::<f64>::new(1.0, 2.0).unwrap();
        let d2 = LogNormal::from_params(d.params()).unwrap();
        assert_eq!(d.mu(), d2.mu());
        assert_eq!(d.sigma(), d2.sigma());
    }

    #[test]
    fn inverse_cdf_extremes() {
        let d = LogNormal::<f64>::new(0.0, 1.0).unwrap();
        assert_eq!(d.inverse_cdf(0.0), 0.0);
        assert_eq!(d.inverse_cdf(1.0), f64::INFINITY);
    }
}
