use num_traits::Float;
use rand::{Rng, RngExt};

use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::unchecked::Unchecked;

/// Continuous uniform distribution over the interval `[a, b]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform<F: Float> {
    a: F,
    b: F,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UniformParams<F> {
    pub a: F,
    pub b: F,
}

// Constructors
impl<F: Float> Uniform<F> {
    /// - `a`: lower bound (inclusive), must be finite
    /// - `b`: upper bound (inclusive), must be finite and > `a`
    pub fn new(a: F, b: F) -> Result<Self, DistributionError> {
        if !a.is_finite() || !b.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "a and b must be finite",
            ));
        }
        if a >= b {
            return Err(DistributionError::InvalidParameter("a must be less than b"));
        }
        // Both bounds are finite but their span can still overflow to +inf
        // (e.g. f64::MIN..f64::MAX), which yields pdf == 0 and inf samples.
        if !(b - a).is_finite() {
            return Err(DistributionError::InvalidParameter("b - a must be finite"));
        }
        Ok(Self { a, b })
    }

    pub fn new_unchecked(_: Unchecked, a: F, b: F) -> Self {
        Self { a, b }
    }

    pub fn a(&self) -> F {
        self.a
    }

    pub fn b(&self) -> F {
        self.b
    }
}

impl<F: Float> Sampleable for Uniform<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        self.a + F::from(rng.random::<f64>()).unwrap() * (self.b - self.a)
    }
}

crate::distributions::traits::impl_rand_distribution!(Uniform<F: Float> => F);

impl<F: Float> Distribution<F> for Uniform<F> {
    fn log_pdf(&self, x: &F) -> F {
        if *x >= self.a && *x <= self.b {
            -(self.b - self.a).ln()
        } else {
            F::neg_infinity()
        }
    }
}

impl<F: Float> UnivariateContinuous<F> for Uniform<F> {
    type Params = UniformParams<F>;

    fn cdf(&self, x: F) -> F {
        if x < self.a {
            F::zero()
        } else if x > self.b {
            F::one()
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    fn inverse_cdf(&self, p: F) -> F {
        self.a + p * (self.b - self.a)
    }

    fn support(&self) -> (F, F) {
        (self.a, self.b)
    }

    fn params(&self) -> UniformParams<F> {
        UniformParams {
            a: self.a,
            b: self.b,
        }
    }

    fn from_params(params: UniformParams<F>) -> Result<Self, DistributionError> {
        Self::new(params.a, params.b)
    }
}

impl<F: Float> HasMean for Uniform<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        Some((self.a + self.b) / two)
    }
}

impl<F: Float> HasVariance for Uniform<F> {
    fn variance(&self) -> Option<F> {
        let twelve = F::from(12.0).unwrap();
        let d = self.b - self.a;
        Some(d * d / twelve)
    }
}

impl<F: Float> HasEntropy for Uniform<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        Some((self.b - self.a).ln())
    }
}

impl<F: Float> HasMode for Uniform<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        Some((self.a + self.b) / two)
    }
}

impl<F: Float> HasSkewness for Uniform<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        Some(F::zero())
    }
}

impl<F: Float> HasKurtosis for Uniform<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        Some(F::from(-6.0 / 5.0).unwrap())
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

    // --- Construction ---

    #[test]
    fn new_valid() {
        assert!(Uniform::<f64>::new(0.0, 1.0).is_ok());
        assert!(Uniform::<f64>::new(-100.0, 100.0).is_ok());
        assert!(Uniform::<f32>::new(-1.0, 1.0).is_ok());
    }

    #[test]
    fn new_rejects_equal() {
        assert!(Uniform::<f64>::new(1.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_flipped() {
        assert!(Uniform::<f64>::new(5.0, 2.0).is_err());
    }

    #[test]
    fn new_rejects_nan() {
        assert!(Uniform::<f64>::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::<f64>::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn new_rejects_infinite() {
        assert!(Uniform::<f64>::new(f64::NEG_INFINITY, 1.0).is_err());
        assert!(Uniform::<f64>::new(0.0, f64::INFINITY).is_err());
    }

    // Regression: finite bounds whose span overflows to +inf must be rejected
    // (otherwise pdf == 0 everywhere and every sample is inf).
    #[test]
    fn new_rejects_overflowing_span() {
        assert!(Uniform::<f64>::new(f64::MIN, f64::MAX).is_err());
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = Uniform::<f64>::new_unchecked(Unchecked, 5.0, 2.0);
        assert_eq!(d.a(), 5.0);
        assert_eq!(d.b(), 2.0);
    }

    #[test]
    fn accessors() {
        let d = Uniform::<f64>::new(2.0, 7.0).unwrap();
        assert_eq!(d.a(), 2.0);
        assert_eq!(d.b(), 7.0);
    }

    // --- Reference data: PDF, CDF, quantile, moments (from R) ---

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        run_continuous_reference_tests(|a, b| Uniform::<f64>::new(a, b).unwrap(), &data, 1e-12);
    }

    // --- CDF numerical derivative ≈ PDF ---

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = Uniform::<f64>::new(0.0, 5.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.5, 1.0, 2.5, 4.5], 1e-8, 1e-6);

        let d = Uniform::<f64>::new(-10.0, 10.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[-5.0, 0.0, 5.0], 1e-8, 1e-6);
    }

    // --- Internal consistency: CDF + CCDF = 1, log_pdf = ln(pdf), monotonicity ---

    #[test]
    fn internal_consistency() {
        let d = Uniform::<f64>::new(0.0, 5.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 1.0, 2.5, 4.9, 5.0, 6.0], 1e-14);
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Uniform::<f64>::new(0.0, 1.0).unwrap();
        let bins: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, (i + 1) as f64 * 0.1))
            .collect();
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Uniform::<f64>::new(0.0, 1.0).unwrap();
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

    // --- Non-finite edge cases ---

    #[test]
    fn non_finite_edge_cases() {
        let d = Uniform::<f64>::new(0.0, 1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    // --- Sampling: range and fill ---

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = Uniform::<f64>::new(2.0, 5.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v >= 2.0 && v <= 5.0, "sample {v} out of range [2, 5]");
        }
    }

    #[test]
    fn samples_in_range_f32() {
        let mut rng = test_rng();
        let d = Uniform::<f32>::new(-1.0, 1.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v >= -1.0 && v <= 1.0, "sample {v} out of range [-1, 1]");
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Uniform::<f64>::new(0.0, 1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    // --- Inverse CDF ---

    #[test]
    fn quantile_inverts_cdf() {
        let d = Uniform::<f64>::new(3.0, 17.0).unwrap();
        for &x in &[3.5, 7.0, 10.0, 16.9] {
            let p = d.cdf(x);
            assert!((d.inverse_cdf(p) - x).abs() < 1e-12);
        }
    }

    #[test]
    fn from_params_round_trip() {
        let d = Uniform::<f64>::new(1.0, 5.0).unwrap();
        let d2 = Uniform::from_params(d.params()).unwrap();
        assert_eq!(d.a(), d2.a());
        assert_eq!(d.b(), d2.b());
    }
}
