use num_traits::Float;
use rand::Rng;

use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::sampling::standard_exponential;
use crate::unchecked::Unchecked;

/// Exponential distribution Exp(θ).
///
/// A one-parameter continuous distribution on the positive reals, parameterized
/// by scale θ > 0. Equivalent to Gamma(1, θ). Has the memoryless property:
/// P(X > s + t | X > s) = P(X > t).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential<F: Float> {
    scale: F,
    inv_scale: F,
}

/// Either a scale (θ) or a rate (λ = 1/θ) parameterization.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ExponentialParams<F> {
    Scale { scale: F },
    Rate { rate: F },
}

/// Serde: `{"scale": 2.0}` or `{"rate": 0.5}`.
/// Presence of `scale` vs `rate` determines the variant.
#[cfg(feature = "serde")]
mod exponential_params_serde {
    use super::ExponentialParams;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize)]
    struct RawRef<'a, F: Serialize> {
        #[serde(skip_serializing_if = "Option::is_none")]
        scale: Option<&'a F>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rate: Option<&'a F>,
    }

    #[derive(Deserialize)]
    struct RawOwned<F> {
        scale: Option<F>,
        rate: Option<F>,
    }

    impl<F: Serialize> Serialize for ExponentialParams<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw = match self {
                ExponentialParams::Scale { scale } => RawRef {
                    scale: Some(scale),
                    rate: None,
                },
                ExponentialParams::Rate { rate } => RawRef {
                    scale: None,
                    rate: Some(rate),
                },
            };
            raw.serialize(serializer)
        }
    }

    impl<'de, F: Deserialize<'de>> Deserialize<'de> for ExponentialParams<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw = RawOwned::<F>::deserialize(deserializer)?;
            match (raw.scale, raw.rate) {
                (Some(scale), None) => Ok(ExponentialParams::Scale { scale }),
                (None, Some(rate)) => Ok(ExponentialParams::Rate { rate }),
                (Some(_), Some(_)) => Err(serde::de::Error::custom(
                    "specify either `scale` or `rate`, not both",
                )),
                (None, None) => Err(serde::de::Error::custom(
                    "must specify either `scale` or `rate`",
                )),
            }
        }
    }
}

impl<F: Float> Exponential<F> {
    /// Constructs an Exponential from scale (θ).
    ///
    /// - `scale` (θ): must be finite and > 0
    pub fn from_scale(scale: F) -> Result<Self, DistributionError> {
        if !scale.is_finite() {
            return Err(DistributionError::InvalidParameter("scale must be finite"));
        }
        if scale <= F::zero() {
            return Err(DistributionError::InvalidParameter(
                "scale must be positive",
            ));
        }
        Ok(Self {
            scale,
            inv_scale: F::one() / scale,
        })
    }

    /// Constructs an Exponential from rate (λ = 1/θ).
    ///
    /// - `rate` (λ): must be finite and > 0
    pub fn from_rate(rate: F) -> Result<Self, DistributionError> {
        if !rate.is_finite() || rate <= F::zero() {
            return Err(DistributionError::InvalidParameter(
                "rate must be finite and positive",
            ));
        }
        Self::from_scale(F::one() / rate)
    }

    pub fn from_scale_unchecked(_: Unchecked, scale: F) -> Self {
        Self {
            scale,
            inv_scale: F::one() / scale,
        }
    }

    pub fn from_rate_unchecked(_: Unchecked, rate: F) -> Self {
        Self::from_scale_unchecked(Unchecked, F::one() / rate)
    }

    pub fn scale(&self) -> F {
        self.scale
    }

    pub fn rate(&self) -> F {
        self.inv_scale
    }
}

impl<F: Float> Sampleable for Exponential<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        // Ziggurat Exp(1) sample scaled by θ
        F::from(standard_exponential(rng)).unwrap() * self.scale
    }
}

crate::distributions::traits::impl_rand_distribution!(Exponential<F: Float> => F);

impl<F: Float> Distribution<F> for Exponential<F> {
    #[inline]
    fn log_pdf(&self, x: &F) -> F {
        if *x < F::zero() || !x.is_finite() {
            return F::neg_infinity();
        }
        // ln(λ) - λx = -ln(θ) - x/θ
        self.inv_scale.ln() - self.inv_scale * *x
    }

    #[inline]
    fn pdf(&self, x: &F) -> F {
        if *x < F::zero() || !x.is_finite() {
            return F::zero();
        }
        // λ · exp(-λx)
        self.inv_scale * (-*x * self.inv_scale).exp()
    }
}

impl<F: Float> UnivariateContinuous<F> for Exponential<F> {
    type Params = ExponentialParams<F>;

    #[inline]
    fn cdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }
        // -expm1(-x/θ) = 1 - exp(-x/θ), numerically stable for small x/θ
        -(-x * self.inv_scale).exp_m1()
    }

    #[inline]
    fn inverse_cdf(&self, p: F) -> F {
        // -θ · log1p(-p)
        -self.scale * (-p).ln_1p()
    }

    fn support(&self) -> (F, F) {
        (F::zero(), F::infinity())
    }

    // Direct exp form avoids cancellation in the default `1 - cdf(x)`.
    fn ccdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::one();
        }
        (-x * self.inv_scale).exp()
    }

    fn params(&self) -> ExponentialParams<F> {
        ExponentialParams::Scale { scale: self.scale }
    }

    fn from_params(params: ExponentialParams<F>) -> Result<Self, DistributionError> {
        match params {
            ExponentialParams::Scale { scale } => Self::from_scale(scale),
            ExponentialParams::Rate { rate } => Self::from_rate(rate),
        }
    }
}

impl<F: Float> HasMean for Exponential<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        Some(self.scale)
    }
}

impl<F: Float> HasVariance for Exponential<F> {
    fn variance(&self) -> Option<F> {
        Some(self.scale * self.scale)
    }
}

impl<F: Float> HasEntropy for Exponential<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        Some(F::one() + self.scale.ln())
    }
}

impl<F: Float> HasMode for Exponential<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        Some(F::zero())
    }
}

impl<F: Float> HasSkewness for Exponential<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        Some(F::from(2.0).unwrap())
    }
}

impl<F: Float> HasKurtosis for Exponential<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        Some(F::from(6.0).unwrap())
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
    fn from_scale_valid() {
        assert!(Exponential::<f64>::from_scale(1.0).is_ok());
        assert!(Exponential::<f64>::from_scale(0.5).is_ok());
        assert!(Exponential::<f64>::from_scale(100.0).is_ok());
        assert!(Exponential::<f32>::from_scale(2.0).is_ok());
    }

    #[test]
    fn from_scale_rejects_non_positive() {
        assert!(Exponential::<f64>::from_scale(0.0).is_err());
        assert!(Exponential::<f64>::from_scale(-1.0).is_err());
    }

    #[test]
    fn from_scale_rejects_nan() {
        assert!(Exponential::<f64>::from_scale(f64::NAN).is_err());
    }

    #[test]
    fn from_scale_rejects_infinite() {
        assert!(Exponential::<f64>::from_scale(f64::INFINITY).is_err());
    }

    #[test]
    fn from_rate_valid() {
        let d = Exponential::<f64>::from_rate(2.0).unwrap();
        assert!((d.scale() - 0.5).abs() < 1e-15);
        assert!((d.rate() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn from_rate_rejects_invalid() {
        assert!(Exponential::<f64>::from_rate(0.0).is_err());
        assert!(Exponential::<f64>::from_rate(-1.0).is_err());
        assert!(Exponential::<f64>::from_rate(f64::NAN).is_err());
        assert!(Exponential::<f64>::from_rate(f64::INFINITY).is_err());
    }

    #[test]
    fn from_rate_unchecked_skips_validation() {
        let d = Exponential::<f64>::from_rate_unchecked(Unchecked, 4.0);
        assert!((d.scale() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn from_scale_unchecked_skips_validation() {
        let d = Exponential::<f64>::from_scale_unchecked(Unchecked, -1.0);
        assert_eq!(d.scale(), -1.0);
    }

    #[test]
    fn accessors() {
        let d = Exponential::<f64>::from_scale(2.0).unwrap();
        assert_eq!(d.scale(), 2.0);
        assert!((d.rate() - 0.5).abs() < 1e-15);
    }

    // --- Reference data: PDF, CDF, quantile, moments (from R) ---

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        run_continuous_reference_tests(
            |scale, _| Exponential::<f64>::from_scale(scale).unwrap(),
            &data,
            1e-12,
        );
    }

    // --- CDF numerical derivative ≈ PDF ---

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = Exponential::<f64>::from_scale(1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.5, 1.0, 2.0, 4.0], 1e-8, 1e-6);

        let d = Exponential::<f64>::from_scale(0.1).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[1.0, 5.0, 10.0, 20.0], 1e-8, 1e-6);

        let d = Exponential::<f64>::from_scale(10.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.1, 0.5, 1.0, 2.0], 1e-8, 1e-6);
    }

    // --- Internal consistency: CDF + CCDF = 1, log_pdf = ln(pdf), monotonicity ---

    #[test]
    fn internal_consistency() {
        let d = Exponential::<f64>::from_scale(1.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0], 1e-14);

        let d = Exponential::<f64>::from_scale(0.01).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 10.0, 100.0, 500.0], 1e-14);
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Exponential::<f64>::from_scale(1.0).unwrap();
        let bins: Vec<(f64, f64)> = vec![
            (0.0, 0.25),
            (0.25, 0.5),
            (0.5, 1.0),
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 3.0),
            (3.0, 5.0),
            (5.0, 10.0),
        ];
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Exponential::<f64>::from_scale(2.0).unwrap();
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
        let d = Exponential::<f64>::from_scale(1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    // --- Sampling: range and fill ---

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = Exponential::<f64>::from_scale(2.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v > 0.0, "sample {v} must be positive");
            assert!(v.is_finite(), "sample {v} must be finite");
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Exponential::<f64>::from_scale(1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v > 0.0);
            assert!(v.is_finite());
        }
    }

    // --- inverse_cdf inverts CDF ---

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = Exponential::<f64>::from_scale(3.0).unwrap();
        for &x in &[0.5, 1.0, 3.0, 5.0, 10.0] {
            let p = d.cdf(x);
            assert!(
                (d.inverse_cdf(p) - x).abs() < 1e-12,
                "inverse_cdf(cdf({x})) = {}, expected {x}",
                d.inverse_cdf(p)
            );
        }
    }

    // --- from_params round-trip ---

    #[test]
    fn from_params_round_trip() {
        let d = Exponential::<f64>::from_scale(2.0).unwrap();
        let d2 = Exponential::from_params(d.params()).unwrap();
        assert_eq!(d.scale(), d2.scale());
    }
}
