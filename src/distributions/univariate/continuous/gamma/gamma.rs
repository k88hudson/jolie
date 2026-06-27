use num_traits::Float;
use rand::{Rng, RngExt};

use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::gamma::{
    digamma, ln_gamma, regularized_gamma_compl, regularized_gamma_inc, regularized_gamma_inc_inv,
};
use crate::special::sampling::{standard_exponential, standard_normal};
use crate::unchecked::Unchecked;

/// Pre-computed sampling constants for the Marsaglia-Tsang method.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GammaSampler<F: Float> {
    /// shape > 1: pre-computed d, c
    Large { d: F, c: F, scale: F },
    /// shape == 1: exponential
    One { scale: F },
    /// shape < 1: Marsaglia-Tsang on shape+1, then boost by U^(1/shape)
    Small { d: F, c: F, inv_shape: F, scale: F },
}

/// Gamma distribution Gamma(α, θ). Generalizes the exponential (α = 1) and
/// chi-squared (α = k/2, θ = 2) distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma<F: Float> {
    shape: F,
    scale: F,
    sampler: GammaSampler<F>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GammaParams<F> {
    ShapeScale { shape: F, scale: F },
    ShapeRate { shape: F, rate: F },
}

/// Serde: `{"shape": 2.0, "scale": 1.0}` or `{"shape": 2.0, "rate": 0.5}`.
/// Presence of `scale` vs `rate` determines the variant.
#[cfg(feature = "serde")]
mod gamma_params_serde {
    use super::GammaParams;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize)]
    struct RawRef<'a, F: Serialize> {
        shape: &'a F,
        #[serde(skip_serializing_if = "Option::is_none")]
        scale: Option<&'a F>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rate: Option<&'a F>,
    }

    #[derive(Deserialize)]
    struct RawOwned<F> {
        shape: F,
        scale: Option<F>,
        rate: Option<F>,
    }

    impl<F: Serialize> Serialize for GammaParams<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw = match self {
                GammaParams::ShapeScale { shape, scale } => RawRef {
                    shape,
                    scale: Some(scale),
                    rate: None,
                },
                GammaParams::ShapeRate { shape, rate } => RawRef {
                    shape,
                    scale: None,
                    rate: Some(rate),
                },
            };
            raw.serialize(serializer)
        }
    }

    impl<'de, F: Deserialize<'de>> Deserialize<'de> for GammaParams<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw = RawOwned::<F>::deserialize(deserializer)?;
            match (raw.scale, raw.rate) {
                (Some(scale), None) => Ok(GammaParams::ShapeScale {
                    shape: raw.shape,
                    scale,
                }),
                (None, Some(rate)) => Ok(GammaParams::ShapeRate {
                    shape: raw.shape,
                    rate,
                }),
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

impl<F: Float> Gamma<F> {
    /// Constructs a Gamma from shape (α) and scale (θ). Both must be finite and
    /// positive.
    pub fn shape_scale(shape: F, scale: F) -> Result<Self, DistributionError> {
        if !shape.is_finite() || !scale.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "shape and scale must be finite",
            ));
        }
        if shape <= F::zero() {
            return Err(DistributionError::InvalidParameter(
                "shape must be positive",
            ));
        }
        if scale <= F::zero() {
            return Err(DistributionError::InvalidParameter(
                "scale must be positive",
            ));
        }
        let sampler = Self::make_sampler(shape, scale);
        Ok(Self {
            shape,
            scale,
            sampler,
        })
    }

    /// Constructs a Gamma from shape (α) and rate (β = 1/θ). Both must be finite
    /// and positive.
    pub fn shape_rate(shape: F, rate: F) -> Result<Self, DistributionError> {
        if !rate.is_finite() || rate <= F::zero() {
            return Err(DistributionError::InvalidParameter(
                "rate must be finite and positive",
            ));
        }
        Self::shape_scale(shape, F::one() / rate)
    }

    pub fn shape_scale_unchecked(_: Unchecked, shape: F, scale: F) -> Self {
        let sampler = Self::make_sampler(shape, scale);
        Self {
            shape,
            scale,
            sampler,
        }
    }

    pub fn shape_rate_unchecked(_: Unchecked, shape: F, rate: F) -> Self {
        Self::shape_scale_unchecked(Unchecked, shape, F::one() / rate)
    }

    fn make_sampler(shape: F, scale: F) -> GammaSampler<F> {
        let one_third = F::from(1.0 / 3.0).unwrap();
        let nine = F::from(9.0).unwrap();
        if shape < F::one() {
            let effective = shape + F::one();
            let d = effective - one_third;
            let c = F::one() / (nine * d).sqrt();
            GammaSampler::Small {
                d,
                c,
                inv_shape: F::one() / shape,
                scale,
            }
        } else if shape == F::one() {
            GammaSampler::One { scale }
        } else {
            let d = shape - one_third;
            let c = F::one() / (nine * d).sqrt();
            GammaSampler::Large { d, c, scale }
        }
    }

    pub fn shape(&self) -> F {
        self.shape
    }

    pub fn scale(&self) -> F {
        self.scale
    }
}

impl<F: Float> Sampleable for Gamma<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        match self.sampler {
            GammaSampler::Large { d, c, scale } => gamma_mt_sample(d, c, rng) * scale,
            GammaSampler::One { scale } => F::from(standard_exponential(rng)).unwrap() * scale,
            GammaSampler::Small {
                d,
                c,
                inv_shape,
                scale,
            } => {
                let u = F::from(rng.random::<f64>()).unwrap();
                gamma_mt_sample(d, c, rng) * u.powf(inv_shape) * scale
            }
        }
    }
}

crate::distributions::traits::impl_rand_distribution!(Gamma<F: Float> => F);

/// Marsaglia-Tsang inner loop with pre-computed d and c, using ziggurat normals
/// and the squeeze test (avoids the log ~97% of the time).
#[inline]
fn gamma_mt_sample<F: Float, R: Rng + ?Sized>(d: F, c: F, rng: &mut R) -> F {
    let half = F::from(0.5).unwrap();
    let squeeze = F::from(0.0331).unwrap();

    loop {
        let x = F::from(standard_normal(rng)).unwrap();

        let v_base = F::one() + c * x;
        if v_base <= F::zero() {
            continue;
        }
        let v = v_base * v_base * v_base;

        let u: F = F::from(rng.random::<f64>()).unwrap();
        let x2 = x * x;

        if u < F::one() - squeeze * x2 * x2 {
            return d * v;
        }
        if u.ln() < half * x2 + d * (F::one() - v + v.ln()) {
            return d * v;
        }
    }
}

impl<F: Float> Distribution<F> for Gamma<F> {
    fn log_pdf(&self, x: &F) -> F {
        if *x < F::zero() || !x.is_finite() {
            return F::neg_infinity();
        }
        let alpha = self.shape;
        let theta = self.scale;
        if *x == F::zero() {
            if alpha > F::one() {
                return F::neg_infinity();
            } else if alpha == F::one() {
                return -theta.ln();
            } else {
                return F::infinity();
            }
        }
        (alpha - F::one()) * x.ln() - *x / theta - ln_gamma(alpha) - alpha * theta.ln()
    }
}

impl<F: Float> UnivariateContinuous<F> for Gamma<F> {
    type Params = GammaParams<F>;

    fn cdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }
        regularized_gamma_inc(self.shape, x / self.scale)
    }

    fn inverse_cdf(&self, p: F) -> F {
        regularized_gamma_inc_inv(self.shape, p) * self.scale
    }

    // Direct Q(shape, x/scale) avoids the 1 - cdf cancellation in the upper tail.
    fn ccdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::one();
        }
        regularized_gamma_compl(self.shape, x / self.scale)
    }

    fn support(&self) -> (F, F) {
        (F::zero(), F::infinity())
    }

    fn params(&self) -> GammaParams<F> {
        GammaParams::ShapeScale {
            shape: self.shape,
            scale: self.scale,
        }
    }

    fn from_params(params: GammaParams<F>) -> Result<Self, DistributionError> {
        match params {
            GammaParams::ShapeScale { shape, scale } => Self::shape_scale(shape, scale),
            GammaParams::ShapeRate { shape, rate } => Self::shape_rate(shape, rate),
        }
    }
}

impl<F: Float> HasMean for Gamma<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        Some(self.shape * self.scale)
    }
}

impl<F: Float> HasVariance for Gamma<F> {
    fn variance(&self) -> Option<F> {
        Some(self.shape * self.scale * self.scale)
    }
}

impl<F: Float> HasEntropy for Gamma<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        let alpha = self.shape;
        let theta = self.scale;
        Some(alpha + theta.ln() + ln_gamma(alpha) + (F::one() - alpha) * digamma(alpha))
    }
}

impl<F: Float> HasMode for Gamma<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        if self.shape >= F::one() {
            Some((self.shape - F::one()) * self.scale)
        } else {
            None
        }
    }
}

impl<F: Float> HasSkewness for Gamma<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        Some(two / self.shape.sqrt())
    }
}

impl<F: Float> HasKurtosis for Gamma<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        let six = F::from(6.0).unwrap();
        Some(six / self.shape)
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
    fn shape_scale_valid() {
        assert!(Gamma::<f64>::shape_scale(1.0, 1.0).is_ok());
        assert!(Gamma::<f64>::shape_scale(0.5, 2.0).is_ok());
        assert!(Gamma::<f64>::shape_scale(100.0, 0.01).is_ok());
        assert!(Gamma::<f32>::shape_scale(2.0, 3.0).is_ok());
    }

    #[test]
    fn shape_rate_valid() {
        let d = Gamma::<f64>::shape_rate(2.0, 0.5).unwrap();
        assert_eq!(d.shape(), 2.0);
        assert_eq!(d.scale(), 2.0); // 1/0.5
    }

    #[test]
    fn shape_rate_unchecked_skips_validation() {
        let d = Gamma::<f64>::shape_rate_unchecked(Unchecked, 3.0, 0.25);
        assert_eq!(d.shape(), 3.0);
        assert_eq!(d.scale(), 4.0);
    }

    #[test]
    fn shape_rate_rejects_invalid() {
        assert!(Gamma::<f64>::shape_rate(2.0, 0.0).is_err());
        assert!(Gamma::<f64>::shape_rate(2.0, -1.0).is_err());
        assert!(Gamma::<f64>::shape_rate(0.0, 1.0).is_err());
        assert!(Gamma::<f64>::shape_rate(2.0, f64::NAN).is_err());
        assert!(Gamma::<f64>::shape_rate(2.0, f64::INFINITY).is_err());
    }

    #[test]
    fn shape_scale_rejects_non_positive_shape() {
        assert!(Gamma::<f64>::shape_scale(0.0, 1.0).is_err());
        assert!(Gamma::<f64>::shape_scale(-1.0, 1.0).is_err());
    }

    #[test]
    fn shape_scale_rejects_non_positive_scale() {
        assert!(Gamma::<f64>::shape_scale(1.0, 0.0).is_err());
        assert!(Gamma::<f64>::shape_scale(1.0, -1.0).is_err());
    }

    #[test]
    fn shape_scale_rejects_non_finite() {
        assert!(Gamma::<f64>::shape_scale(f64::NAN, 1.0).is_err());
        assert!(Gamma::<f64>::shape_scale(1.0, f64::NAN).is_err());
        assert!(Gamma::<f64>::shape_scale(f64::INFINITY, 1.0).is_err());
        assert!(Gamma::<f64>::shape_scale(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn shape_scale_unchecked_skips_validation() {
        let d = Gamma::<f64>::shape_scale_unchecked(Unchecked, -1.0, -1.0);
        assert_eq!(d.shape(), -1.0);
        assert_eq!(d.scale(), -1.0);
    }

    #[test]
    fn accessors() {
        let d = Gamma::<f64>::shape_scale(2.0, 3.0).unwrap();
        assert_eq!(d.shape(), 2.0);
        assert_eq!(d.scale(), 3.0);
    }

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        run_continuous_reference_tests(
            |shape, scale| Gamma::<f64>::shape_scale(shape, scale).unwrap(),
            &data,
            1e-12,
        );
    }

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.5, 1.0, 2.0, 4.0], 1e-8, 1e-6);

        let d = Gamma::<f64>::shape_scale(5.0, 2.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[2.0, 5.0, 10.0, 15.0], 1e-8, 1e-6);

        let d = Gamma::<f64>::shape_scale(0.5, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.1, 0.5, 1.0, 3.0], 1e-8, 1e-6);
    }

    #[test]
    fn internal_consistency() {
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0], 1e-14);

        // shape < 1: pdf(0) = inf, so skip x=0.
        let d = Gamma::<f64>::shape_scale(0.5, 3.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.1, 1.0, 5.0, 20.0], 1e-14);
    }

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
        let bins: Vec<(f64, f64)> = vec![
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 3.0),
            (3.0, 4.0),
            (4.0, 6.0),
            (6.0, 10.0),
        ];
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
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
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = Gamma::<f64>::shape_scale(2.0, 3.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v > 0.0 && v.is_finite(), "sample {v}");
        }
    }

    #[test]
    fn samples_in_range_small_shape() {
        let mut rng = test_rng();
        let d = Gamma::<f64>::shape_scale(0.1, 1.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v > 0.0 && v.is_finite(), "sample {v}");
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Gamma::<f64>::shape_scale(2.0, 1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v > 0.0 && v.is_finite());
        }
    }

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = Gamma::<f64>::shape_scale(3.0, 2.0).unwrap();
        for &x in &[0.5, 2.0, 5.0, 10.0, 15.0] {
            let p = d.cdf(x);
            assert!(
                (d.inverse_cdf(p) - x).abs() < 1e-8,
                "inverse_cdf(cdf({x})) = {}",
                d.inverse_cdf(p)
            );
        }
    }

    #[test]
    fn ccdf_upper_tail_stable() {
        // ccdf must not collapse to 0 in the upper tail (the 1 - cdf trap).
        // shape=1: S(x) = e^{-x/θ}; shape=2: S = (1 + x/θ) e^{-x/θ}.
        let scale = 2.0;
        let d1 = Gamma::<f64>::shape_scale(1.0, scale).unwrap();
        let d2 = Gamma::<f64>::shape_scale(2.0, scale).unwrap();
        for &x in &[40.0, 60.0, 80.0, 120.0] {
            let t = x / scale;
            let exact1 = (-t).exp();
            let exact2 = (1.0 + t) * (-t).exp();
            assert!((d1.ccdf(x) - exact1).abs() / exact1 < 1e-12, "ccdf1 at {x}");
            assert!((d2.ccdf(x) - exact2).abs() / exact2 < 1e-12, "ccdf2 at {x}");
            assert!(d2.ccdf(x) > 0.0, "ccdf underflowed at {x}");
            assert!(d2.ccdf(x).ln().is_finite(), "log ccdf not finite at {x}");
        }
        for &x in &[1.0, 3.0, 6.0, 10.0] {
            assert!((d2.cdf(x) + d2.ccdf(x) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mode_values() {
        let d = Gamma::<f64>::shape_scale(3.0, 2.0).unwrap();
        assert!((d.mode().unwrap() - 4.0).abs() < 1e-14); // (3-1)*2

        let d = Gamma::<f64>::shape_scale(1.0, 5.0).unwrap();
        assert!((d.mode().unwrap()).abs() < 1e-14); // (1-1)*5

        let d = Gamma::<f64>::shape_scale(0.5, 1.0).unwrap();
        assert_eq!(d.mode(), None); // shape < 1
    }

    #[test]
    fn from_params_round_trip() {
        let d = Gamma::<f64>::shape_scale(2.0, 3.0).unwrap();
        let d2 = Gamma::from_params(d.params()).unwrap();
        assert_eq!(d.shape(), d2.shape());
        assert_eq!(d.scale(), d2.scale());
    }

    #[test]
    fn from_params_shape_rate() {
        let d = Gamma::<f64>::from_params(GammaParams::ShapeRate {
            shape: 2.0,
            rate: 0.5,
        })
        .unwrap();
        assert_eq!(d.shape(), 2.0);
        assert_eq!(d.scale(), 2.0);
    }

    #[test]
    fn shape_one_is_exponential() {
        let d = Gamma::<f64>::shape_scale(1.0, 2.0).unwrap();
        assert!((d.mean().unwrap() - 2.0).abs() < 1e-14);
        assert!((d.variance().unwrap() - 4.0).abs() < 1e-14);
        let x = 1.0;
        let expected_cdf = 1.0 - (-x / 2.0_f64).exp();
        assert!((d.cdf(x) - expected_cdf).abs() < 1e-12);
    }
}
