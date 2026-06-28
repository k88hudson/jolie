use num_traits::Float;
use rand::Rng;

use crate::constants::EULER_MASCHERONI;
use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::gamma::ln_gamma;
use crate::special::sampling::standard_exponential;
use crate::unchecked::Unchecked;

/// Weibull distribution Weibull(α, θ).
///
/// A two-parameter continuous distribution on the positive reals, with shape
/// α > 0 and scale θ > 0. Generalizes the Exponential (α = 1) by raising the
/// exponential clock to a power; α < 1 models infant mortality, α > 1 wear-out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull<F: Float> {
    shape: F,
    scale: F,
    inv_shape: F,
    inv_scale: F,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeibullParams<F> {
    pub shape: F,
    pub scale: F,
}

impl<F: Float> Weibull<F> {
    /// Constructs a Weibull from shape (α) and scale (θ).
    ///
    /// - `shape` (α): must be finite and > 0
    /// - `scale` (θ): must be finite and > 0
    pub fn new(shape: F, scale: F) -> Result<Self, DistributionError> {
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
        Ok(Self {
            shape,
            scale,
            inv_shape: F::one() / shape,
            inv_scale: F::one() / scale,
        })
    }

    pub fn new_unchecked(_: Unchecked, shape: F, scale: F) -> Self {
        Self {
            shape,
            scale,
            inv_shape: F::one() / shape,
            inv_scale: F::one() / scale,
        }
    }

    pub fn shape(&self) -> F {
        self.shape
    }

    pub fn scale(&self) -> F {
        self.scale
    }

    /// `Γ(1 + n/α)`. The argument is always > 1 (positive), so exponentiating
    /// the log-gamma is exact and sign-safe.
    #[inline]
    fn gamma_moment(&self, n: F) -> F {
        ln_gamma(F::one() + n * self.inv_shape).exp()
    }
}

impl<F: Float> Sampleable for Weibull<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        // Inverse-CDF: X = θ · E^(1/α), E ~ Exp(1) via ziggurat (no ln per sample).
        let e = F::from(standard_exponential(rng)).unwrap();
        self.scale * e.powf(self.inv_shape)
    }
}

crate::distributions::traits::impl_rand_distribution!(Weibull<F: Float> => F);

impl<F: Float> Distribution<F> for Weibull<F> {
    #[inline]
    fn log_pdf(&self, x: &F) -> F {
        let x = *x;
        if x < F::zero() || !x.is_finite() {
            return F::neg_infinity();
        }
        if x == F::zero() {
            // z^α → 0 and (α-1)·ln(z) is the deciding term as z → 0.
            return if self.shape > F::one() {
                F::neg_infinity()
            } else if self.shape == F::one() {
                self.inv_scale.ln() // -ln(θ)
            } else {
                F::infinity()
            };
        }
        // ln(α/θ) + (α-1)·ln(z) - z^α, with z = x/θ
        let z = x * self.inv_scale;
        (self.shape * self.inv_scale).ln() + (self.shape - F::one()) * z.ln() - z.powf(self.shape)
    }
}

impl<F: Float> UnivariateContinuous<F> for Weibull<F> {
    type Params = WeibullParams<F>;

    #[inline]
    fn cdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::zero();
        }
        // 1 - exp(-z^α), stable for small z^α
        -(-(x * self.inv_scale).powf(self.shape)).exp_m1()
    }

    #[inline]
    fn inverse_cdf(&self, p: F) -> F {
        // θ · (-ln(1-p))^(1/α)
        self.scale * (-(-p).ln_1p()).powf(self.inv_shape)
    }

    fn support(&self) -> (F, F) {
        (F::zero(), F::infinity())
    }

    fn ccdf(&self, x: F) -> F {
        if x <= F::zero() {
            return F::one();
        }
        // exp(-z^α), direct form avoids cancellation in 1 - cdf(x)
        (-(x * self.inv_scale).powf(self.shape)).exp()
    }

    fn params(&self) -> WeibullParams<F> {
        WeibullParams {
            shape: self.shape,
            scale: self.scale,
        }
    }

    fn from_params(params: WeibullParams<F>) -> Result<Self, DistributionError> {
        Self::new(params.shape, params.scale)
    }
}

impl<F: Float> HasMean for Weibull<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        Some(self.scale * self.gamma_moment(F::one()))
    }
}

impl<F: Float> HasVariance for Weibull<F> {
    fn variance(&self) -> Option<F> {
        let g1 = self.gamma_moment(F::one());
        let g2 = self.gamma_moment(F::from(2.0).unwrap());
        Some(self.scale * self.scale * (g2 - g1 * g1))
    }
}

impl<F: Float> HasEntropy for Weibull<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        let gamma_e = F::from(EULER_MASCHERONI).unwrap();
        Some(gamma_e * (F::one() - self.inv_shape) + (self.scale * self.inv_shape).ln() + F::one())
    }
}

impl<F: Float> HasMode for Weibull<F> {
    type Value = F;

    fn mode(&self) -> Option<F> {
        if self.shape > F::one() {
            Some(self.scale * ((self.shape - F::one()) * self.inv_shape).powf(self.inv_shape))
        } else {
            // Density is monotone decreasing (or diverges at 0) for α ≤ 1.
            Some(F::zero())
        }
    }
}

impl<F: Float> HasSkewness for Weibull<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let g1 = self.gamma_moment(F::one());
        let g2 = self.gamma_moment(two);
        let g3 = self.gamma_moment(three);
        let var = g2 - g1 * g1;
        // (g3 - 3 g1 g2 + 2 g1³) / (g2 - g1²)^(3/2)
        Some((g3 - three * g1 * g2 + two * g1 * g1 * g1) / var.powf(three / two))
    }
}

impl<F: Float> HasKurtosis for Weibull<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let six = F::from(6.0).unwrap();
        let twelve = F::from(12.0).unwrap();
        let g1 = self.gamma_moment(F::one());
        let g2 = self.gamma_moment(two);
        let g3 = self.gamma_moment(three);
        let g4 = self.gamma_moment(four);
        let g1_2 = g1 * g1;
        let var = g2 - g1_2;
        // Excess kurtosis: (g4 - 4 g1 g3 + 12 g1² g2 - 3 g2² - 6 g1⁴) / (g2 - g1²)²
        let numer = g4 - four * g1 * g3 + twelve * g1_2 * g2 - three * g2 * g2 - six * g1_2 * g1_2;
        Some(numer / (var * var))
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
        assert!(Weibull::<f64>::new(1.0, 1.0).is_ok());
        assert!(Weibull::<f64>::new(0.5, 2.0).is_ok());
        assert!(Weibull::<f64>::new(10.0, 0.5).is_ok());
        assert!(Weibull::<f32>::new(2.0, 3.0).is_ok());
    }

    #[test]
    fn new_rejects_non_positive_shape() {
        assert!(Weibull::<f64>::new(0.0, 1.0).is_err());
        assert!(Weibull::<f64>::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_positive_scale() {
        assert!(Weibull::<f64>::new(1.0, 0.0).is_err());
        assert!(Weibull::<f64>::new(1.0, -1.0).is_err());
    }

    #[test]
    fn new_rejects_nan() {
        assert!(Weibull::<f64>::new(f64::NAN, 1.0).is_err());
        assert!(Weibull::<f64>::new(1.0, f64::NAN).is_err());
    }

    #[test]
    fn new_rejects_infinite() {
        assert!(Weibull::<f64>::new(f64::INFINITY, 1.0).is_err());
        assert!(Weibull::<f64>::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = Weibull::<f64>::new_unchecked(Unchecked, -1.0, 2.0);
        assert_eq!(d.shape(), -1.0);
        assert_eq!(d.scale(), 2.0);
    }

    #[test]
    fn accessors() {
        let d = Weibull::<f64>::new(2.0, 3.0).unwrap();
        assert_eq!(d.shape(), 2.0);
        assert_eq!(d.scale(), 3.0);
    }

    // --- Reference data: PDF, CDF, quantile, moments (from R) ---

    #[test]
    fn reference_pdf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        // Skewness/kurtosis combine up to four Γ(1 + n/α) evaluations with heavy
        // cancellation, so they get a looser tolerance than pdf/cdf and the lower
        // moments (which stay at 1e-12).
        run_continuous_reference_tests_with_moment_tol(
            |shape, scale| Weibull::<f64>::new(shape, scale).unwrap(),
            &data,
            1e-12,
            1e-10,
        );
    }

    // --- CDF numerical derivative ≈ PDF ---

    #[test]
    fn cdf_derivative_approx_pdf() {
        let d = Weibull::<f64>::new(1.5, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.25, 0.5, 1.0, 2.0, 4.0], 1e-8, 1e-6);

        let d = Weibull::<f64>::new(5.0, 2.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.5, 1.0, 2.0, 3.0, 4.0], 1e-8, 1e-6);

        let d = Weibull::<f64>::new(0.8, 1.0).unwrap();
        assert_cdf_derivative_approx_pdf(&d, &[0.1, 0.5, 1.0, 2.0, 5.0], 1e-8, 1e-6);
    }

    // --- Internal consistency: CDF + CCDF = 1, log_pdf = ln(pdf), monotonicity ---

    #[test]
    fn internal_consistency() {
        // Use α > 1 so the PDF is finite everywhere (avoids inf - inf at x = 0).
        let d = Weibull::<f64>::new(2.0, 1.0).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], 1e-14);

        let d = Weibull::<f64>::new(5.0, 0.5).unwrap();
        assert_continuous_consistency(&d, &[-1.0, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0], 1e-14);
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Weibull::<f64>::new(1.5, 2.0).unwrap();
        let bins: Vec<(f64, f64)> = vec![
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 2.5),
            (2.5, 3.0),
            (3.0, 4.0),
            (4.0, 6.0),
            (6.0, 12.0),
        ];
        assert_continuous_sampling_binomial_ci(&d, &mut rng, 100_000, &bins, 5.0);
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Weibull::<f64>::new(2.0, 3.0).unwrap();
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
        let d = Weibull::<f64>::new(2.0, 1.0).unwrap();
        assert_continuous_edge_cases(&d);
    }

    // --- Sampling: range and fill ---

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = Weibull::<f64>::new(0.7, 2.0).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!(v >= 0.0, "sample {v} must be non-negative");
            assert!(v.is_finite(), "sample {v} must be finite");
        }
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Weibull::<f64>::new(3.0, 1.0).unwrap();
        let mut buf = [0.0f64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }
    }

    // --- inverse_cdf inverts CDF ---

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = Weibull::<f64>::new(2.5, 1.5).unwrap();
        // Stay out of the deep upper tail: once cdf(x) rounds to ~1, the round-trip
        // is ill-conditioned (cquantile handles the tail). These points keep p well
        // away from 1.
        for &x in &[0.25, 0.5, 1.0, 1.5, 2.0] {
            let p = d.cdf(x);
            assert!(
                (d.inverse_cdf(p) - x).abs() < 1e-10,
                "inverse_cdf(cdf({x})) = {}, expected {x}",
                d.inverse_cdf(p)
            );
        }
    }

    // --- Stable upper-tail survival (ccdf) ---

    #[test]
    fn ccdf_upper_tail_stable() {
        let d = Weibull::<f64>::new(1.5, 2.0).unwrap();

        // Deep in the upper tail the CDF rounds to 1.0, so the naive `1 - cdf`
        // survival collapses to 0; the direct ccdf must not.
        let x = 50.0;
        let z_pow = (x * 0.5_f64).powf(1.5); // (x/θ)^α
        assert_eq!(
            1.0 - d.cdf(x),
            0.0,
            "naive 1 - cdf should underflow to 0 here"
        );

        // ccdf = exp(-z^α): a tiny positive value (not 0) with the right log.
        let s = d.ccdf(x);
        assert!(
            s > 0.0 && s < 1e-50,
            "ccdf deep tail should be tiny positive, got {s}"
        );
        assert!(
            (s.ln() + z_pow).abs() <= 1e-13 * z_pow,
            "ln(ccdf({x})) = {}, expected {}",
            s.ln(),
            -z_pow
        );
    }

    // --- Mode ---

    #[test]
    fn mode_values() {
        // shape ≤ 1: mode at 0 (density monotone decreasing, or diverges at 0)
        assert_eq!(Weibull::<f64>::new(1.0, 1.0).unwrap().mode().unwrap(), 0.0);
        assert_eq!(Weibull::<f64>::new(0.5, 2.0).unwrap().mode().unwrap(), 0.0);
        // shape > 1: mode = θ·((α-1)/α)^(1/α)
        let m = Weibull::<f64>::new(2.0, 1.0).unwrap().mode().unwrap();
        assert!((m - 0.5_f64.sqrt()).abs() < 1e-15, "mode = {m}");
        // Independent reference value (statrs) for Weibull(10, 10)
        let m = Weibull::<f64>::new(10.0, 10.0).unwrap().mode().unwrap();
        assert!((m - 9.895_192_582_062_144).abs() < 1e-13, "mode = {m}");
    }

    // --- from_params round-trip ---

    #[test]
    fn from_params_round_trip() {
        let d = Weibull::<f64>::new(2.0, 3.0).unwrap();
        let d2 = Weibull::from_params(d.params()).unwrap();
        assert_eq!(d.shape(), d2.shape());
        assert_eq!(d.scale(), d2.scale());
    }

    // --- Special property: Weibull(1, θ) ≡ Exponential(θ) ---

    #[test]
    fn matches_exponential_shape_one() {
        use crate::distributions::Exponential;
        let theta = 2.5;
        let wb = Weibull::<f64>::new(1.0, theta).unwrap();
        let exp = Exponential::<f64>::from_scale(theta).unwrap();

        // wb moments go through Γ(1 + n/α), so they carry a few ULP more error than
        // Exponential's exact closed forms.
        assert!((wb.mean().unwrap() - exp.mean().unwrap()).abs() < 1e-13);
        assert!((wb.variance().unwrap() - exp.variance().unwrap()).abs() < 1e-13);
        assert!((wb.entropy().unwrap() - exp.entropy().unwrap()).abs() < 1e-13);

        for &x in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            assert!(
                (wb.pdf(&x) - exp.pdf(&x)).abs() < 1e-14,
                "pdf({x}): wb={}, exp={}",
                wb.pdf(&x),
                exp.pdf(&x)
            );
            assert!(
                (wb.cdf(x) - exp.cdf(x)).abs() < 1e-14,
                "cdf({x}): wb={}, exp={}",
                wb.cdf(x),
                exp.cdf(x)
            );
        }
    }
}
