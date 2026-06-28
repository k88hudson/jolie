use num_traits::Float;
use rand::Rng;

use crate::distributions::Gamma;
use crate::distributions::Poisson;
use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::beta::{regularized_beta_compl, regularized_beta_inc};
use crate::special::gamma::{ln_factorial, ln_gamma};
use crate::unchecked::Unchecked;

/// Negative Binomial distribution NB(r, p).
///
/// Models the number of failures before `r` successes in independent Bernoulli
/// trials with success probability `p`. Generalizes the geometric distribution
/// (r = 1). The parameter `r` can be any positive real, not just an integer.
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial<F: Float> {
    r: F,
    p: F,
    // Precomputed for sampling: Gamma(r, (1-p)/p)
    gamma_sampler: Gamma<F>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NegativeBinomialParams<F> {
    RP { r: F, p: F },
    MeanDispersion { mean: F, k: F },
}

/// Serde: `{"r": 5.0, "p": 0.5}` or `{"mean": 10.0, "k": 5.0}`.
/// Presence of `p` vs `mean` determines the variant.
#[cfg(feature = "serde")]
mod nb_params_serde {
    use super::NegativeBinomialParams;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize)]
    struct RawRef<'a, F: Serialize> {
        #[serde(skip_serializing_if = "Option::is_none")]
        r: Option<&'a F>,
        #[serde(skip_serializing_if = "Option::is_none")]
        p: Option<&'a F>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mean: Option<&'a F>,
        #[serde(skip_serializing_if = "Option::is_none")]
        k: Option<&'a F>,
    }

    #[derive(Deserialize)]
    struct RawOwned<F> {
        r: Option<F>,
        p: Option<F>,
        mean: Option<F>,
        k: Option<F>,
    }

    impl<F: Serialize> Serialize for NegativeBinomialParams<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw = match self {
                NegativeBinomialParams::RP { r, p } => RawRef {
                    r: Some(r),
                    p: Some(p),
                    mean: None,
                    k: None,
                },
                NegativeBinomialParams::MeanDispersion { mean, k } => RawRef {
                    r: None,
                    p: None,
                    mean: Some(mean),
                    k: Some(k),
                },
            };
            raw.serialize(serializer)
        }
    }

    impl<'de, F: Deserialize<'de>> Deserialize<'de> for NegativeBinomialParams<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw = RawOwned::<F>::deserialize(deserializer)?;
            match (raw.r, raw.p, raw.mean, raw.k) {
                (Some(r), Some(p), None, None) => Ok(NegativeBinomialParams::RP { r, p }),
                (None, None, Some(mean), Some(k)) => {
                    Ok(NegativeBinomialParams::MeanDispersion { mean, k })
                }
                _ => Err(serde::de::Error::custom(
                    "specify either {r, p} or {mean, k}",
                )),
            }
        }
    }
}

impl<F: Float> PartialEq for NegativeBinomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r && self.p == other.p
    }
}

impl<F: Float> NegativeBinomial<F> {
    /// Constructs from mean (μ) and dispersion (k).
    ///
    /// Var = μ + μ²/k. Smaller k means more dispersion; as k → ∞, approaches
    /// Poisson. Converts via r = k, p = k / (k + μ).
    ///
    /// Both must be finite and positive.
    pub fn mean_dispersion(mean: F, k: F) -> Result<Self, DistributionError> {
        if !mean.is_finite() || !k.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "mean and k must be finite",
            ));
        }
        if mean <= F::zero() {
            return Err(DistributionError::InvalidParameter("mean must be positive"));
        }
        if k <= F::zero() {
            return Err(DistributionError::InvalidParameter("k must be positive"));
        }
        let p = k / (k + mean);
        Self::new(k, p)
    }

    pub fn mean_dispersion_unchecked(_: Unchecked, mean: F, k: F) -> Self {
        let p = k / (k + mean);
        Self::new_unchecked(Unchecked, k, p)
    }

    /// - `r`: number of successes (positive real), must be finite and > 0
    /// - `p`: success probability, must be finite and in (0, 1\]
    pub fn new(r: F, p: F) -> Result<Self, DistributionError> {
        if !r.is_finite() || !p.is_finite() {
            return Err(DistributionError::InvalidParameter(
                "r and p must be finite",
            ));
        }
        if r <= F::zero() {
            return Err(DistributionError::InvalidParameter("r must be positive"));
        }
        if p <= F::zero() || p > F::one() {
            return Err(DistributionError::InvalidParameter("p must be in (0, 1]"));
        }
        let gamma_sampler = Self::make_gamma(r, p);
        Ok(Self {
            r,
            p,
            gamma_sampler,
        })
    }

    pub fn new_unchecked(_: Unchecked, r: F, p: F) -> Self {
        let gamma_sampler = Self::make_gamma(r, p);
        Self {
            r,
            p,
            gamma_sampler,
        }
    }

    fn make_gamma(r: F, p: F) -> Gamma<F> {
        // Gamma(shape=r, scale=(1-p)/p)
        let scale = (F::one() - p) / p;
        // When p=1, scale=0 which is invalid, but we short-circuit in sample()
        if scale > F::zero() {
            Gamma::shape_scale_unchecked(Unchecked, r, scale)
        } else {
            // Dummy gamma, won't be used (p=1 → always 0 failures)
            Gamma::shape_scale_unchecked(Unchecked, F::one(), F::one())
        }
    }

    pub fn r(&self) -> F {
        self.r
    }

    pub fn p(&self) -> F {
        self.p
    }
}

// Gamma-Poisson mixture sampling (used by Julia Distributions.jl and statrs)
impl<F: Float> Sampleable for NegativeBinomial<F> {
    type Value = u64;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.p == F::one() {
            return 0;
        }
        // Step 1: λ ~ Gamma(r, (1-p)/p)
        let lambda = self.gamma_sampler.sample(rng);
        let lam_f64 = num_traits::ToPrimitive::to_f64(&lambda).unwrap();
        if lam_f64 <= 0.0 {
            return 0;
        }
        // Step 2: X ~ Poisson(λ)
        Poisson::<f64>::new_unchecked(Unchecked, lam_f64).sample(rng)
    }
}

crate::distributions::traits::impl_rand_distribution!(NegativeBinomial<F: Float> => u64);

impl<F: Float> Distribution<F> for NegativeBinomial<F> {
    fn log_pdf(&self, x: &u64) -> F {
        let k = *x;
        if k == 0 {
            // P(X=0) = p^r; also avoids 0 * ln(1-p) = NaN when p = 1
            return self.r * self.p.ln();
        }
        let r: f64 = num_traits::ToPrimitive::to_f64(&self.r).unwrap();
        let p: f64 = num_traits::ToPrimitive::to_f64(&self.p).unwrap();
        let kf = k as f64;
        // log P(X=k) = lnΓ(r+k) - lnΓ(r) - ln(k!) + r·ln(p) + k·ln(1-p)
        // Matches GSL randist/nbinomial.c and Julia Distributions.jl (expanded form).
        // Uses ln_factorial(k) (table lookup for k ≤ 170) instead of ln_gamma(k+1).
        let result =
            ln_gamma(r + kf) - ln_gamma(r) - ln_factorial(k) + r * p.ln() + kf * (-p).ln_1p();
        F::from(result).unwrap()
    }
}

impl<F: Float> UnivariateDiscrete<F, u64> for NegativeBinomial<F> {
    type Params = NegativeBinomialParams<F>;

    fn cdf(&self, x: u64) -> F {
        regularized_beta_inc(self.r, F::from(x + 1).unwrap(), self.p)
    }

    // P(X > k) = 1 - I_p(r, k+1), computed directly via the complement so the
    // upper tail stays accurate where the default 1 - cdf would cancel to 0.
    fn ccdf(&self, x: u64) -> F {
        regularized_beta_compl(self.r, F::from(x + 1).unwrap(), self.p)
    }

    fn inverse_cdf(&self, q: F) -> u64 {
        if q <= F::zero() {
            return 0;
        }
        if q >= F::one() {
            return u64::MAX;
        }
        let mean_f64 = num_traits::ToPrimitive::to_f64(&self.mean().unwrap()).unwrap();
        let var_f64 = num_traits::ToPrimitive::to_f64(&self.variance().unwrap()).unwrap();
        let std = var_f64.sqrt();
        let mut lo = 0u64;
        let mut hi = (mean_f64 + 20.0 * std).ceil() as u64;
        while self.cdf(hi) < q {
            hi = hi.saturating_mul(2);
            if hi == u64::MAX {
                break;
            }
        }
        // Left-continuity fuzz matching R's qnbinom: nudge q down by a few ULPs
        // so a quantile probability landing one ULP below an exact cdf jump
        // (cross-implementation rounding in regularized_beta_inc) still maps to
        // the lower integer. R uses 64*DBL_EPSILON; the previous absolute 1e-12
        // was ~70x coarser and silently corrupted deep upper-tail quantiles
        // (where consecutive cdf gaps fall below 1e-12).
        let q = q * (F::one() - F::from(64.0 * f64::EPSILON).unwrap());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) >= q {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    fn support(&self) -> (u64, u64) {
        (0, u64::MAX)
    }

    fn params(&self) -> NegativeBinomialParams<F> {
        NegativeBinomialParams::RP {
            r: self.r,
            p: self.p,
        }
    }

    fn from_params(params: NegativeBinomialParams<F>) -> Result<Self, DistributionError> {
        match params {
            NegativeBinomialParams::RP { r, p } => Self::new(r, p),
            NegativeBinomialParams::MeanDispersion { mean, k } => Self::mean_dispersion(mean, k),
        }
    }
}

impl<F: Float> HasMean for NegativeBinomial<F> {
    type Value = F;
    fn mean(&self) -> Option<F> {
        Some(self.r * (F::one() - self.p) / self.p)
    }
}

impl<F: Float> HasVariance for NegativeBinomial<F> {
    fn variance(&self) -> Option<F> {
        Some(self.r * (F::one() - self.p) / (self.p * self.p))
    }
}

impl<F: Float> HasEntropy for NegativeBinomial<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        // No closed form — compute by series summation
        let r_f64 = num_traits::ToPrimitive::to_f64(&self.r).unwrap();
        let p_f64 = num_traits::ToPrimitive::to_f64(&self.p).unwrap();
        let q = 1.0 - p_f64;

        if p_f64 == 1.0 {
            return Some(F::zero());
        }

        let mut s = 0.0_f64;
        // log P(0) = r * ln(p)
        let mut log_pmf = r_f64 * p_f64.ln();
        let p0 = log_pmf.exp();
        if p0 > 0.0 {
            s += p0 * log_pmf;
        }

        let mean = r_f64 * q / p_f64;
        let std = (r_f64 * q).sqrt() / p_f64;
        let max_k = (mean + 8.0 * std.max(5.0)).ceil() as usize;

        for k in 1..=max_k {
            let kf = k as f64;
            // Recurrence: P(k) / P(k-1) = (k+r-1)/k * q
            log_pmf += ((kf + r_f64 - 1.0) / kf).ln() + q.ln();
            let pk = log_pmf.exp();
            if pk < 1e-20 && kf > mean {
                break;
            }
            if pk > 0.0 {
                s += pk * log_pmf;
            }
        }
        Some(F::from(-s).unwrap())
    }
}

impl<F: Float> HasMode for NegativeBinomial<F> {
    type Value = u64;
    fn mode(&self) -> Option<u64> {
        if self.r > F::one() {
            let val = (self.r - F::one()) * (F::one() - self.p) / self.p;
            Some(num_traits::ToPrimitive::to_u64(&val.floor()).unwrap())
        } else {
            Some(0)
        }
    }
}

impl<F: Float> HasSkewness for NegativeBinomial<F> {
    type Value = F;
    fn skewness(&self) -> Option<F> {
        let two = F::from(2.0).unwrap();
        Some((two - self.p) / (self.r * (F::one() - self.p)).sqrt())
    }
}

impl<F: Float> HasKurtosis for NegativeBinomial<F> {
    type Value = F;
    fn kurtosis(&self) -> Option<F> {
        let six = F::from(6.0).unwrap();
        Some(six / self.r + self.p * self.p / (self.r * (F::one() - self.p)))
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
        assert!(NegativeBinomial::<f64>::new(1.0, 0.5).is_ok());
        assert!(NegativeBinomial::<f64>::new(0.5, 0.5).is_ok()); // non-integer r
        assert!(NegativeBinomial::<f64>::new(10.0, 0.9).is_ok());
        assert!(NegativeBinomial::<f64>::new(5.0, 1.0).is_ok()); // p=1
    }

    #[test]
    fn mean_dispersion_valid() {
        // mean=10, k=5 → r=5, p = 5/(5+10) = 1/3
        let d = NegativeBinomial::<f64>::mean_dispersion(10.0, 5.0).unwrap();
        assert_eq!(d.r(), 5.0);
        assert!((d.p() - 1.0 / 3.0).abs() < 1e-14);
        assert!((d.mean().unwrap() - 10.0).abs() < 1e-12);
        // Var = μ + μ²/k = 10 + 100/5 = 30
        assert!((d.variance().unwrap() - 30.0).abs() < 1e-12);
    }

    #[test]
    fn mean_dispersion_unchecked() {
        let d = NegativeBinomial::<f64>::mean_dispersion_unchecked(Unchecked, 10.0, 5.0);
        assert_eq!(d.r(), 5.0);
        assert!((d.p() - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn mean_dispersion_rejects_invalid() {
        assert!(NegativeBinomial::<f64>::mean_dispersion(0.0, 5.0).is_err());
        assert!(NegativeBinomial::<f64>::mean_dispersion(-1.0, 5.0).is_err());
        assert!(NegativeBinomial::<f64>::mean_dispersion(10.0, 0.0).is_err());
        assert!(NegativeBinomial::<f64>::mean_dispersion(10.0, -1.0).is_err());
        assert!(NegativeBinomial::<f64>::mean_dispersion(f64::NAN, 5.0).is_err());
        assert!(NegativeBinomial::<f64>::mean_dispersion(10.0, f64::INFINITY).is_err());
    }

    #[test]
    fn from_params_mean_dispersion() {
        let d = NegativeBinomial::<f64>::from_params(NegativeBinomialParams::MeanDispersion {
            mean: 10.0,
            k: 5.0,
        })
        .unwrap();
        assert!((d.mean().unwrap() - 10.0).abs() < 1e-12);
        assert_eq!(d.r(), 5.0);
    }

    #[test]
    fn new_rejects_non_positive_r() {
        assert!(NegativeBinomial::<f64>::new(0.0, 0.5).is_err());
        assert!(NegativeBinomial::<f64>::new(-1.0, 0.5).is_err());
    }

    #[test]
    fn new_rejects_invalid_p() {
        assert!(NegativeBinomial::<f64>::new(1.0, 0.0).is_err());
        assert!(NegativeBinomial::<f64>::new(1.0, 1.1).is_err());
        assert!(NegativeBinomial::<f64>::new(1.0, -0.1).is_err());
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(NegativeBinomial::<f64>::new(f64::NAN, 0.5).is_err());
        assert!(NegativeBinomial::<f64>::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = NegativeBinomial::<f64>::new_unchecked(Unchecked, -1.0, 2.0);
        assert_eq!(d.r(), -1.0);
        assert_eq!(d.p(), 2.0);
    }

    #[test]
    fn accessors() {
        let d = NegativeBinomial::<f64>::new(5.0, 0.3).unwrap();
        assert_eq!(d.r(), 5.0);
        assert_eq!(d.p(), 0.3);
    }

    // --- Reference data ---

    #[test]
    fn reference_pmf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        for case in &data.cases {
            let r = case.params.a;
            let p = case.params.b;
            let d = NegativeBinomial::<f64>::new(r, p).unwrap();
            let tol = 1e-10;

            let m = &case.moments;
            if let Some(expected) = m.mean {
                assert!(
                    (d.mean().unwrap() - expected).abs() < tol,
                    "mean for r={r},p={p}: got {}, expected {expected}",
                    d.mean().unwrap()
                );
            }
            if let Some(expected) = m.variance {
                assert!(
                    (d.variance().unwrap() - expected).abs() < tol,
                    "variance for r={r},p={p}: got {}, expected {expected}",
                    d.variance().unwrap()
                );
            }
            if let Some(expected) = m.entropy {
                assert!(
                    (d.entropy().unwrap() - expected).abs() < 1e-4,
                    "entropy for r={r},p={p}: got {}, expected {expected}",
                    d.entropy().unwrap()
                );
            }
            if let Some(expected) = m.skewness {
                assert!(
                    (d.skewness().unwrap() - expected).abs() < tol,
                    "skewness for r={r},p={p}: got {}, expected {expected}",
                    d.skewness().unwrap()
                );
            }
            if let Some(expected) = m.kurtosis {
                assert!(
                    (d.kurtosis().unwrap() - expected).abs() < tol,
                    "kurtosis for r={r},p={p}: got {}, expected {expected}",
                    d.kurtosis().unwrap()
                );
            }

            for pt in &case.pdf_cdf {
                if pt.x < 0.0 {
                    continue;
                }
                let x = pt.x as u64;
                if let Some(expected) = pt.pdf {
                    let actual = d.pdf(&x);
                    assert!(
                        (actual - expected).abs() < tol,
                        "pmf({x}) for r={r},p={p}: got {actual}, expected {expected}"
                    );
                }
                if let Some(expected) = pt.cdf {
                    let actual = d.cdf(x);
                    assert!(
                        (actual - expected).abs() < tol,
                        "cdf({x}) for r={r},p={p}: got {actual}, expected {expected}"
                    );
                }
                if let Some(expected) = pt.log_pdf {
                    let actual = d.log_pdf(&x);
                    assert!(
                        (actual - expected).abs() < tol,
                        "log_pmf({x}) for r={r},p={p}: got {actual}, expected {expected}"
                    );
                }
            }

            for qpt in &case.quantiles {
                if let Some(expected_x) = qpt.x {
                    assert_eq!(
                        d.inverse_cdf(qpt.p),
                        expected_x as u64,
                        "inverse_cdf({}) for r={r},p={p}: got {}, expected {}",
                        qpt.p,
                        d.inverse_cdf(qpt.p),
                        expected_x as u64,
                    );
                }
            }
        }
    }

    // --- Internal consistency (manual, unbounded support) ---

    #[test]
    fn internal_consistency() {
        for &(r, p) in &[(5.0, 0.5), (1.0, 0.3), (10.0, 0.8)] {
            let d = NegativeBinomial::<f64>::new(r, p).unwrap();
            let mean = r * (1.0 - p) / p;
            let std = (r * (1.0 - p)).sqrt() / p;
            let hi = (mean + 5.0 * std).ceil() as u64;
            let tol = 1e-10;

            let mut cumulative = 0.0;
            for x in 0..=hi {
                cumulative += d.pdf(&x);
                let cdf = d.cdf(x);
                assert!(
                    (cumulative - cdf).abs() < tol,
                    "r={r},p={p}: cum PMF at {x} = {cumulative} != cdf = {cdf}",
                );
            }
        }
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = NegativeBinomial::<f64>::new(5.0, 0.5).unwrap();
        assert_discrete_sampling_binomial_ci::<_, f64, _>(&d, &mut rng, 100_000, (0, 30), 5.0);
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = NegativeBinomial::<f64>::new(5.0, 0.5).unwrap();
        let n = 100_000;
        let samples: Vec<u64> = (0..n).map(|_| d.sample(&mut rng)).collect();

        let expected_mean = d.mean().unwrap();
        let expected_var = d.variance().unwrap();

        let sample_mean = samples.iter().sum::<u64>() as f64 / n as f64;
        let sample_var = samples
            .iter()
            .map(|&x| (x as f64 - sample_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        let mean_tol = clt_mean_tolerance(expected_var, n);
        let kurt = d.kurtosis().unwrap();
        let var_tol = clt_variance_tolerance(expected_var, kurt, n);

        assert!(
            (sample_mean - expected_mean).abs() < mean_tol,
            "mean: got {sample_mean}, expected {expected_mean}, tol {mean_tol}"
        );
        assert!(
            (sample_var - expected_var).abs() < var_tol,
            "variance: got {sample_var}, expected {expected_var}, tol {var_tol}"
        );
    }

    // --- Edge cases ---

    #[test]
    fn edge_cases() {
        let d = NegativeBinomial::<f64>::new(5.0, 0.5).unwrap();
        assert!(d.pdf(&0) > 0.0);
        assert!(d.cdf(0) > 0.0);
    }

    // --- PMF sums to ~1 ---

    #[test]
    fn pmf_sums_to_one() {
        for &(r, p) in &[(1.0, 0.5), (5.0, 0.3), (10.0, 0.8)] {
            let d = NegativeBinomial::<f64>::new(r, p).unwrap();
            let mean = r * (1.0 - p) / p;
            let std = (r * (1.0 - p)).sqrt() / p;
            let hi = (mean + 8.0 * std.max(5.0)).ceil() as u64;
            let sum: f64 = (0..=hi).map(|x| d.pdf(&x)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "r={r},p={p}: PMF sum = {sum}, expected 1.0"
            );
        }
    }

    // --- Sampling: sample_fill ---

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = NegativeBinomial::<f64>::new(5.0, 0.5).unwrap();
        let mut buf = [0u64; 100];
        d.sample_fill(&mut rng, &mut buf);
        assert!(buf.iter().any(|&v| v > 0), "expected some non-zero samples");
    }

    // --- p=1 always returns 0 ---

    #[test]
    fn p_one_always_zero() {
        let mut rng = test_rng();
        let d = NegativeBinomial::<f64>::new(5.0, 1.0).unwrap();
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 0);
        }
        assert_eq!(d.mean().unwrap(), 0.0);
    }

    // --- inverse_cdf inverts CDF ---

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = NegativeBinomial::<f64>::new(5.0, 0.5).unwrap();
        for k in 0..=15 {
            let p = d.cdf(k);
            if p > 0.0 && p < 1.0 {
                let q = d.inverse_cdf(p);
                assert!(q >= k, "inverse_cdf(cdf({k})) = {q} should be >= {k}");
            }
        }
    }

    // Left-continuity fuzz boundary: cdf(1) for the geometric NegBinom(1, 0.5)
    // is exactly 0.75 mathematically but rounds to 0.75 - 1 ULP via
    // regularized_beta_inc. inverse_cdf(0.75) must still return 1 (matching R's
    // qnbinom), not overshoot to 2.
    #[test]
    fn inverse_cdf_geometric_boundary() {
        let d = NegativeBinomial::<f64>::new(1.0, 0.5).unwrap();
        assert!(d.cdf(1) < 0.75, "expected cdf(1) just below 0.75");
        assert_eq!(d.inverse_cdf(0.75), 1);
    }

    // Regression for the over-coarse 1e-12 quantile tolerance, which made
    // inverse_cdf return values that were too small across the whole upper tail
    // (any q within 1e-12 of 1). Across the usable range the quantile must
    // equal the smallest k with cdf(k) >= q, computed by an independent linear
    // scan over the SAME cdf.
    #[test]
    fn inverse_cdf_matches_linear_scan() {
        for &(r, p) in &[(1.0, 0.5), (5.0, 0.3), (10.0, 0.8), (30.0, 0.7), (0.5, 0.5)] {
            let d = NegativeBinomial::<f64>::new(r, p).unwrap();
            let mean = r * (1.0 - p) / p;
            let std = (r * (1.0 - p)).sqrt() / p;
            let hi = (mean + 30.0 * std).ceil() as u64 + 10;
            // q grid up to 1 - 1e-7, well outside the razor tail where any
            // left-continuity fuzz (R's included) legitimately rounds down.
            for i in 1..=200u64 {
                let q = i as f64 / 200.0 * (1.0 - 1e-7);
                let want = (0..=hi).find(|&k| d.cdf(k) >= q).unwrap_or(hi);
                assert_eq!(
                    d.inverse_cdf(q),
                    want,
                    "NegBinom(r={r},p={p}): inverse_cdf({q}) = {}, linear-scan = {want}",
                    d.inverse_cdf(q)
                );
            }
        }
    }

    #[test]
    fn from_params_round_trip() {
        let d = NegativeBinomial::<f64>::new(5.0, 0.4).unwrap();
        let d2 = NegativeBinomial::from_params(d.params()).unwrap();
        assert_eq!(d.r(), d2.r());
        assert_eq!(d.p(), d2.p());
    }

    // Pearson chi-square goodness-of-fit on the Gamma-Poisson mixture sampler,
    // a stronger check than per-bin binomial CIs (uses the survival p-value).
    #[test]
    fn sampling_chi_square_gof() {
        let mut rng = test_rng();
        for &(r, p) in &[(5.0, 0.5), (10.0, 0.6), (2.0, 0.3)] {
            let d = NegativeBinomial::<f64>::new(r, p).unwrap();
            let mean = r * (1.0 - p) / p;
            let std = (r * (1.0 - p)).sqrt() / p;
            let hi = (mean + 8.0 * std).ceil() as u64;
            let pval = chi_square_pmf_pvalue::<_, f64, _>(&d, &mut rng, 200_000, (0, hi));
            assert!(pval > 0.001, "NB(r={r},p={p}) chi-square GoF p={pval}");
        }
    }
}
