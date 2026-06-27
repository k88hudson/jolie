use num_traits::Float;
use rand::{Rng, RngExt};

use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::unchecked::Unchecked;

/// Discrete uniform distribution over the integers `{a, a+1, ..., b}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscreteUniform<F: Float> {
    a: i64,
    b: i64,
    _marker: core::marker::PhantomData<F>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscreteUniformParams {
    pub a: i64,
    pub b: i64,
}

impl<F: Float> DiscreteUniform<F> {
    /// - `a`: lower bound (inclusive)
    /// - `b`: upper bound (inclusive), must be >= `a`
    pub fn new(a: i64, b: i64) -> Result<Self, DistributionError> {
        if a > b {
            return Err(DistributionError::InvalidParameter("a must be <= b"));
        }
        // Reject ranges so wide that the support size `b - a + 1` overflows i64
        // (which would otherwise panic in debug / wrap silently in release in
        // cdf/pmf/sample). The check itself is done in i128 to avoid overflow.
        if (b as i128) - (a as i128) + 1 > i64::MAX as i128 {
            return Err(DistributionError::InvalidParameter("range too large"));
        }
        Ok(Self {
            a,
            b,
            _marker: core::marker::PhantomData,
        })
    }

    pub fn new_unchecked(_: Unchecked, a: i64, b: i64) -> Self {
        Self {
            a,
            b,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn a(&self) -> i64 {
        self.a
    }

    pub fn b(&self) -> i64 {
        self.b
    }
}

impl<F: Float> Sampleable for DiscreteUniform<F> {
    type Value = i64;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        let n = (self.b - self.a + 1) as u64;
        self.a + (rng.random_range(0..n) as i64)
    }
}

crate::distributions::traits::impl_rand_distribution!(DiscreteUniform<F: Float> => i64);

impl<F: Float> Distribution<F> for DiscreteUniform<F> {
    fn log_pdf(&self, x: &i64) -> F {
        let k = *x;
        if k < self.a || k > self.b {
            F::neg_infinity()
        } else {
            let n = F::from(self.b - self.a + 1).unwrap();
            -n.ln()
        }
    }
}

impl<F: Float> UnivariateDiscrete<F, i64> for DiscreteUniform<F> {
    type Params = DiscreteUniformParams;

    fn cdf(&self, x: i64) -> F {
        if x < self.a {
            F::zero()
        } else if x >= self.b {
            F::one()
        } else {
            let num = F::from(x - self.a + 1).unwrap();
            let den = F::from(self.b - self.a + 1).unwrap();
            num / den
        }
    }

    fn inverse_cdf(&self, q: F) -> i64 {
        // Smallest k such that CDF(k) >= q.
        // CDF(k) = (k - a + 1) / n, so k = a + ceil(q * n) - 1.
        //
        // `q * n` is computed in floating point, so a `q` that is the rounded
        // image of an exact CDF value m/n can land a hair *above* m (e.g.
        // 0.28 * 25 = 7.000000000000001), making `ceil` overshoot by one.
        // Subtract a small relative tolerance so such values round to m.
        let n = F::from(self.b - self.a + 1).unwrap();
        let target = q * n;
        let tol = F::from(1e-9).unwrap() * target.abs().max(F::one());
        let val = F::from(self.a).unwrap() + (target - tol).ceil() - F::one();
        let result = num_traits::ToPrimitive::to_i64(&val).unwrap();
        result.clamp(self.a, self.b)
    }

    fn support(&self) -> (i64, i64) {
        (self.a, self.b)
    }

    fn params(&self) -> DiscreteUniformParams {
        DiscreteUniformParams {
            a: self.a,
            b: self.b,
        }
    }

    fn from_params(params: DiscreteUniformParams) -> Result<Self, DistributionError> {
        Self::new(params.a, params.b)
    }
}

impl<F: Float> HasMean for DiscreteUniform<F> {
    type Value = F;

    fn mean(&self) -> Option<F> {
        let af = F::from(self.a).unwrap();
        let bf = F::from(self.b).unwrap();
        Some((af + bf) / F::from(2.0).unwrap())
    }
}

impl<F: Float> HasVariance for DiscreteUniform<F> {
    fn variance(&self) -> Option<F> {
        let n = F::from(self.b - self.a + 1).unwrap();
        Some((n * n - F::one()) / F::from(12.0).unwrap())
    }
}

impl<F: Float> HasEntropy for DiscreteUniform<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        let n = F::from(self.b - self.a + 1).unwrap();
        Some(n.ln())
    }
}

impl<F: Float> HasMode for DiscreteUniform<F> {
    type Value = i64;

    fn mode(&self) -> Option<i64> {
        Some(self.a)
    }
}

impl<F: Float> HasSkewness for DiscreteUniform<F> {
    type Value = F;

    fn skewness(&self) -> Option<F> {
        // Undefined when the support is a single point (variance = 0).
        if self.a == self.b {
            None
        } else {
            Some(F::zero())
        }
    }
}

impl<F: Float> HasKurtosis for DiscreteUniform<F> {
    type Value = F;

    fn kurtosis(&self) -> Option<F> {
        // Excess kurtosis -6(n²+1) / (5(n²-1)); undefined when n = 1.
        if self.a == self.b {
            return None;
        }
        let n = F::from(self.b - self.a + 1).unwrap();
        let n2 = n * n;
        let num = F::from(-6.0).unwrap() * (n2 + F::one());
        let den = F::from(5.0).unwrap() * (n2 - F::one());
        Some(num / den)
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
        assert!(DiscreteUniform::<f64>::new(0, 10).is_ok());
        assert!(DiscreteUniform::<f64>::new(-100, 100).is_ok());
        assert!(DiscreteUniform::<f64>::new(5, 5).is_ok());
    }

    #[test]
    fn new_rejects_flipped() {
        assert!(DiscreteUniform::<f64>::new(10, 5).is_err());
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = DiscreteUniform::<f64>::new_unchecked(Unchecked, 10, 5);
        assert_eq!(d.a(), 10);
        assert_eq!(d.b(), 5);
    }

    #[test]
    fn accessors() {
        let d = DiscreteUniform::<f64>::new(2, 7).unwrap();
        assert_eq!(d.a(), 2);
        assert_eq!(d.b(), 7);
    }

    // --- Reference data: PMF, CDF, quantile, moments (from R) ---

    #[test]
    fn reference_pmf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        run_discrete_reference_tests(
            |a, b| DiscreteUniform::<f64>::new(a, b).unwrap(),
            &data,
            1e-12,
        );
    }

    // --- PMF sums to 1 ---

    #[test]
    fn pmf_sums_to_one() {
        for &(a, b) in &[(0, 9), (1, 6), (-5, 5), (0, 0), (0, 100)] {
            let d = DiscreteUniform::<f64>::new(a, b).unwrap();
            assert_pmf_sums_to_one(&d, 1e-12);
        }
    }

    // --- Internal consistency: cumulative PMF == CDF, log_pmf = ln(pmf), monotonicity ---

    #[test]
    fn internal_consistency() {
        for &(a, b) in &[(0, 9), (1, 6), (-5, 5), (0, 0)] {
            let d = DiscreteUniform::<f64>::new(a, b).unwrap();
            assert_discrete_consistency(&d, 1e-14);
        }
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(0, 9).unwrap();
        assert_discrete_sampling_binomial_ci::<_, f64, _>(&d, &mut rng, 100_000, (0, 9), 5.0);
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(0, 100).unwrap();
        let n = 100_000;
        let samples: Vec<i64> = (0..n).map(|_| d.sample(&mut rng)).collect();

        let expected_mean = d.mean().unwrap();
        let expected_var = d.variance().unwrap();

        let sample_mean = samples.iter().sum::<i64>() as f64 / n as f64;
        let sample_var = samples
            .iter()
            .map(|&x| (x as f64 - sample_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        let mean_tol = clt_mean_tolerance(expected_var, n);
        // Discrete uniform excess kurtosis = -6(n^2+1) / (5(n^2-1))
        let nn = 101.0_f64;
        let kurt = -6.0 * (nn * nn + 1.0) / (5.0 * (nn * nn - 1.0));
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

    // --- Non-finite edge cases ---

    #[test]
    fn non_finite_edge_cases() {
        let d = DiscreteUniform::<f64>::new(0, 9).unwrap();
        assert_discrete_edge_cases(&d);
    }

    // --- Sampling: range, single value, coverage, fill ---

    #[test]
    fn samples_in_range() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(0, 100).unwrap();
        for _ in 0..10_000 {
            let v = d.sample(&mut rng);
            assert!((0..=100).contains(&v), "sample {v} out of range [0, 100]");
        }
    }

    #[test]
    fn samples_single_value() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(7, 7).unwrap();
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 7);
        }
    }

    #[test]
    fn samples_cover_full_range() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(0, 4).unwrap();
        let mut seen = [false; 5];
        for _ in 0..10_000 {
            let v = d.sample(&mut rng) as usize;
            seen[v] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "not all values in [0,4] were sampled"
        );
    }

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = DiscreteUniform::<f64>::new(0, 10).unwrap();
        let mut buf = [0i64; 100];
        d.sample_fill(&mut rng, &mut buf);
        for &v in &buf {
            assert!((0..=10).contains(&v));
        }
    }

    #[test]
    fn from_params_round_trip() {
        let d = DiscreteUniform::<f64>::new(1, 10).unwrap();
        let d2 = DiscreteUniform::<f64>::from_params(d.params()).unwrap();
        assert_eq!(d.a(), d2.a());
        assert_eq!(d.b(), d2.b());
    }

    // Regression: inverse_cdf must exactly invert cdf and return the
    // mathematically smallest k for each m/n boundary (no fp off-by-one).
    #[test]
    fn inverse_cdf_inverts_cdf_exhaustive() {
        for a in -3..=3i64 {
            for b in a..=(a + 40) {
                let d = DiscreteUniform::<f64>::new(a, b).unwrap();
                let n = b - a + 1;
                for k in a..=b {
                    let q = d.cdf(k);
                    assert_eq!(d.inverse_cdf(q), k, "roundtrip a={a} b={b} k={k} q={q}");
                }
                for m in 1..=n {
                    let q = m as f64 / n as f64;
                    assert_eq!(
                        d.inverse_cdf(q),
                        a + m - 1,
                        "quantile a={a} b={b} m={m} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn new_rejects_overflowing_range() {
        assert!(DiscreteUniform::<f64>::new(i64::MIN, i64::MAX).is_err());
        assert!(DiscreteUniform::<f64>::new(0, i64::MAX).is_err());
        assert!(DiscreteUniform::<f64>::new(i64::MIN, 0).is_err());
        // A wide-but-representable range is accepted and does not panic.
        let d = DiscreteUniform::<f64>::new(-1_000_000_000, 1_000_000_000).unwrap();
        let _ = d.cdf(0);
        let _ = d.variance();
        let _ = d.mean();
    }
}
