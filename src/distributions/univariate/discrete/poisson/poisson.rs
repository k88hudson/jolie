use num_traits::Float;
use rand::{Rng, RngExt};

use crate::constants::SQRT_2PI;
use crate::distributions::traits::*;
use crate::error::DistributionError;
use crate::special::gamma::{ln_factorial, regularized_gamma_compl, regularized_gamma_inc};
use crate::special::sampling::standard_normal;
use crate::unchecked::Unchecked;

#[derive(Debug, Clone, Copy)]
enum SamplingMethod {
    Degenerate,
    Knuth {
        exp_neg_lambda: f64,
    },
    Rejection {
        lambda: f64,
        s: f64,
        d: f64,
        l: f64,
        c: f64,
        c0: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        omega: f64,
    },
}

/// Poisson distribution Poisson(λ).
///
/// Models the number of events occurring in a fixed interval, given a
/// constant average rate λ. Mean equals variance (both are λ).
#[derive(Debug, Clone, Copy)]
pub struct Poisson<F: Float> {
    lambda: F,
    sampling: SamplingMethod,
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoissonParams<F> {
    pub lambda: F,
}

impl<F: Float> PartialEq for Poisson<F> {
    fn eq(&self, other: &Self) -> bool {
        self.lambda == other.lambda
    }
}

impl<F: Float> Poisson<F> {
    /// - `lambda` (λ): rate parameter, must be finite and >= 0.
    ///   When λ = 0 the distribution degenerates to a point mass at 0.
    pub fn new(lambda: F) -> Result<Self, DistributionError> {
        if !lambda.is_finite() {
            return Err(DistributionError::InvalidParameter("lambda must be finite"));
        }
        if lambda < F::zero() {
            return Err(DistributionError::InvalidParameter(
                "lambda must be non-negative",
            ));
        }
        let sampling = Self::compute_sampling(lambda);
        Ok(Self { lambda, sampling })
    }

    pub fn new_unchecked(_: Unchecked, lambda: F) -> Self {
        let sampling = Self::compute_sampling(lambda);
        Self { lambda, sampling }
    }

    pub fn lambda(&self) -> F {
        self.lambda
    }

    fn compute_sampling(lambda: F) -> SamplingMethod {
        let lam = num_traits::ToPrimitive::to_f64(&lambda).unwrap();
        if lam == 0.0 {
            SamplingMethod::Degenerate
        } else if lam < 10.0 {
            SamplingMethod::Knuth {
                exp_neg_lambda: (-lam).exp(),
            }
        } else {
            // Ahrens-Dieter PD: paper specifies μ ≥ 10 (Case A); R agrees.
            // Bench shows PD ~5× faster than Knuth here. (rand_distr/NRC use 12,
            // an artifact of NRC's unrelated Lorentzian rejection method.)
            let b1 = (1.0 / 24.0) / lam;
            let b2 = 0.3 * b1 * b1;
            let c3 = (1.0 / 7.0) * b1 * b2;
            let c2 = b2 - 15.0 * c3;
            let c1 = b1 - 6.0 * b2 + 45.0 * c3;
            let c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3;
            let s = lam.sqrt();
            SamplingMethod::Rejection {
                lambda: lam,
                s,
                d: 6.0 * lam * lam,
                l: (lam - 1.1484).floor(),
                c: 0.1069 / lam,
                c0,
                c1,
                c2,
                c3,
                omega: 1.0 / (SQRT_2PI * s),
            }
        }
    }
}

// Ahrens-Dieter Step F: compute (px, py, fx, fy) for candidate k.
// From: J. H. Ahrens and U. Dieter (1982), "Computer Generation of Poisson
// Deviates from Modified Normal Distributions", ACM TOMS 8(2), 163-179.
#[inline]
fn step_f(k: f64, rej: &RejParams) -> (f64, f64, f64, f64) {
    const FACT: [f64; 10] = [
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    ];
    const A: [f64; 10] = [
        -0.5000000002,
        0.3333333343,
        -0.2499998565,
        0.1999997049,
        -0.1666848753,
        0.1428833286,
        -0.1241963125,
        0.1101687109,
        -0.1142650302,
        0.1055093006,
    ];

    let (px, py) = if k < 10.0 {
        let px = -rej.lambda;
        let py = rej.lambda.powf(k) / FACT[k as usize];
        (px, py)
    } else {
        let delta = (12.0 * k).recip();
        let delta = delta - 4.8 * delta.powi(3);
        let v = (rej.lambda - k) / k;
        let px = if v.abs() <= 0.25 {
            k * v * v * A.iter().rev().fold(0.0, |acc, &a| acc * v + a) - delta
        } else {
            k * (1.0 + v).ln() - (rej.lambda - k) - delta
        };
        let py = 1.0 / (SQRT_2PI * k.sqrt());
        (px, py)
    };

    let x = (k - rej.lambda + 0.5) / rej.s;
    let fx = -0.5 * x * x;
    let fy = rej.omega * (((rej.c3 * x * x + rej.c2) * x * x + rej.c1) * x * x + rej.c0);

    (px, py, fx, fy)
}

struct RejParams {
    lambda: f64,
    s: f64,
    d: f64,
    l: f64,
    c: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    omega: f64,
}

fn sample_rejection<R: Rng + ?Sized>(rng: &mut R, rej: &RejParams) -> u64 {
    // Step N: generate from Normal(lambda, s)
    let g = rej.lambda + rej.s * standard_normal(rng);
    if g >= 0.0 {
        let k1 = g.floor();

        // Step I: immediate acceptance for large k
        if k1 >= rej.l {
            return k1 as u64;
        }

        // Step S: squeeze
        let u: f64 = rng.random();
        if rej.d * u >= (rej.lambda - k1).powi(3) {
            return k1 as u64;
        }

        let (px, py, fx, fy) = step_f(k1, rej);
        if fy * (1.0 - u) <= py * (px - fx).exp() {
            return k1 as u64;
        }
    }

    loop {
        // Step E: exponential inter-arrival
        let e = -rng.random::<f64>().ln(); // Exp(1) sample
        let u: f64 = rng.random::<f64>() * 2.0 - 1.0;
        let t = 1.8 + e * u.signum();
        if t > -0.6744 {
            let k2 = (rej.lambda + rej.s * t).floor();
            if k2 >= 0.0 {
                let (px, py, fx, fy) = step_f(k2, rej);
                // Step H
                if rej.c * u.abs() <= py * (px + e).exp() - fy * (fx + e).exp() {
                    return k2 as u64;
                }
            }
        }
    }
}

crate::distributions::traits::impl_rand_distribution!(Poisson<F: Float> => u64);

impl<F: Float> Sampleable for Poisson<F> {
    type Value = u64;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        match self.sampling {
            SamplingMethod::Degenerate => 0,
            SamplingMethod::Knuth { exp_neg_lambda } => {
                let mut count = 0u64;
                let mut p: f64 = rng.random();
                while p > exp_neg_lambda {
                    p *= rng.random::<f64>();
                    count += 1;
                }
                count
            }
            SamplingMethod::Rejection {
                lambda,
                s,
                d,
                l,
                c,
                c0,
                c1,
                c2,
                c3,
                omega,
            } => {
                let rej = RejParams {
                    lambda,
                    s,
                    d,
                    l,
                    c,
                    c0,
                    c1,
                    c2,
                    c3,
                    omega,
                };
                sample_rejection(rng, &rej)
            }
        }
    }
}

impl<F: Float> Distribution<F> for Poisson<F> {
    fn log_pdf(&self, x: &u64) -> F {
        if self.lambda == F::zero() {
            return if *x == 0 {
                F::zero()
            } else {
                F::neg_infinity()
            };
        }
        let k = *x;
        let lambda: f64 = num_traits::ToPrimitive::to_f64(&self.lambda).unwrap();
        // log P(X=k) = -λ + k·ln(λ) - ln(k!)
        // Matches GSL randist/poisson.c and statrs. Uses ln_factorial(k)
        // (table lookup for k ≤ 170) instead of ln_gamma(k+1).
        let result = -lambda + (k as f64) * lambda.ln() - ln_factorial(k);
        F::from(result).unwrap()
    }
}

impl<F: Float> UnivariateDiscrete<F, u64> for Poisson<F> {
    type Params = PoissonParams<F>;

    fn cdf(&self, x: u64) -> F {
        if self.lambda == F::zero() {
            return F::one(); // P(X <= k) = 1 for all k >= 0
        }
        // P(X <= k) = Q(k+1, λ), the upper regularized gamma. Computed directly
        // (not 1 - P) so the lower tail stays accurate instead of collapsing to 0.
        regularized_gamma_compl(F::from(x + 1).unwrap(), self.lambda)
    }

    // P(X > k) = P(k+1, λ), the lower regularized gamma. Direct form keeps the
    // upper tail accurate where the default 1 - cdf would cancel to 0.
    fn ccdf(&self, x: u64) -> F {
        if self.lambda == F::zero() {
            return F::zero();
        }
        regularized_gamma_inc(F::from(x + 1).unwrap(), self.lambda)
    }

    fn inverse_cdf(&self, q: F) -> u64 {
        if self.lambda == F::zero() {
            return 0;
        }
        if q <= F::zero() {
            return 0;
        }
        if q >= F::one() {
            return u64::MAX;
        }
        // Binary search
        let mean_f64 = num_traits::ToPrimitive::to_f64(&self.lambda).unwrap();
        let std = mean_f64.sqrt();
        let mut lo = 0u64;
        let mut hi = (mean_f64 + 20.0 * std).ceil() as u64;
        // Expand hi if needed
        while self.cdf(hi) < q {
            hi = hi.saturating_mul(2);
            if hi == u64::MAX {
                break;
            }
        }
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

    fn params(&self) -> PoissonParams<F> {
        PoissonParams {
            lambda: self.lambda,
        }
    }

    fn from_params(params: PoissonParams<F>) -> Result<Self, DistributionError> {
        Self::new(params.lambda)
    }
}

impl<F: Float> HasMean for Poisson<F> {
    type Value = F;
    fn mean(&self) -> Option<F> {
        Some(self.lambda)
    }
}

impl<F: Float> HasVariance for Poisson<F> {
    fn variance(&self) -> Option<F> {
        Some(self.lambda)
    }
}

impl<F: Float> HasEntropy for Poisson<F> {
    type Value = F;

    fn entropy(&self) -> Option<F> {
        let lam = num_traits::ToPrimitive::to_f64(&self.lambda).unwrap();
        let h = if lam < 50.0 {
            // Exact series: H = -Σ p(k) ln(p(k))
            let mut s = 0.0_f64;
            let mut log_p = -lam; // log P(0) = -λ
            let p0 = log_p.exp();
            if p0 > 0.0 {
                s += p0 * log_p;
            }
            for k in 1..=200 {
                log_p += lam.ln() - (k as f64).ln();
                let pk = log_p.exp();
                if pk < 1e-20 && k as f64 > lam {
                    break;
                }
                if pk > 0.0 {
                    s += pk * log_p;
                }
            }
            -s
        } else {
            // Stirling approximation for large λ
            0.5 * (crate::constants::LN_2PI_E + lam.ln())
                - 1.0 / (12.0 * lam)
                - 1.0 / (24.0 * lam * lam)
                - 19.0 / (360.0 * lam * lam * lam)
        };
        Some(F::from(h).unwrap())
    }
}

impl<F: Float> HasMode for Poisson<F> {
    type Value = u64;
    fn mode(&self) -> Option<u64> {
        Some(num_traits::ToPrimitive::to_u64(&self.lambda.floor()).unwrap())
    }
}

impl<F: Float> HasSkewness for Poisson<F> {
    type Value = F;
    fn skewness(&self) -> Option<F> {
        if self.lambda == F::zero() {
            return None;
        }
        Some(F::one() / self.lambda.sqrt())
    }
}

impl<F: Float> HasKurtosis for Poisson<F> {
    type Value = F;
    fn kurtosis(&self) -> Option<F> {
        if self.lambda == F::zero() {
            return None;
        }
        Some(F::one() / self.lambda)
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
        assert!(Poisson::<f64>::new(1.0).is_ok());
        assert!(Poisson::<f64>::new(0.001).is_ok());
        assert!(Poisson::<f64>::new(100.0).is_ok());
        assert!(Poisson::<f32>::new(5.0).is_ok());
    }

    #[test]
    fn new_accepts_zero() {
        assert!(Poisson::<f64>::new(0.0).is_ok());
    }

    #[test]
    fn new_rejects_negative() {
        assert!(Poisson::<f64>::new(-1.0).is_err());
    }

    // --- Degenerate case: λ = 0 (point mass at 0) ---

    #[test]
    fn degenerate_pmf() {
        let d = Poisson::<f64>::new(0.0).unwrap();
        assert_eq!(d.pdf(&0), 1.0);
        assert_eq!(d.log_pdf(&0), 0.0);
        assert_eq!(d.pdf(&1), 0.0);
        assert_eq!(d.log_pdf(&1), f64::NEG_INFINITY);
        assert_eq!(d.pdf(&100), 0.0);
    }

    #[test]
    fn degenerate_cdf() {
        let d = Poisson::<f64>::new(0.0).unwrap();
        assert_eq!(d.cdf(0), 1.0);
        assert_eq!(d.cdf(1), 1.0);
        assert_eq!(d.cdf(100), 1.0);
    }

    #[test]
    fn degenerate_inverse_cdf() {
        let d = Poisson::<f64>::new(0.0).unwrap();
        assert_eq!(d.inverse_cdf(0.0), 0);
        assert_eq!(d.inverse_cdf(0.5), 0);
        assert_eq!(d.inverse_cdf(0.99), 0);
    }

    #[test]
    fn degenerate_sampling() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(0.0).unwrap();
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 0);
        }
    }

    #[test]
    fn degenerate_moments() {
        let d = Poisson::<f64>::new(0.0).unwrap();
        assert_eq!(d.mean().unwrap(), 0.0);
        assert_eq!(d.variance().unwrap(), 0.0);
        assert_eq!(d.mode().unwrap(), 0);
        assert_eq!(d.entropy().unwrap(), 0.0);
        assert!(d.skewness().is_none());
        assert!(d.kurtosis().is_none());
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(Poisson::<f64>::new(f64::NAN).is_err());
        assert!(Poisson::<f64>::new(f64::INFINITY).is_err());
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let d = Poisson::<f64>::new_unchecked(Unchecked, -1.0);
        assert_eq!(d.lambda(), -1.0);
    }

    #[test]
    fn accessors() {
        let d = Poisson::<f64>::new(5.0).unwrap();
        assert_eq!(d.lambda(), 5.0);
    }

    // --- Reference data: PMF, CDF, quantile, moments (from R) ---

    #[test]
    fn reference_pmf_cdf_quantile_moments() {
        let data = load_reference(REFERENCE_JSON);
        for case in &data.cases {
            let lambda = case.params.a;
            let d = Poisson::<f64>::new(lambda).unwrap();
            let tol = 1e-10;

            let m = &case.moments;
            if let Some(expected) = m.mean {
                assert!(
                    (d.mean().unwrap() - expected).abs() < tol,
                    "mean for λ={lambda}: got {}, expected {expected}",
                    d.mean().unwrap()
                );
            }
            if let Some(expected) = m.variance {
                assert!(
                    (d.variance().unwrap() - expected).abs() < tol,
                    "variance for λ={lambda}: got {}, expected {expected}",
                    d.variance().unwrap()
                );
            }
            if let Some(expected) = m.entropy {
                assert!(
                    (d.entropy().unwrap() - expected).abs() < 1e-6,
                    "entropy for λ={lambda}: got {}, expected {expected}",
                    d.entropy().unwrap()
                );
            }
            if let Some(expected) = m.skewness {
                assert!(
                    (d.skewness().unwrap() - expected).abs() < tol,
                    "skewness for λ={lambda}: got {}, expected {expected}",
                    d.skewness().unwrap()
                );
            }
            if let Some(expected) = m.kurtosis {
                assert!(
                    (d.kurtosis().unwrap() - expected).abs() < tol,
                    "kurtosis for λ={lambda}: got {}, expected {expected}",
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
                        "pmf({x}) for λ={lambda}: got {actual}, expected {expected}"
                    );
                }
                if let Some(expected) = pt.cdf {
                    let actual = d.cdf(x);
                    assert!(
                        (actual - expected).abs() < tol,
                        "cdf({x}) for λ={lambda}: got {actual}, expected {expected}"
                    );
                }
                if let Some(expected) = pt.log_pdf {
                    let actual = d.log_pdf(&x);
                    assert!(
                        (actual - expected).abs() < tol,
                        "log_pmf({x}) for λ={lambda}: got {actual}, expected {expected}"
                    );
                }
            }

            for qpt in &case.quantiles {
                if let Some(expected_x) = qpt.x {
                    assert_eq!(
                        d.inverse_cdf(qpt.p),
                        expected_x as u64,
                        "quantile({}) for λ={lambda}: got {}, expected {}",
                        qpt.p,
                        d.inverse_cdf(qpt.p),
                        expected_x as u64,
                    );
                }
            }
        }
    }

    // --- Internal consistency (manual, since support is unbounded) ---

    #[test]
    fn internal_consistency() {
        for &lam in &[0.5, 5.0, 50.0] {
            let d = Poisson::<f64>::new(lam).unwrap();
            let hi = (lam + 5.0 * lam.sqrt()).ceil() as u64;
            let tol = 1e-10;

            // Cumulative PMF matches CDF
            let mut cumulative = 0.0;
            for x in 0..=hi {
                cumulative += d.pdf(&x);
                let cdf = d.cdf(x);
                assert!(
                    (cumulative - cdf).abs() < tol,
                    "λ={lam}: cumulative PMF at {x} = {cumulative} != cdf({x}) = {cdf}",
                );
            }

            // log_pmf == ln(pmf)
            for x in 0..=hi {
                let pmf = d.pdf(&x);
                if pmf > 0.0 {
                    let log_pmf = d.log_pdf(&x);
                    assert!(
                        (log_pmf - pmf.ln()).abs() < tol,
                        "λ={lam}: log_pmf({x}) = {log_pmf} != ln(pmf) = {}",
                        pmf.ln(),
                    );
                }
            }
        }
    }

    // --- Sampling: binomial CI ---

    #[test]
    fn sampling_binomial_ci() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(5.0).unwrap();
        assert_discrete_sampling_binomial_ci::<_, f64, _>(&d, &mut rng, 100_000, (0, 20), 5.0);
    }

    #[test]
    fn sampling_binomial_ci_large_lambda() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(50.0).unwrap();
        assert_discrete_sampling_binomial_ci::<_, f64, _>(&d, &mut rng, 100_000, (20, 80), 5.0);
    }

    // --- Sampling: chi-square goodness-of-fit ---
    //
    // Regression: catches tail-localized biases in the PD rejection sampler
    // that pass per-bin CI but inflate the summed χ². A standard_normal tail
    // bug surfaced here as p ≈ 0 across multiple seeds at λ ≥ 30.

    #[test]
    fn sampling_chi_square_pd_lambda_30() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(30.0).unwrap();
        let p = chi_square_pmf_pvalue::<_, f64, _>(&d, &mut rng, 1_000_000, (0, 100));
        assert!(p > 0.001, "chi-square p-value = {p:.6}, expected > 0.001");
    }

    #[test]
    fn sampling_chi_square_pd_lambda_100() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(100.0).unwrap();
        let p = chi_square_pmf_pvalue::<_, f64, _>(&d, &mut rng, 1_000_000, (0, 200));
        assert!(p > 0.001, "chi-square p-value = {p:.6}, expected > 0.001");
    }

    // --- Sampling: CLT moment validation ---

    #[test]
    fn sample_moments_clt() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(10.0).unwrap();
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

    // --- Edge cases (manual, since support is unbounded) ---

    #[test]
    fn edge_cases() {
        let d = Poisson::<f64>::new(5.0).unwrap();
        // u64 can't be negative, so just verify boundary values
        assert!(d.pdf(&0) > 0.0);
        assert!(d.cdf(0) > 0.0);
    }

    // --- PMF sums to ~1 (bounded range) ---

    #[test]
    fn pmf_sums_to_one() {
        for &lam in &[1.0, 10.0, 50.0] {
            let d = Poisson::<f64>::new(lam).unwrap();
            let hi = (lam + 8.0 * lam.sqrt().max(3.0)).ceil() as u64;
            let sum: f64 = (0..=hi).map(|x| d.pdf(&x)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "λ={lam}: PMF sum over [0, {hi}] = {sum}, expected 1.0"
            );
        }
    }

    // --- Sampling: sample_fill ---

    #[test]
    fn sample_fill() {
        let mut rng = test_rng();
        let d = Poisson::<f64>::new(5.0).unwrap();
        let mut buf = [0u64; 100];
        d.sample_fill(&mut rng, &mut buf);
        assert!(buf.iter().any(|&v| v > 0), "expected some non-zero samples");
    }

    // --- Quantile inverts CDF ---

    #[test]
    fn inverse_cdf_inverts_cdf() {
        let d = Poisson::<f64>::new(5.0).unwrap();
        for k in 0..=15 {
            let p = d.cdf(k);
            if p > 0.0 && p < 1.0 {
                let q = d.inverse_cdf(p);
                assert!(q >= k, "quantile(cdf({k})) = {q} should be >= {k}");
            }
        }
    }

    // --- Mean equals variance ---

    #[test]
    fn mean_equals_variance() {
        for &lam in &[0.5, 1.0, 5.0, 50.0, 100.0] {
            let d = Poisson::<f64>::new(lam).unwrap();
            assert_eq!(d.mean().unwrap(), d.variance().unwrap());
        }
    }

    #[test]
    fn from_params_round_trip() {
        let d = Poisson::<f64>::new(3.0).unwrap();
        let d2 = Poisson::from_params(d.params()).unwrap();
        assert_eq!(d.lambda(), d2.lambda());
    }

    // cdf (lower tail) and ccdf (upper tail) compute Q/P directly so the deep
    // tails stay accurate rather than collapsing to 0 via a 1 - P cancellation.
    #[test]
    fn deep_tail_accurate() {
        // P(X <= 0) = e^{-λ} exactly.
        let d = Poisson::<f64>::new(100.0).unwrap();
        let exact = (-100.0_f64).exp();
        assert!(
            (d.cdf(0) - exact).abs() / exact < 1e-10,
            "cdf(0)={}",
            d.cdf(0)
        );
        assert!(
            (d.log_cdf(0) - (-100.0)).abs() < 1e-9,
            "log_cdf(0)={}",
            d.log_cdf(0)
        );

        // Upper-tail survival must not underflow to 0.
        let d = Poisson::<f64>::new(5.0).unwrap();
        assert!(d.ccdf(40) > 0.0, "ccdf(40) underflowed");
        assert!(d.ccdf(60) > 0.0, "ccdf(60) underflowed");
    }
}
