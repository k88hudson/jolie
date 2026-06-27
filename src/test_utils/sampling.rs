use num_traits::Float;
use rand::Rng;

use crate::distributions::traits::*;

// ── CLT-based tolerances ─────────────────────────────────────────────

pub fn clt_mean_tolerance(variance: f64, n: usize) -> f64 {
    5.0 * (variance / n as f64).sqrt()
}

pub fn clt_variance_tolerance(variance: f64, kurtosis: f64, n: usize) -> f64 {
    // Var(sample_var) ≈ var^2 * (kurt + 2) / n
    // Use 5-sigma tolerance
    5.0 * variance * ((kurtosis + 2.0) / n as f64).sqrt()
}

// ── Binomial CI sampling test ────────────────────────────────────────

/// For continuous distributions: bin samples and check counts against CDF-derived expectations.
/// `bins` are (lo, hi) edges; expected proportion per bin = CDF(hi) - CDF(lo).
pub fn assert_continuous_sampling_binomial_ci<D>(
    dist: &D,
    rng: &mut impl Rng,
    n_samples: usize,
    bins: &[(f64, f64)],
    z: f64, // z-score for confidence (e.g., 4.0 for ~1e-5 false positive)
) where
    D: UnivariateContinuous<f64> + Sampleable<Value = f64>,
{
    let mut counts = vec![0usize; bins.len()];
    for _ in 0..n_samples {
        let x = dist.sample(rng);
        for (i, &(lo, hi)) in bins.iter().enumerate() {
            if x >= lo && x < hi {
                counts[i] += 1;
                break;
            }
        }
    }

    let n = n_samples as f64;
    for (i, (&(lo, hi), &count)) in bins.iter().zip(counts.iter()).enumerate() {
        let p = dist.cdf(hi) - dist.cdf(lo);
        let expected = n * p;
        let std_dev = (n * p * (1.0 - p)).sqrt();
        let lower = (expected - z * std_dev).max(0.0);
        let upper = expected + z * std_dev;
        assert!(
            (count as f64) >= lower && (count as f64) <= upper,
            "bin {i} [{lo}, {hi}): count={count}, expected={expected:.0}, bounds=[{lower:.0}, {upper:.0}]"
        );
    }
}

/// For discrete distributions: check each support value's count against PMF-derived expectations.
pub fn assert_discrete_sampling_binomial_ci<D, F, K>(
    dist: &D,
    rng: &mut impl Rng,
    n_samples: usize,
    support: (K, K),
    z: f64,
) where
    D: Distribution<F> + Sampleable<Value = K>,
    F: Float + Into<f64>,
    K: DiscreteInt,
{
    let (a, b) = support;
    let range = K::range_size(a, b);
    let mut counts = vec![0usize; range];
    for _ in 0..n_samples {
        let x = dist.sample(rng);
        if x >= a && x <= b {
            counts[(x - a).to_usize_saturating()] += 1;
        }
    }

    let n = n_samples as f64;
    for (i, &count) in counts.iter().enumerate() {
        let x = a + K::from_usize(i).unwrap();
        let p: f64 = dist.pdf(&x).into();
        let expected = n * p;
        let std_dev = (n * p * (1.0 - p)).sqrt();
        let lower = (expected - z * std_dev).max(0.0);
        let upper = expected + z * std_dev;
        assert!(
            (count as f64) >= lower && (count as f64) <= upper,
            "value {x}: count={count}, expected={expected:.0}, bounds=[{lower:.0}, {upper:.0}]"
        );
    }
}
