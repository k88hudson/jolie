use crate::distributions::traits::*;

pub fn assert_continuous_edge_cases<D>(dist: &D)
where
    D: UnivariateContinuous<f64>,
{
    // CDF at infinities
    assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0, "cdf(-inf) should be 0");
    assert_eq!(dist.cdf(f64::INFINITY), 1.0, "cdf(+inf) should be 1");

    // PDF/CDF at NaN: implementations may return NaN or 0.0/0.0 depending
    // on how comparisons propagate. We just verify no panic.
    let _ = dist.pdf(&f64::NAN);
    let _ = dist.cdf(f64::NAN);

    // Log PDF outside support should be -inf
    let (lo, hi) = dist.support();
    assert!(
        dist.log_pdf(&(lo - 1.0)).is_infinite() && dist.log_pdf(&(lo - 1.0)) < 0.0,
        "log_pdf below support should be -inf"
    );
    assert!(
        dist.log_pdf(&(hi + 1.0)).is_infinite() && dist.log_pdf(&(hi + 1.0)) < 0.0,
        "log_pdf above support should be -inf"
    );
}

pub fn assert_discrete_edge_cases<D, K: DiscreteInt>(dist: &D)
where
    D: UnivariateDiscrete<f64, K>,
{
    let (lo, hi) = dist.support();
    if let Some(below) = lo.checked_sub(&K::one()) {
        assert!(
            dist.log_pdf(&below).is_infinite() && dist.log_pdf(&below) < 0.0,
            "log_pmf below support should be -inf"
        );
        assert_eq!(dist.pdf(&below), 0.0, "pmf below support should be 0");
    }
    if let Some(above) = hi.checked_add(&K::one()) {
        assert!(
            dist.log_pdf(&above).is_infinite() && dist.log_pdf(&above) < 0.0,
            "log_pmf above support should be -inf"
        );
        assert_eq!(dist.pdf(&above), 0.0, "pmf above support should be 0");
    }
}

/// For discrete distributions: verify PMF sums to 1 over the support.
pub fn assert_pmf_sums_to_one<D, K: DiscreteInt>(dist: &D, tol: f64)
where
    D: UnivariateDiscrete<f64, K>,
{
    let (lo, hi) = dist.support();
    let mut sum = 0.0;
    K::for_each_in_range(lo, hi, |x| {
        sum += dist.pdf(&x);
    });
    assert!(
        (sum - 1.0).abs() < tol,
        "PMF sum over [{lo}, {hi}] = {sum}, expected 1.0"
    );
}
