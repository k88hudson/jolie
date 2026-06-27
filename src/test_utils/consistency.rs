use crate::distributions::traits::*;

/// For continuous distributions: verify internal consistency of PDF/CDF/inverse CDF.
pub fn assert_continuous_consistency<D>(dist: &D, points: &[f64], tol: f64)
where
    D: UnivariateContinuous<f64>,
{
    let (lo, hi) = dist.support();
    assert!(lo <= hi, "support: lo={lo} > hi={hi}");

    for &x in points {
        // cdf + ccdf == 1
        let cdf = dist.cdf(x);
        let ccdf = dist.ccdf(x);
        assert!(
            (cdf + ccdf - 1.0).abs() < tol,
            "cdf({x}) + ccdf({x}) = {} != 1.0",
            cdf + ccdf,
        );

        // log_pdf == ln(pdf) when pdf > 0
        let pdf = dist.pdf(&x);
        if pdf > 0.0 {
            let log_pdf = dist.log_pdf(&x);
            assert!(
                (log_pdf - pdf.ln()).abs() < tol,
                "log_pdf({x}) = {log_pdf} != ln(pdf({x})) = {}",
                pdf.ln(),
            );
        }

        // log_cdf == ln(cdf) when cdf > 0
        if cdf > 0.0 {
            let log_cdf = dist.log_cdf(x);
            assert!(
                (log_cdf - cdf.ln()).abs() < tol,
                "log_cdf({x}) = {log_cdf} != ln(cdf({x})) = {}",
                cdf.ln(),
            );
        }
    }

    // CDF monotonicity
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in sorted_points.windows(2) {
        assert!(
            dist.cdf(w[0]) <= dist.cdf(w[1]),
            "CDF not monotone: cdf({}) = {} > cdf({}) = {}",
            w[0],
            dist.cdf(w[0]),
            w[1],
            dist.cdf(w[1]),
        );
    }
}

/// For discrete distributions: verify cumulative PMF == CDF, and consistency checks.
pub fn assert_discrete_consistency<D, K: DiscreteInt>(dist: &D, tol: f64)
where
    D: UnivariateDiscrete<f64, K>,
{
    let (lo, hi) = dist.support();
    assert!(lo <= hi, "support: lo={lo} > hi={hi}");

    // Cumulative PMF should match CDF
    let mut cumulative = 0.0;
    K::for_each_in_range(lo, hi, |x| {
        cumulative += dist.pdf(&x);
        let cdf = dist.cdf(x);
        assert!(
            (cumulative - cdf).abs() < tol,
            "cumulative PMF at {x} = {cumulative} != cdf({x}) = {cdf}",
        );
    });

    // CDF monotonicity
    let mut prev_cdf = 0.0_f64;
    K::for_each_in_range(lo, hi, |x| {
        let cdf = dist.cdf(x);
        assert!(
            cdf >= prev_cdf,
            "CDF not monotone at {x}: {prev_cdf} > {cdf}"
        );
        prev_cdf = cdf;
    });

    // log_pmf == ln(pmf) for values in support
    K::for_each_in_range(lo, hi, |x| {
        let pmf = dist.pdf(&x);
        if pmf > 0.0 {
            let log_pmf = dist.log_pdf(&x);
            assert!(
                (log_pmf - pmf.ln()).abs() < tol,
                "log_pmf({x}) = {log_pmf} != ln(pmf({x})) = {}",
                pmf.ln(),
            );
        }
    });
}
