use num_traits::FromPrimitive;
use serde::Deserialize;

use crate::distributions::traits::*;

// ── JSON reference data types ────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ReferenceData {
    pub distribution: String,
    pub cases: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
pub struct TestCase {
    pub params: Params,
    pub moments: Moments,
    pub pdf_cdf: Vec<PdfCdfPoint>,
    pub quantiles: Vec<QuantilePoint>,
    #[allow(dead_code)]
    pub edge_cases: EdgeCases,
}

#[derive(Debug, Deserialize)]
pub struct Params {
    #[serde(default)]
    pub a: f64,
    #[serde(default)]
    pub b: f64,
    #[serde(default)]
    pub n: Option<f64>,
    #[serde(default)]
    pub p: Option<f64>,
    #[serde(default)]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct Moments {
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    pub entropy: Option<f64>,
    #[serde(default)]
    pub mode: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct PdfCdfPoint {
    pub x: f64,
    pub pdf: Option<f64>,
    pub cdf: Option<f64>,
    pub log_pdf: Option<f64>,
    #[serde(default)]
    pub ccdf: Option<f64>,
    #[serde(default)]
    pub log_cdf: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct QuantilePoint {
    pub p: f64,
    pub x: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EdgeCases {
    pub pdf_nan: Option<f64>,
    pub cdf_neg_inf: f64,
    pub cdf_pos_inf: f64,
    pub log_pdf_below_support: Option<f64>,
    pub log_pdf_above_support: Option<f64>,
}

pub fn load_reference(json_str: &str) -> ReferenceData {
    serde_json::from_str(json_str).expect("failed to parse reference JSON")
}

// ── Reference data test drivers ──────────────────────────────────────

pub fn run_continuous_reference_tests<D, C>(make_dist: C, reference: &ReferenceData, tol: f64)
where
    D: UnivariateContinuous<f64>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>
        + HasMode<Value = f64>,
    C: Fn(f64, f64) -> D,
{
    run_continuous_reference_tests_with_moment_tol(make_dist, reference, tol, tol);
}

/// Like [`run_continuous_reference_tests`] but with a separate `moment_tol` for
/// skewness and kurtosis. Those combine several gamma-function evaluations with
/// heavy cancellation (e.g. Weibull), so they carry more error than the tight
/// `tol` used for pdf/cdf and the lower moments.
pub fn run_continuous_reference_tests_with_moment_tol<D, C>(
    make_dist: C,
    reference: &ReferenceData,
    tol: f64,
    moment_tol: f64,
) where
    D: UnivariateContinuous<f64>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>
        + HasMode<Value = f64>,
    C: Fn(f64, f64) -> D,
{
    for case in &reference.cases {
        let dist = make_dist(case.params.a, case.params.b);

        // Moments
        let m = &case.moments;
        if let Some(expected) = m.mean {
            assert_close(dist.mean().unwrap(), expected, tol, "mean", case);
        }
        if let Some(expected) = m.variance {
            assert_close(dist.variance().unwrap(), expected, tol, "variance", case);
        }
        if let Some(expected) = m.skewness {
            assert_close(
                dist.skewness().unwrap(),
                expected,
                moment_tol,
                "skewness",
                case,
            );
        }
        if let Some(expected) = m.kurtosis {
            assert_close(
                dist.kurtosis().unwrap(),
                expected,
                moment_tol,
                "kurtosis",
                case,
            );
        }
        if let Some(expected) = m.entropy {
            assert_close(dist.entropy().unwrap(), expected, tol, "entropy", case);
        }
        if let Some(expected) = m.mode {
            assert_close(dist.mode().unwrap(), expected, tol, "mode", case);
        }

        // PDF / CDF point evaluations
        for pt in &case.pdf_cdf {
            if let Some(expected_pdf) = pt.pdf {
                assert_close(
                    dist.pdf(&pt.x),
                    expected_pdf,
                    tol,
                    &format!("pdf({})", pt.x),
                    case,
                );
            }
            if let Some(expected_cdf) = pt.cdf {
                assert_close(
                    dist.cdf(pt.x),
                    expected_cdf,
                    tol,
                    &format!("cdf({})", pt.x),
                    case,
                );
            }
            if let Some(expected_ccdf) = pt.ccdf {
                assert_close(
                    dist.ccdf(pt.x),
                    expected_ccdf,
                    tol,
                    &format!("ccdf({})", pt.x),
                    case,
                );
            }
            if let Some(expected_log_pdf) = pt.log_pdf {
                assert_close(
                    dist.log_pdf(&pt.x),
                    expected_log_pdf,
                    tol,
                    &format!("log_pdf({})", pt.x),
                    case,
                );
            }
            if let Some(expected_log_cdf) = pt.log_cdf {
                assert_close(
                    dist.log_cdf(pt.x),
                    expected_log_cdf,
                    tol,
                    &format!("log_cdf({})", pt.x),
                    case,
                );
            }
        }

        // Quantiles
        for qpt in &case.quantiles {
            if let Some(expected_x) = qpt.x {
                assert_close(
                    dist.inverse_cdf(qpt.p),
                    expected_x,
                    tol,
                    &format!("quantile({})", qpt.p),
                    case,
                );
            }
        }
    }
}

pub fn run_discrete_reference_tests<D, K, C>(make_dist: C, reference: &ReferenceData, tol: f64)
where
    D: UnivariateDiscrete<f64, K>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>
        + HasMode<Value = K>,
    K: DiscreteInt,
    C: Fn(K, K) -> D,
{
    for case in &reference.cases {
        let a = K::from_f64(case.params.a).unwrap();
        let b = K::from_f64(case.params.b).unwrap();
        let dist = make_dist(a, b);

        let m = &case.moments;
        if let Some(expected) = m.mean {
            assert_close(dist.mean().unwrap(), expected, tol, "mean", case);
        }
        if let Some(expected) = m.variance {
            assert_close(dist.variance().unwrap(), expected, tol, "variance", case);
        }
        if let Some(expected) = m.skewness {
            assert_close(dist.skewness().unwrap(), expected, tol, "skewness", case);
        }
        if let Some(expected) = m.kurtosis {
            assert_close(dist.kurtosis().unwrap(), expected, tol, "kurtosis", case);
        }
        if let Some(expected) = m.entropy {
            assert_close(dist.entropy().unwrap(), expected, tol, "entropy", case);
        }
        if let Some(expected) = m.mode {
            let expected_k = K::from_f64(expected).unwrap();
            assert_eq!(
                dist.mode().unwrap(),
                expected_k,
                "mode for params a={}, b={}: got {}, expected {}",
                case.params.a,
                case.params.b,
                dist.mode().unwrap(),
                expected_k,
            );
        }

        for pt in &case.pdf_cdf {
            // Skip test points that can't be represented in K (e.g. negative values for u64)
            let Some(x) = K::from_f64(pt.x) else { continue };
            if let Some(expected_pdf) = pt.pdf {
                assert_close(
                    dist.pdf(&x),
                    expected_pdf,
                    tol,
                    &format!("pmf({})", x),
                    case,
                );
            }
            if let Some(expected_cdf) = pt.cdf {
                assert_close(dist.cdf(x), expected_cdf, tol, &format!("cdf({})", x), case);
            }
            if let Some(expected_ccdf) = pt.ccdf {
                assert_close(
                    dist.ccdf(x),
                    expected_ccdf,
                    tol,
                    &format!("ccdf({})", x),
                    case,
                );
            }
            if let Some(expected_log_pdf) = pt.log_pdf {
                assert_close(
                    dist.log_pdf(&x),
                    expected_log_pdf,
                    tol,
                    &format!("log_pmf({})", x),
                    case,
                );
            }
            if let Some(expected_log_cdf) = pt.log_cdf {
                assert_close(
                    dist.log_cdf(x),
                    expected_log_cdf,
                    tol,
                    &format!("log_cdf({})", x),
                    case,
                );
            }
        }

        for qpt in &case.quantiles {
            if let Some(expected_x) = qpt.x {
                let expected_k = K::from_f64(expected_x).unwrap();
                assert_eq!(
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                    "quantile({}) for params a={}, b={}: got {}, expected {}",
                    qpt.p,
                    case.params.a,
                    case.params.b,
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                );
            }
        }
    }
}

pub fn run_discrete_np_reference_tests<D, K, C>(make_dist: C, reference: &ReferenceData, tol: f64)
where
    D: UnivariateDiscrete<f64, K>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>,
    K: DiscreteInt,
    C: Fn(K, f64) -> D,
{
    run_discrete_np_reference_tests_with_entropy_tol(make_dist, reference, tol, tol);
}

pub fn run_discrete_np_reference_tests_with_entropy_tol<D, K, C>(
    make_dist: C,
    reference: &ReferenceData,
    tol: f64,
    entropy_tol: f64,
) where
    D: UnivariateDiscrete<f64, K>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>,
    K: DiscreteInt,
    C: Fn(K, f64) -> D,
{
    for case in &reference.cases {
        let n = K::from_f64(case.params.n.expect("missing 'n' in params")).unwrap();
        let p = case.params.p.expect("missing 'p' in params");
        let dist = make_dist(n, p);

        let m = &case.moments;
        if let Some(expected) = m.mean {
            assert_close(dist.mean().unwrap(), expected, tol, "mean", case);
        }
        if let Some(expected) = m.variance {
            assert_close(dist.variance().unwrap(), expected, tol, "variance", case);
        }
        if let Some(expected) = m.entropy {
            assert_close(
                dist.entropy().unwrap(),
                expected,
                entropy_tol,
                "entropy",
                case,
            );
        }
        if let Some(expected) = m.skewness {
            assert_close(dist.skewness().unwrap(), expected, tol, "skewness", case);
        }
        if let Some(expected) = m.kurtosis {
            assert_close(dist.kurtosis().unwrap(), expected, tol, "kurtosis", case);
        }

        for pt in &case.pdf_cdf {
            // Skip test points that can't be represented in K (e.g. negative values for u64)
            let Some(x) = K::from_f64(pt.x) else { continue };
            if let Some(expected_pdf) = pt.pdf {
                assert_close(
                    dist.pdf(&x),
                    expected_pdf,
                    tol,
                    &format!("pmf({})", x),
                    case,
                );
            }
            if let Some(expected_cdf) = pt.cdf {
                assert_close(dist.cdf(x), expected_cdf, tol, &format!("cdf({})", x), case);
            }
            if let Some(expected_log_pdf) = pt.log_pdf {
                assert_close(
                    dist.log_pdf(&x),
                    expected_log_pdf,
                    tol,
                    &format!("log_pmf({})", x),
                    case,
                );
            }
        }

        for qpt in &case.quantiles {
            if let Some(expected_x) = qpt.x {
                let expected_k = K::from_f64(expected_x).unwrap();
                assert_eq!(
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                    "quantile({}) for params n={}, p={}: got {}, expected {}",
                    qpt.p,
                    n,
                    p,
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                );
            }
        }
    }
}

pub fn run_categorical_reference_tests<D, C>(make_dist: C, reference: &ReferenceData, tol: f64)
where
    D: UnivariateDiscrete<f64, u64>
        + HasMean<Value = f64>
        + HasVariance
        + HasEntropy<Value = f64>
        + HasSkewness<Value = f64>
        + HasKurtosis<Value = f64>,
    C: Fn(&[f64]) -> D,
{
    for case in &reference.cases {
        let weights = case
            .params
            .weights
            .as_ref()
            .expect("missing 'weights' in params");
        let dist = make_dist(weights);

        let m = &case.moments;
        if let Some(expected) = m.mean {
            assert_close(dist.mean().unwrap(), expected, tol, "mean", case);
        }
        if let Some(expected) = m.variance {
            assert_close(dist.variance().unwrap(), expected, tol, "variance", case);
        }
        if let Some(expected) = m.entropy {
            assert_close(dist.entropy().unwrap(), expected, tol, "entropy", case);
        }
        if let Some(expected) = m.skewness {
            assert_close(dist.skewness().unwrap(), expected, tol, "skewness", case);
        }
        if let Some(expected) = m.kurtosis {
            assert_close(dist.kurtosis().unwrap(), expected, tol, "kurtosis", case);
        }

        for pt in &case.pdf_cdf {
            let Some(x) = u64::from_f64(pt.x) else {
                continue;
            };
            if let Some(expected_pdf) = pt.pdf {
                assert_close(
                    dist.pdf(&x),
                    expected_pdf,
                    tol,
                    &format!("pmf({})", x),
                    case,
                );
            }
            if let Some(expected_cdf) = pt.cdf {
                assert_close(dist.cdf(x), expected_cdf, tol, &format!("cdf({})", x), case);
            }
            if let Some(expected_log_pdf) = pt.log_pdf {
                assert_close(
                    dist.log_pdf(&x),
                    expected_log_pdf,
                    tol,
                    &format!("log_pmf({})", x),
                    case,
                );
            }
        }

        for qpt in &case.quantiles {
            if let Some(expected_x) = qpt.x {
                let expected_k = u64::from_f64(expected_x).unwrap();
                assert_eq!(
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                    "quantile({}) for weights {:?}: got {}, expected {}",
                    qpt.p,
                    weights,
                    dist.inverse_cdf(qpt.p),
                    expected_k,
                );
            }
        }
    }
}

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str, case: &TestCase) {
    let params_desc = if let Some(ref w) = case.params.weights {
        format!("weights={:?}", w)
    } else if let (Some(n), Some(p)) = (case.params.n, case.params.p) {
        format!("n={}, p={}", n, p)
    } else {
        format!("a={}, b={}", case.params.a, case.params.b)
    };
    // Mixed tolerance: absolute when |expected| <= 1, relative above. Keeps
    // pdf/cdf checks tight while letting large-magnitude moments (e.g. lognormal
    // kurtosis) be bounded relatively.
    let threshold = tol * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() < threshold,
        "{label} for params {params_desc}: got {actual}, expected {expected}, diff={}, tol={threshold}",
        (actual - expected).abs(),
    );
}
