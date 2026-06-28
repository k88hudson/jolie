use super::gamma::ln_gamma;
use num_traits::Float;

#[inline]
pub(crate) fn ln_beta<F: Float>(a: F, b: F) -> F {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Core continued-fraction evaluation of I_x(a,b).
/// When `complement` is true, returns 1 - I_x(a,b) without a double subtraction.
fn regularized_beta_cf64(a: f64, b: f64, x: f64, ln_beta_ab: f64, complement: bool) -> f64 {
    if x <= 0.0 {
        return if complement { 1.0 } else { 0.0 };
    }
    if x >= 1.0 {
        return if complement { 0.0 } else { 1.0 };
    }

    const EPS: f64 = 1e-15;
    const FPMIN: f64 = f64::MIN_POSITIVE / EPS;

    // Symmetry transform: evaluate the continued fraction on the side that
    // converges faster, then choose which branch is the "direct" result.
    let symm_transform = x >= (a + 1.0) / (a + b + 2.0);
    let (a, b, x) = if symm_transform {
        (b, a, 1.0 - x)
    } else {
        (a, b, x)
    };

    // bt = x^a * (1-x)^b / (a * B(a,b))
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (-ln_beta_ab + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=140 {
        let mf = m as f64;
        let m2 = mf * 2.0;

        // Even step
        let mut aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() <= EPS {
            break;
        }
    }

    // `result` is I_x(a',b') where (a',b') may be swapped.
    // Without the symmetry transform, result = I_x(a,b).
    // With the symmetry transform, result = I_{1-x}(b,a) = 1 - I_x(a,b).
    //
    // We want to return:
    //   complement=false → I_x(a,b)
    //   complement=true  → 1 - I_x(a,b)
    //
    // Return `result` directly when it already represents the requested
    // quantity, avoiding the `1 - nearly_one` cancellation.
    let result = bt * h / a;
    if symm_transform == complement {
        result
    } else {
        1.0 - result
    }
}

fn regularized_beta_inc_f64(a: f64, b: f64, x: f64) -> f64 {
    let lb: f64 = ln_beta(a, b);
    regularized_beta_cf64(a, b, x, lb, false)
}

fn regularized_beta_compl_f64(a: f64, b: f64, x: f64) -> f64 {
    let lb: f64 = ln_beta(a, b);
    regularized_beta_cf64(a, b, x, lb, true)
}

#[inline]
pub(crate) fn regularized_beta_inc<F: Float>(a: F, b: F, x: F) -> F {
    F::from(regularized_beta_inc_f64(
        a.to_f64().unwrap(),
        b.to_f64().unwrap(),
        x.to_f64().unwrap(),
    ))
    .unwrap()
}

/// Returns 1 - I_x(a,b) without the double-complement precision loss that
/// occurs when computing `1.0 - regularized_beta_inc(a, b, x)`.
#[inline]
pub(crate) fn regularized_beta_compl<F: Float>(a: F, b: F, x: F) -> F {
    F::from(regularized_beta_compl_f64(
        a.to_f64().unwrap(),
        b.to_f64().unwrap(),
        x.to_f64().unwrap(),
    ))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // --- ln_beta ---

    #[test]
    fn ln_beta_1_1() {
        // B(1,1) = 1, ln(1) = 0
        assert!(ln_beta(1.0_f64, 1.0).abs() < TOL);
    }

    #[test]
    fn ln_beta_2_2() {
        // B(2,2) = 1/6, ln(1/6) ≈ -1.791759469228327
        assert!((ln_beta(2.0_f64, 2.0) - (-1.791759469228327)).abs() < TOL);
    }

    #[test]
    fn ln_beta_half_half() {
        // B(0.5, 0.5) = pi, ln(pi) ≈ 1.1447298858494002
        assert!((ln_beta(0.5_f64, 0.5) - std::f64::consts::PI.ln()).abs() < TOL);
    }

    #[test]
    fn ln_beta_5_5() {
        // B(5,5) = Gamma(5)^2/Gamma(10) = (24*24)/362880
        let expected = (24.0_f64 * 24.0 / 362880.0).ln();
        assert!((ln_beta(5.0_f64, 5.0) - expected).abs() < TOL);
    }

    #[test]
    fn ln_beta_symmetric() {
        assert!((ln_beta(3.0_f64, 7.0) - ln_beta(7.0_f64, 3.0)).abs() < TOL);
    }

    #[test]
    fn ln_beta_at_quarter_points() {
        // B(1/4, 3/4) = pi/sin(pi/4) = pi*sqrt(2)
        // ln(B(1/4, 3/4)) = ln(pi) + 0.5*ln(2)
        let expected = std::f64::consts::PI.ln() + 0.5 * 2.0_f64.ln();
        let val = ln_beta(0.25_f64, 0.75);
        assert!(
            (val - expected).abs() < 1e-9,
            "ln_beta(1/4, 3/4): got {val}, expected {expected}"
        );
    }

    #[test]
    fn ln_beta_at_third_points() {
        // B(1/3, 2/3) = pi/sin(pi/3) = 2*pi/sqrt(3)
        let expected = (2.0 * std::f64::consts::PI / 3.0_f64.sqrt()).ln();
        let val = ln_beta(1.0_f64 / 3.0, 2.0 / 3.0);
        assert!(
            (val - expected).abs() < 1e-9,
            "ln_beta(1/3, 2/3): got {val}, expected {expected}"
        );
    }

    // --- regularized_beta_inc ---

    #[test]
    fn beta_inc_symmetry_1_1() {
        // I_0.5(1,1) = 0.5
        assert!((regularized_beta_inc(1.0_f64, 1.0, 0.5) - 0.5).abs() < TOL);
    }

    #[test]
    fn beta_inc_symmetry_2_2() {
        // I_0.5(2,2) = 0.5 (symmetric)
        assert!((regularized_beta_inc(2.0_f64, 2.0, 0.5) - 0.5).abs() < TOL);
    }

    #[test]
    fn beta_inc_symmetry_10_10() {
        // I_0.5(10,10) = 0.5
        assert!((regularized_beta_inc(10.0_f64, 10.0, 0.5) - 0.5).abs() < TOL);
    }

    #[test]
    fn beta_inc_2_5_at_03() {
        // I_0.3(2,5) ≈ 0.57983 (SciPy: betainc(2,5,0.3))
        assert!((regularized_beta_inc(2.0_f64, 5.0, 0.3) - 0.57981750).abs() < 1e-4);
    }

    #[test]
    fn beta_inc_5_2_at_07() {
        // I_x(a,b) + I_{1-x}(b,a) = 1
        let forward = regularized_beta_inc(5.0_f64, 2.0, 0.7);
        let complement = regularized_beta_inc(2.0_f64, 5.0, 0.3);
        assert!(
            (forward + complement - 1.0).abs() < 1e-8,
            "symmetry check: I_0.7(5,2) = {forward}, I_0.3(2,5) = {complement}"
        );
    }

    #[test]
    fn beta_inc_boundary() {
        assert_eq!(regularized_beta_inc(2.0_f64, 3.0, 0.0), 0.0);
        assert_eq!(regularized_beta_inc(2.0_f64, 3.0, 1.0), 1.0);
    }

    #[test]
    fn beta_inc_uniform() {
        // I_x(1,1) = x for all x in [0,1]
        for &x in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert!((regularized_beta_inc(1.0_f64, 1.0, x) - x).abs() < TOL);
        }
    }

    #[test]
    fn beta_inc_symmetry_identity() {
        let cases: &[(f64, f64, f64)] = &[
            (2.0, 5.0, 0.3),
            (10.0, 3.0, 0.7),
            (0.5, 0.5, 0.25),
            (1.0, 1.0, 0.6),
            (100.0, 50.0, 0.5),
            (0.1, 0.1, 0.9),
        ];
        for &(a, b, x) in cases {
            let lhs = regularized_beta_inc(a, b, x);
            let rhs = regularized_beta_inc(b, a, 1.0 - x);
            assert!(
                (lhs + rhs - 1.0).abs() < 1e-10,
                "I_{x}({a},{b}) + I_{{1-x}}({b},{a}) != 1: {lhs} + {rhs} = {}",
                lhs + rhs
            );
        }
    }

    #[test]
    fn beta_inc_extreme_small_p() {
        // For a=1: I_x(1, b) = 1 - (1-x)^b
        let val = regularized_beta_inc(1.0_f64, 1000.0, 0.001);
        let expected = 1.0 - (1.0 - 0.001_f64).powi(1000);
        assert!(
            (val - expected).abs() < 1e-10,
            "I_0.001(1,1000): got {val}, expected {expected}"
        );
    }

    #[test]
    fn beta_inc_extreme_large_p() {
        // For b=1, I_x(a, 1) = x^a
        let val = regularized_beta_inc(1000.0_f64, 1.0, 0.999);
        let expected = 0.999_f64.powi(1000);
        assert!(
            (val - expected).abs() < 1e-10,
            "I_0.999(1000,1): got {val}, expected {expected}"
        );
    }

    #[test]
    fn beta_inc_a1_closed_form() {
        // I_x(1, b) = 1 - (1-x)^b (closed form for a=1)
        for &(b, x) in &[(5.0, 0.3), (10.0, 0.5), (100.0, 0.01), (0.5, 0.9)] {
            let val = regularized_beta_inc(1.0_f64, b, x);
            let expected = 1.0 - (1.0 - x).powf(b);
            assert!(
                (val - expected).abs() < 1e-10,
                "I_{x}(1,{b}): got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn beta_inc_b1_closed_form() {
        // I_x(a, 1) = x^a (closed form for b=1)
        for &(a, x) in &[(5.0, 0.3), (10.0, 0.5), (100.0, 0.99), (0.5, 0.25)] {
            let val = regularized_beta_inc(a, 1.0_f64, x);
            let expected = x.powf(a);
            assert!(
                (val - expected).abs() < 1e-10,
                "I_{x}({a},1): got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn beta_inc_large_symmetric_params() {
        // I_0.5(n, n) = 0.5 for all n (by symmetry of Beta(n,n) around 0.5)
        for &n in &[50.0, 100.0, 500.0, 1000.0] {
            let val = regularized_beta_inc(n, n, 0.5);
            assert!(
                (val - 0.5).abs() < 1e-10,
                "I_0.5({n},{n}): got {val}, expected 0.5"
            );
        }
    }

    // --- Cross-validation: beta_compl matches binomial CDF computation ---

    #[test]
    fn beta_compl_matches_binomial_cdf_n10_p05() {
        // For Binomial(10, 0.5), CDF(k) = 1 - I_p(k+1, n-k)
        // CDF(5) = 1 - I_0.5(6, 5) = 0.623046875 (SciPy reference)
        let cdf_5 = regularized_beta_compl(6.0_f64, 5.0, 0.5);
        assert!(
            (cdf_5 - 0.623046875).abs() < 1e-12,
            "Binomial(10,0.5) CDF(5): got {cdf_5}, expected 0.623046875"
        );
    }

    #[test]
    fn beta_compl_matches_binomial_cdf_n10_p05_at_3() {
        // CDF(3) = 1 - I_0.5(4, 7) = 0.171875
        let cdf_3 = regularized_beta_compl(4.0_f64, 7.0, 0.5);
        assert!(
            (cdf_3 - 0.171875).abs() < 1e-12,
            "Binomial(10,0.5) CDF(3): got {cdf_3}, expected 0.171875"
        );
    }

    #[test]
    fn beta_compl_matches_binomial_cdf_various() {
        let cases: &[(i64, f64)] = &[
            (0, 0.0009765625),
            (1, 0.0107421875),
            (4, 0.376953125),
            (5, 0.623046875),
            (6, 0.828125),
            (7, 0.9453125),
            (9, 0.9990234375),
        ];
        for &(k, expected_cdf) in cases {
            let cdf = regularized_beta_compl((k + 1) as f64, (10 - k) as f64, 0.5);
            assert!(
                (cdf - expected_cdf).abs() < 1e-12,
                "Binomial(10,0.5) CDF({k}): got {cdf}, expected {expected_cdf}"
            );
        }
    }

    #[test]
    fn beta_compl_binomial_cdf_extreme_n() {
        // For Binomial(1000, 0.001), CDF(0) = (1-p)^n
        let cdf_0 = regularized_beta_compl(1.0_f64, 1000.0, 0.001);
        let expected = 0.999_f64.powi(1000);
        assert!(
            (cdf_0 - expected).abs() < 1e-10,
            "Binomial(1000,0.001) CDF(0): got {cdf_0}, expected {expected}"
        );
    }
}
