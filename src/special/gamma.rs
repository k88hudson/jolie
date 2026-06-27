#![allow(clippy::excessive_precision)]

use num_traits::Float;

use crate::special::erf::erfc;

const GAMMA_R: f64 = 10.900511;

const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

use crate::constants::{EULER_MASCHERONI, LN_2_SQRT_E_OVER_PI, LN_PI, PI_SQUARED_OVER_6, SQRT_2PI};

fn ln_gamma_f64(x: f64) -> f64 {
    use core::f64::consts::{E, PI};

    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, (i, &dk)| s + dk / (i as f64 - x));

        LN_PI
            - (PI * x).sin().ln()
            - s.ln()
            - LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / E).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, (i, &dk)| s + dk / (x + i as f64 - 1.0));

        s.ln() + LN_2_SQRT_E_OVER_PI + (x - 0.5) * ((x - 0.5 + GAMMA_R) / E).ln()
    }
}

#[inline]
pub(crate) fn ln_gamma<F: Float>(x: F) -> F {
    F::from(ln_gamma_f64(x.to_f64().unwrap())).unwrap()
}

fn digamma_f64(x: f64) -> f64 {
    use core::f64::consts::PI;

    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x <= 0.0 && x.floor() == x {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return digamma_f64(1.0 - x) + PI / (-PI * x).tan();
    }
    if x <= 1e-6 {
        return -EULER_MASCHERONI - 1.0 / x + PI_SQUARED_OVER_6 * x;
    }

    let mut result = 0.0;
    let mut z = x;
    while z < 12.0 {
        result -= 1.0 / z;
        z += 1.0;
    }

    let mut r = 1.0 / z;
    result += z.ln() - 0.5 * r;
    r *= r;
    result -=
        r * (1.0 / 12.0 - r * (1.0 / 120.0 - r * (1.0 / 252.0 - r * (1.0 / 240.0 - r / 132.0))));
    result
}

#[inline]
pub(crate) fn digamma<F: Float>(x: F) -> F {
    F::from(digamma_f64(x.to_f64().unwrap())).unwrap()
}

/// Compute log(1+x) - x with good numerical accuracy for small x.
/// Uses Taylor series for |x| < 0.5, direct computation otherwise.
/// See GSL specfunc/log.c (gsl_sf_log_1plusx_mx_e).
pub(crate) fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        // Taylor series: log(1+x) - x = -x^2/2 + x^3/3 - x^4/4 + ...
        let mut sum = 0.0;
        let mut xn = x * x;
        for n in 2..=40 {
            let term = if n % 2 == 0 {
                -xn / n as f64
            } else {
                xn / n as f64
            };
            sum += term;
            if term.abs() < 1e-18 {
                break;
            }
            xn *= x;
        }
        sum
    } else {
        (1.0 + x).ln() - x
    }
}

/// Temme uniform asymptotic expansion for Q(a,x) when a is large and x ≈ a.
/// See GSL specfunc/gamma_inc.c:192-230 (gamma_inc_Q_asymp_unif).
fn gamma_inc_q_asymp_unif(a: f64, x: f64) -> f64 {
    let rta = a.sqrt();
    let eps = (x - a) / a;
    let ln_term_val = log1pmx(eps);
    let eta = if eps >= 0.0 { 1.0 } else { -1.0 } * (-2.0 * ln_term_val).sqrt();

    let erfc_val = erfc(eta * rta / std::f64::consts::SQRT_2);

    let (c0, c1);
    if eps.abs() < 7.45e-4 {
        // Small eps: use Taylor expansion of coefficients
        c0 = -1.0 / 3.0
            + eps
                * (1.0 / 12.0
                    - eps * (23.0 / 540.0 - eps * (353.0 / 12960.0 - eps * 589.0 / 30240.0)));
        c1 = -1.0 / 540.0 - eps / 288.0;
    } else {
        let rt_term = (-2.0 * ln_term_val / (eps * eps)).sqrt();
        let lam = x / a;
        c0 = (1.0 - 1.0 / rt_term) / eps;
        c1 = -(eta * eta * eta * (lam * lam + 10.0 * lam + 1.0) - 12.0 * eps * eps * eps)
            / (12.0 * eta * eta * eta * eps * eps * eps);
    }

    let r = (-0.5 * a * eta * eta).exp() / (SQRT_2PI * rta) * (c0 + c1 / a);
    0.5 * erfc_val + r
}

/// Concrete f64 implementation of both regularized incomplete gamma functions,
/// returning `(P(a,x), Q(a,x))` with `P + Q == 1`. Computing them together lets
/// each be returned from the branch where it is accurate: the continued fraction
/// yields `Q` directly (so the upper tail avoids the `1 - P` cancellation that
/// collapses `Q` to 0 once `P` rounds to 1), while the series yields `P`.
/// Algorithm selection based on GSL specfunc/gamma_inc.c:500-577.
fn gamma_inc_pq_f64(a: f64, x: f64) -> (f64, f64) {
    if x.is_nan() || a.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if x == f64::INFINITY {
        return (1.0, 0.0);
    }

    // Temme uniform asymptotic for large a near x
    // See GSL specfunc/gamma_inc.c:527 (gsl_sf_gamma_inc_Q_e)
    if a >= 1e6 {
        let diff = x - a;
        if diff * diff < a {
            let q = gamma_inc_q_asymp_unif(a, x);
            return (1.0 - q, q);
        }
    }

    const EPS: f64 = 1e-15;
    const BIG: f64 = 4503599627370496.0;
    const BIG_INV: f64 = 2.22044604925031308085e-16;

    let ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.78271289338399 {
        return if a < x { (1.0, 0.0) } else { (0.0, 1.0) };
    }

    if x <= 1.0 || x <= a {
        // Series expansion for P(a,x); Q = 1 - P is accurate here because P is
        // bounded away from 1 on this branch (x <= a or x <= 1).
        let mut r2 = a;
        let mut c2 = 1.0;
        let mut ans2 = 1.0;
        loop {
            r2 += 1.0;
            c2 *= x / r2;
            ans2 += c2;
            if c2 / ans2 <= EPS {
                break;
            }
        }
        let p = ax.exp() * ans2 / a;
        return (p, 1.0 - p);
    }

    // Continued fraction for Q(a,x) directly (the accurate upper-tail branch).
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0i32;

    let mut p3 = 1.0;
    let mut q3 = x;
    let mut p2 = x + 1.0;
    let mut q2 = z * x;
    let mut ans = p2 / q2;

    loop {
        y += 1.0;
        z += 2.0;
        c += 1;
        let yc = y * c as f64;

        let p = p2 * z - p3 * yc;
        let q = q2 * z - q3 * yc;

        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        if p.abs() > BIG {
            p3 *= BIG_INV;
            p2 *= BIG_INV;
            q3 *= BIG_INV;
            q2 *= BIG_INV;
        }

        if q != 0.0 {
            let nextans = p / q;
            let error = ((ans - nextans) / nextans).abs();
            ans = nextans;
            if error <= EPS {
                break;
            }
        }
    }
    let q = ax.exp() * ans;
    (1.0 - q, q)
}

#[inline]
fn regularized_gamma_inc_f64(a: f64, x: f64) -> f64 {
    gamma_inc_pq_f64(a, x).0
}

#[inline]
fn regularized_gamma_compl_f64(a: f64, x: f64) -> f64 {
    gamma_inc_pq_f64(a, x).1
}

#[inline]
pub(crate) fn regularized_gamma_inc<F: Float>(a: F, x: F) -> F {
    // Delegate to concrete f64 implementation to avoid generic overhead
    F::from(regularized_gamma_inc_f64(
        a.to_f64().unwrap(),
        x.to_f64().unwrap(),
    ))
    .unwrap()
}

/// Regularized upper incomplete gamma function `Q(a,x) = 1 - P(a,x)`, the
/// survival function of the Gamma distribution. Computed without the `1 - P`
/// cancellation, so it stays accurate deep into the upper tail where `P`
/// rounds to 1.
#[inline]
pub(crate) fn regularized_gamma_compl<F: Float>(a: F, x: F) -> F {
    F::from(regularized_gamma_compl_f64(
        a.to_f64().unwrap(),
        x.to_f64().unwrap(),
    ))
    .unwrap()
}

fn regularized_gamma_inc_inv_f64(a: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let gln: f64 = ln_gamma(a);

    // Region-specific initial approximation, see GSL cdf/gammainv.c:53-68
    let mut x = if a <= 1.0 {
        let t_val = 1.0 - a * (0.253 + a * 0.12);
        if p < t_val {
            (p / t_val).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t_val) / (1.0 - t_val)).ln()
        }
    } else if p < 0.05 {
        ((gln + p.ln()) / a).exp()
    } else if p > 0.95 {
        -(1.0 - p).ln() + gln
    } else {
        let xg = if p < 0.5 {
            let t = (-2.0 * (1.0 - p).ln()).sqrt();
            t - (2.515517 + t * (0.802853 + t * 0.010328))
                / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)))
        } else {
            let t = (-2.0 * p.ln()).sqrt();
            -t + (2.515517 + t * (0.802853 + t * 0.010328))
                / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)))
        };

        let sqrt_a = a.sqrt();
        if xg < -0.5 * sqrt_a {
            a
        } else {
            sqrt_a * xg + a
        }
    };

    if x <= 0.0 {
        x = 1e-10;
    }

    // Lagrange interpolation refinement, see GSL cdf/gammainv.c:76-106
    for _ in 0..32 {
        let dp = p - regularized_gamma_inc_f64(a, x);
        let phi = ((a - 1.0) * x.ln() - x - gln).exp();

        if dp == 0.0 {
            break;
        }

        let lambda = dp / (2.0 * (dp / x).abs()).max(phi);

        let step0 = lambda;
        let step1 = -((a - 1.0) / x - 1.0) * lambda * lambda / 4.0;

        let step = if step1.abs() < 0.5 * step0.abs() {
            step0 + step1
        } else {
            step0
        };

        if x + step > 0.0 {
            x += step;
        } else {
            x /= 2.0;
        }

        if step0.abs() <= 1e-10 * x && (step0 * phi).abs() <= 1e-10 * p {
            break;
        }
    }

    x
}

#[inline]
pub(crate) fn regularized_gamma_inc_inv<F: Float>(a: F, p: F) -> F {
    F::from(regularized_gamma_inc_inv_f64(
        a.to_f64().unwrap(),
        p.to_f64().unwrap(),
    ))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn ln_gamma_known_values() {
        assert!(ln_gamma(1.0_f64).abs() < TOL);
        assert!((ln_gamma(0.5_f64) - 0.5723649429247001).abs() < TOL);
        assert!((ln_gamma(5.0_f64) - 24.0_f64.ln()).abs() < TOL);
        assert!((ln_gamma(10.0_f64) - 362880.0_f64.ln()).abs() < TOL);
        assert!((ln_gamma(0.1_f64) - 2.2527126517342055).abs() < TOL);
        assert!((ln_gamma(100.0_f64) - 359.1342053695754).abs() < 1e-7);
    }

    #[test]
    fn digamma_known_values() {
        assert!((digamma(1.0_f64) - (-0.5772156649015329)).abs() < TOL);
        assert!((digamma(2.0_f64) - 0.4227843350984671).abs() < TOL);
        assert!((digamma(0.5_f64) - (-1.9635100260214235)).abs() < TOL);
        assert!((digamma(10.0_f64) - 2.2517525890667211).abs() < TOL);
    }

    #[test]
    fn digamma_edge_cases() {
        assert_eq!(digamma(-1.0_f64), f64::NEG_INFINITY);
        assert_eq!(digamma(0.0_f64), f64::NEG_INFINITY);
        assert!(digamma(f64::NAN).is_nan());
    }

    #[test]
    fn gamma_inc_known_values() {
        assert!((regularized_gamma_inc(1.0_f64, 1.0) - 0.6321205588285577).abs() < TOL);
        assert!((regularized_gamma_inc(2.0_f64, 3.0) - 0.8008517265285442).abs() < TOL);
        assert!((regularized_gamma_inc(0.5_f64, 0.5) - 0.6826894921370859).abs() < TOL);
        assert!((regularized_gamma_inc(5.0_f64, 5.0) - 0.5595067149347701).abs() < TOL);
    }

    #[test]
    fn gamma_inc_boundary() {
        assert_eq!(regularized_gamma_inc(1.0_f64, 0.0), 0.0);
        assert_eq!(regularized_gamma_inc(1.0_f64, f64::INFINITY), 1.0);
        assert_eq!(regularized_gamma_compl(1.0_f64, 0.0), 1.0);
        assert_eq!(regularized_gamma_compl(1.0_f64, f64::INFINITY), 0.0);
    }

    #[test]
    fn gamma_compl_complements_inc() {
        for &(a, x) in &[(0.5_f64, 0.3), (2.0, 1.5), (5.0, 4.0), (3.0, 3.0)] {
            let p = regularized_gamma_inc(a, x);
            let q = regularized_gamma_compl(a, x);
            assert!((p + q - 1.0).abs() < 1e-14, "P+Q != 1 for a={a},x={x}");
        }
    }

    #[test]
    fn gamma_compl_upper_tail_accurate() {
        // Deep tail: Q(1,x)=e^-x, Q(2,x)=(1+x)e^-x must stay accurate where 1-P collapses.
        for &x in &[20.0_f64, 30.0, 40.0, 50.0, 80.0] {
            let q1 = regularized_gamma_compl(1.0, x);
            let exact1 = (-x).exp();
            assert!((q1 - exact1).abs() / exact1 < 1e-12, "Q(1,{x})");
            let q2 = regularized_gamma_compl(2.0, x);
            let exact2 = (1.0 + x) * (-x).exp();
            assert!((q2 - exact2).abs() / exact2 < 1e-12, "Q(2,{x})");
            assert!(q2 > 0.0, "Q(2,{x}) underflowed");
        }
    }

    #[test]
    fn gamma_inc_inv_boundary() {
        assert_eq!(regularized_gamma_inc_inv(1.0_f64, 0.0), 0.0);
        assert_eq!(regularized_gamma_inc_inv(1.0_f64, 1.0), f64::INFINITY);
    }

    #[test]
    fn gamma_inc_inv_various_shapes() {
        for &a in &[0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            for &p in &[0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] {
                let x = regularized_gamma_inc_inv(a, p);
                let roundtrip = regularized_gamma_inc(a, x);
                assert!(
                    (roundtrip - p).abs() < 1e-10,
                    "a={a}, p={p}: roundtrip={roundtrip}"
                );
            }
        }
    }

    #[test]
    fn log1pmx_accuracy() {
        assert!((log1pmx(0.001) - (-4.99666916466872e-7)).abs() < 1e-13);
        assert!((log1pmx(1.0) - (2.0_f64.ln() - 1.0)).abs() < 1e-14);
    }

    #[test]
    fn gamma_inc_large_a_temme() {
        // Temme asymptotic path: a >= 1e6, (x-a)^2 < a
        assert!((regularized_gamma_inc(1_000_000.0_f64, 1_000_000.0) - 0.5).abs() < 1e-3);
        assert!(regularized_gamma_inc(1_000_000.0_f64, 999_500.0) < 0.5);
        assert!(regularized_gamma_inc(1_000_000.0_f64, 1_000_500.0) > 0.5);
    }

    #[test]
    fn gamma_inc_tail_stability() {
        assert!((regularized_gamma_inc(2.0_f64, 20.0) - 1.0).abs() < 1e-7);
        let v = regularized_gamma_inc(20.0_f64, 2.0);
        assert!(v > 0.0 && v < 0.01);
    }
}
