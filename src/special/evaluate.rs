#[inline]
pub(crate) fn polynomial_f64(z: f64, coeff: &[f64]) -> f64 {
    let n = coeff.len();
    if n == 0 {
        return 0.0;
    }

    let mut sum = coeff[n - 1];
    for c in coeff[0..n - 1].iter().rev() {
        sum = *c + z * sum;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn polynomial_empty() {
        assert_eq!(polynomial_f64(2.0_f64, &[]), 0.0);
    }

    #[test]
    fn polynomial_constant() {
        assert!((polynomial_f64(99.0_f64, &[5.0]) - 5.0).abs() < TOL);
    }

    #[test]
    fn polynomial_linear() {
        assert!((polynomial_f64(4.0_f64, &[3.0, 2.0]) - 11.0).abs() < TOL);
    }

    #[test]
    fn polynomial_quadratic() {
        assert!((polynomial_f64(2.0_f64, &[2.0, 3.0, 1.0]) - 12.0).abs() < TOL);
    }

    #[test]
    fn polynomial_cubic() {
        assert!((polynomial_f64(3.0_f64, &[1.0, 0.0, 0.0, 1.0]) - 28.0).abs() < TOL);
    }
}
