//! Shared mathematical constants. Added as distributions need them.

/// ln(2π)
pub(crate) const LN_2PI: f64 = 1.8378770664093453;

/// ln(2πe)
pub(crate) const LN_2PI_E: f64 = 2.8378770664093453;

/// √(2π)
pub(crate) const SQRT_2PI: f64 = 2.5066282746310002;

/// ln(π)
pub(crate) const LN_PI: f64 = 1.1447298858494002;

/// ln(2√(e/π))
pub(crate) const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452;

/// Euler–Mascheroni constant (γ)
pub(crate) const EULER_MASCHERONI: f64 = 0.5772156649015329;

/// π²/6 = ζ(2)
pub(crate) const PI_SQUARED_OVER_6: f64 = 1.6449340668482264;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn values_match_computed() {
        let ln_2pi = (2.0 * PI).ln();
        assert!((LN_2PI - ln_2pi).abs() < 1e-15);
        assert!((LN_2PI_E - (ln_2pi + 1.0)).abs() < 1e-15);
    }
}
