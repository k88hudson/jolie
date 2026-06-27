//! Shared mathematical constants. Added as distributions need them.

/// ln(2π)
pub(crate) const LN_2PI: f64 = 1.8378770664093453;

/// ln(2πe)
pub(crate) const LN_2PI_E: f64 = 2.8378770664093453;

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
