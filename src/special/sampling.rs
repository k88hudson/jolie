//! Ziggurat samplers (ZIGNOR variant, Doornik 2005; adapted from rand_distr,
//! Apache-2.0 / MIT).

use super::ziggurat_tables;
use rand::{Rng, RngExt};

/// Sample from the standard normal N(0,1) via the ziggurat method.
#[inline]
pub(crate) fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    #[inline]
    fn pdf(x: f64) -> f64 {
        (-x * x / 2.0).exp()
    }

    #[inline]
    fn tail_sample<R: Rng + ?Sized>(rng: &mut R, negative: bool) -> f64 {
        let r = ziggurat_tables::ZIG_NORM_R;
        loop {
            let x = -rng.random::<f64>().ln() / r;
            let y = -rng.random::<f64>().ln();
            if 2.0 * y >= x * x {
                return if negative { -(r + x) } else { r + x };
            }
        }
    }

    let x_tab = &ziggurat_tables::ZIG_NORM_X;
    let f_tab = &ziggurat_tables::ZIG_NORM_F;

    loop {
        let bits = rng.next_u64();
        let i = (bits & 0xff) as usize;

        // upper 52 bits -> float in [-1, 1)
        let u = f64::from_bits((bits >> 12) | 0x4000000000000000u64) - 3.0;
        let x = u * x_tab[i];

        if x.abs() < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return tail_sample(rng, u < 0.0);
        }
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.random::<f64>() < pdf(x) {
            return x;
        }
    }
}

/// Sample from the standard exponential Exp(1) via the ziggurat method.
#[inline]
pub(crate) fn standard_exponential<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    #[inline]
    fn pdf(x: f64) -> f64 {
        (-x).exp()
    }

    #[inline]
    fn tail_sample<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        // log1p(-U) instead of log(U) so that U=0 produces X=R (the tail
        // infimum, a valid exponential draw) rather than +inf.
        ziggurat_tables::ZIG_EXP_R - (-rng.random::<f64>()).ln_1p()
    }

    let x_tab = &ziggurat_tables::ZIG_EXP_X;
    let f_tab = &ziggurat_tables::ZIG_EXP_F;

    loop {
        let bits = rng.next_u64();
        let i = (bits & 0xff) as usize;

        // Convert upper 52 bits to a float in (0, 1).
        let u = f64::from_bits((bits >> 12) | 0x3FF0000000000000u64) - (1.0 - f64::EPSILON / 2.0);
        let x = u * x_tab[i];

        // Fast path: point is inside rectangle i.
        if x < x_tab[i + 1] {
            return x;
        }

        // Bottom rectangle: sample from the tail.
        if i == 0 {
            return tail_sample(rng);
        }

        // Slow path: between rectangles, need PDF evaluation.
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.random::<f64>() < pdf(x) {
            return x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn standard_normal_mean_variance() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() < 0.01, "mean = {mean}, expected ~0");
        assert!((var - 1.0).abs() < 0.02, "var = {var}, expected ~1");
    }

    #[test]
    fn standard_normal_symmetry() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let n = 100_000;
        let neg = (0..n).filter(|_| standard_normal(&mut rng) < 0.0).count();
        let frac = neg as f64 / n as f64;
        assert!((frac - 0.5).abs() < 0.01, "negative fraction = {frac}");
    }

    #[test]
    fn standard_normal_tail_mass() {
        let cases: &[(f64, f64)] = &[
            (1.0, 0.31731050786291404),
            (2.0, 0.04550026389635842),
            (3.0, 0.0026997960632601866),
            (4.0, 6.334248366623985e-5),
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(20260417);
        let n: usize = 5_000_000;
        let samples: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        for &(t, expected) in cases {
            let observed = samples.iter().filter(|x| x.abs() > t).count() as f64 / n as f64;
            let stderr = (expected * (1.0 - expected) / n as f64).sqrt();
            let z = (observed - expected) / stderr;
            assert!(
                z.abs() < 6.0,
                "P(|X|>{t}): observed={observed:.7}, z={z:+.2}"
            );
        }
    }

    #[test]
    fn standard_exponential_mean_variance() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| standard_exponential(&mut rng)).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        assert!((mean - 1.0).abs() < 0.02, "mean = {mean}, expected ~1");
        assert!((var - 1.0).abs() < 0.03, "var = {var}, expected ~1");
    }

    // Regression: pins P(X>t) for t ∈ {1, 3, R, 10}. Exercises the tail branch
    // at t=R and t=10 (t > ZIG_EXP_R ≈ 7.697).
    #[test]
    fn standard_exponential_tail_mass() {
        let r = ziggurat_tables::ZIG_EXP_R;
        let cases: &[(f64, f64)] = &[
            (1.0, 0.36787944117144233),    // e^-1
            (3.0, 0.04978706836786394),    // e^-3
            (r, 0.00045413435384149677),   // e^-R, tail boundary
            (10.0, 4.5399929762484854e-5), // deep tail
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(20260418);
        let n: usize = 5_000_000;
        let samples: Vec<f64> = (0..n).map(|_| standard_exponential(&mut rng)).collect();

        for &(t, expected) in cases {
            let observed_count = samples.iter().filter(|&&x| x > t).count();
            let observed = observed_count as f64 / n as f64;
            let stderr = (expected * (1.0 - expected) / n as f64).sqrt();
            let z = (observed - expected) / stderr;
            assert!(
                z.abs() < 6.0,
                "P(X>{t}): observed={observed:.7}, expected={expected:.7}, z={z:+.2}σ"
            );
        }
    }

    // Regression for the ln(0) tail blowup. Pre-fix, the tail branch was
    // `R - ln(U)`, which returns +inf when U=0 (probability 2^-53). The MockRng
    // forces i=0 + fast-path failure + U=0 in the tail to exercise the boundary
    // deterministically; the fixed form returns exactly R.
    #[test]
    fn standard_exponential_tail_u_zero_returns_r() {
        use rand::TryRng;
        use std::convert::Infallible;

        struct MockRng {
            values: Vec<u64>,
            pos: usize,
        }
        impl TryRng for MockRng {
            type Error = Infallible;
            fn try_next_u32(&mut self) -> Result<u32, Infallible> {
                Ok(self.try_next_u64()? as u32)
            }
            fn try_next_u64(&mut self) -> Result<u64, Infallible> {
                let v = self.values.get(self.pos).copied().unwrap_or(0);
                self.pos += 1;
                Ok(v)
            }
            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
                for chunk in dst.chunks_mut(8) {
                    let bytes = self.try_next_u64()?.to_le_bytes();
                    let n = chunk.len().min(8);
                    chunk[..n].copy_from_slice(&bytes[..n]);
                }
                Ok(())
            }
        }

        // First u64: low 8 bits = 0 → i=0; upper 52 bits all 1 → u ≈ 1 - 2^-53,
        // so x = u·x_tab[0] ≈ 8.697 > x_tab[1] ≈ 7.697, fast-path fails, enters
        // tail branch. Second u64: 0 → random::<f64>() = 0 → U=0 in tail.
        let mut rng = MockRng {
            values: vec![0xFFFF_FFFF_FFFF_F000, 0],
            pos: 0,
        };
        let x = standard_exponential(&mut rng);
        assert!(
            x.is_finite(),
            "got non-finite sample {x}; ln(0) blowup regressed"
        );
        assert_eq!(
            x,
            ziggurat_tables::ZIG_EXP_R,
            "U=0 in tail should produce X=R; got {x}"
        );
    }
}
