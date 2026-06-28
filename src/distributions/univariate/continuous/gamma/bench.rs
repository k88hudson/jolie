//! Benchmarks for `Gamma`, comparing each trait method against `rand_distr`
//! (sampling) and `statrs` (everything else) where an equivalent exists.
//! Methods with no equivalent (`log_cdf`, `kurtosis`) are benched alone.
//!
//! jolie and `rand_distr` use scale (θ); `statrs` uses rate (β = 1/θ).

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use jolie::distributions::*;

use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};
use rand_distr::Gamma as RandGamma;

// statrs 0.18 samples through rand 0.8 (aliased `rand08`).
use rand08::rngs::StdRng as StdRng08;
use rand08::{Rng as _, SeedableRng as _};

use statrs::distribution::{Continuous, ContinuousCDF, Gamma as StatrsGamma};
use statrs::statistics::{Distribution as StatrsMoments, Mode};

const SHAPE: f64 = 2.0;
const SCALE: f64 = 2.0;
const RATE: f64 = 0.5; // 1 / SCALE
const X: f64 = 2.5; // an in-support evaluation point
const P: f64 = 0.65; // a quantile probability

fn jolie_dist() -> Gamma<f64> {
    Gamma::shape_scale(SHAPE, SCALE).unwrap()
}

fn statrs_dist() -> StatrsGamma {
    StatrsGamma::new(SHAPE, RATE).unwrap()
}

fn sample(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();
    let rd = RandGamma::new(SHAPE, SCALE).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group("gamma/sample");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(Sampleable::sample(&jd, &mut rng)))
    });
    g.bench_function("rand_distr", |b| b.iter(|| black_box(rng.sample(rd))));
    g.bench_function("statrs", |b| {
        b.iter(|| {
            let v: f64 = rng8.sample(sd);
            black_box(v)
        })
    });
    g.finish();
}

// Each sampler branch (Small / One / Large) is a distinct code path, so bench
// them separately. scale = 1 (statrs rate = 1).
fn sample_branch(c: &mut Criterion, name: &str, shape: f64) {
    let jd = Gamma::shape_scale(shape, 1.0).unwrap();
    let sd = StatrsGamma::new(shape, 1.0).unwrap();
    let rd = RandGamma::new(shape, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group(name);
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(Sampleable::sample(&jd, &mut rng)))
    });
    g.bench_function("rand_distr", |b| b.iter(|| black_box(rng.sample(rd))));
    g.bench_function("statrs", |b| {
        b.iter(|| {
            let v: f64 = rng8.sample(sd);
            black_box(v)
        })
    });
    g.finish();
}

fn sample_branches(c: &mut Criterion) {
    sample_branch(c, "gamma/sample_shape_one", 1.0); // One (exponential) path
    sample_branch(c, "gamma/sample_small_shape", 0.5); // Small (boost) path
    sample_branch(c, "gamma/sample_large_shape", 100.0); // very large shape
}

// Algorithm-specific cdf/inverse_cdf branches: the Temme uniform-asymptotic cdf
// (large shape) and the extreme-quantile / large-shape inverse_cdf init regions.
fn tail_cases(c: &mut Criterion) {
    let jd = Gamma::shape_scale(1e6, 1.0).unwrap();
    let sd = StatrsGamma::new(1e6, 1.0).unwrap();
    let mut g = c.benchmark_group("gamma/cdf_large_shape");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(1e6)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(1e6)))));
    g.finish();

    let jd = Gamma::shape_scale(5.0, 1.0).unwrap();
    let sd = StatrsGamma::new(5.0, 1.0).unwrap();
    for (name, p) in [
        ("gamma/inverse_cdf_extreme_low", 0.001),
        ("gamma/inverse_cdf_extreme_high", 0.999),
    ] {
        let mut g = c.benchmark_group(name);
        g.bench_function("jolie", |b| {
            b.iter(|| black_box(jd.inverse_cdf(black_box(p))))
        });
        g.bench_function("statrs", |b| {
            b.iter(|| black_box(sd.inverse_cdf(black_box(p))))
        });
        g.finish();
    }

    let jd = Gamma::shape_scale(1000.0, 1.0).unwrap();
    let sd = StatrsGamma::new(1000.0, 1.0).unwrap();
    let mut g = c.benchmark_group("gamma/inverse_cdf_large_shape");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(0.5))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(0.5))))
    });
    g.finish();
}

fn density(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    // jolie's pdf/log_pdf take `&F`. black_box the *value* then pass its
    // reference (models a real call site); `black_box(&X)` would instead make
    // the pointer opaque and force a spurious load that dwarfs the op itself.
    let mut g = c.benchmark_group("gamma/pdf");
    g.bench_function("jolie", |b| {
        b.iter(|| {
            let x = black_box(X);
            black_box(jd.pdf(&x))
        })
    });
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.pdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("gamma/log_pdf");
    g.bench_function("jolie", |b| {
        b.iter(|| {
            let x = black_box(X);
            black_box(jd.log_pdf(&x))
        })
    });
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.ln_pdf(black_box(X)))));
    g.finish();
}

fn cumulative(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("gamma/cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("gamma/ccdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.ccdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.sf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("gamma/inverse_cdf");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(P))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(P))))
    });
    g.finish();

    // No statrs/rand_distr equivalent for log_cdf.
    let mut g = c.benchmark_group("gamma/log_cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_cdf(black_box(X)))));
    g.finish();
}

fn moments(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("gamma/mean");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mean())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mean())));
    g.finish();

    let mut g = c.benchmark_group("gamma/variance");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.variance())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.variance())));
    g.finish();

    let mut g = c.benchmark_group("gamma/entropy");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.entropy())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.entropy())));
    g.finish();

    let mut g = c.benchmark_group("gamma/skewness");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.skewness())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.skewness())));
    g.finish();

    let mut g = c.benchmark_group("gamma/mode");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mode())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mode())));
    g.finish();

    // No statrs equivalent for kurtosis.
    let mut g = c.benchmark_group("gamma/kurtosis");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.kurtosis())));
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(50);
    targets = sample, sample_branches, density, cumulative, tail_cases, moments
}
criterion_main!(benches);
