//! Benchmarks for `Exponential`, comparing each trait method against
//! `rand_distr` (sampling) and `statrs` (everything else) where an equivalent
//! exists. Methods with no equivalent (`log_cdf`, `kurtosis`) are benched alone.
//!
//! jolie is parameterized by scale (θ); `rand_distr` and `statrs` use rate
//! (λ = 1/θ), so their constructors take `RATE`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use jolie::distributions::*;

use rand::rngs::StdRng;
use rand::{RngExt as _, SeedableRng};
use rand_distr::Exp as RandExp;

// statrs 0.18 samples through rand 0.8 (aliased `rand08`).
use rand08::rngs::StdRng as StdRng08;
use rand08::{Rng as _, SeedableRng as _};

use statrs::distribution::{Continuous, ContinuousCDF, Exp as StatrsExp};
use statrs::statistics::{Distribution as StatrsMoments, Mode};

const SCALE: f64 = 2.0;
const RATE: f64 = 0.5; // 1 / SCALE
const X: f64 = 1.5; // an in-support evaluation point
const P: f64 = 0.65; // a quantile probability

fn jolie_dist() -> Exponential<f64> {
    Exponential::from_scale(SCALE).unwrap()
}

fn statrs_dist() -> StatrsExp {
    StatrsExp::new(RATE).unwrap()
}

fn sample(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();
    let rd = RandExp::new(RATE).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group("exponential/sample");
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

fn density(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    // jolie's pdf/log_pdf take `&F`. black_box the *value* then pass its
    // reference (models a real call site); `black_box(&X)` would instead make
    // the pointer opaque and force a spurious load that dwarfs the op itself.
    let mut g = c.benchmark_group("exponential/pdf");
    g.bench_function("jolie", |b| {
        b.iter(|| {
            let x = black_box(X);
            black_box(jd.pdf(&x))
        })
    });
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.pdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("exponential/log_pdf");
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

    let mut g = c.benchmark_group("exponential/cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("exponential/ccdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.ccdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.sf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("exponential/inverse_cdf");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(P))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(P))))
    });
    g.finish();

    // No statrs/rand_distr equivalent for log_cdf.
    let mut g = c.benchmark_group("exponential/log_cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_cdf(black_box(X)))));
    g.finish();
}

fn moments(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("exponential/mean");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mean())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mean())));
    g.finish();

    let mut g = c.benchmark_group("exponential/variance");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.variance())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.variance())));
    g.finish();

    let mut g = c.benchmark_group("exponential/entropy");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.entropy())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.entropy())));
    g.finish();

    let mut g = c.benchmark_group("exponential/skewness");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.skewness())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.skewness())));
    g.finish();

    let mut g = c.benchmark_group("exponential/mode");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mode())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mode())));
    g.finish();

    // No statrs equivalent for kurtosis.
    let mut g = c.benchmark_group("exponential/kurtosis");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.kurtosis())));
    g.finish();
}

criterion_group! {
    name = benches;
    // These are sub-nanosecond ops; short windows still give stable estimates.
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(50);
    targets = sample, density, cumulative, moments
}
criterion_main!(benches);
