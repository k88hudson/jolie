//! Benchmarks for `Uniform`, comparing each trait method against `rand_distr`
//! (sampling) and `statrs` (everything else) where an equivalent exists.
//! Methods with no equivalent (`log_cdf`, `kurtosis`) are benched on their own.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use jolie::distributions::*;

use rand::rngs::StdRng;
use rand::{RngExt as _, SeedableRng};
use rand_distr::Uniform as RandUniform;

// statrs 0.18 samples through rand 0.8 (aliased `rand08`).
use rand08::rngs::StdRng as StdRng08;
use rand08::{Rng as _, SeedableRng as _};

use statrs::distribution::{Continuous, ContinuousCDF, Uniform as StatrsUniform};
use statrs::statistics::{Distribution as StatrsMoments, Mode};

const A: f64 = -2.0;
const B: f64 = 5.0;
const X: f64 = 1.5; // an in-support evaluation point
const P: f64 = 0.65; // a quantile probability

fn jolie_dist() -> Uniform<f64> {
    Uniform::new(A, B).unwrap()
}

fn statrs_dist() -> StatrsUniform {
    StatrsUniform::new(A, B).unwrap()
}

fn sample(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();
    let rd = RandUniform::new(A, B).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group("uniform/sample");
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

    let mut g = c.benchmark_group("uniform/pdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.pdf(black_box(&X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.pdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("uniform/log_pdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_pdf(black_box(&X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.ln_pdf(black_box(X)))));
    g.finish();
}

fn cumulative(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("uniform/cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("uniform/ccdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.ccdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.sf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("uniform/inverse_cdf");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(P))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(P))))
    });
    g.finish();

    // No statrs/rand_distr equivalent for log_cdf.
    let mut g = c.benchmark_group("uniform/log_cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_cdf(black_box(X)))));
    g.finish();
}

fn moments(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("uniform/mean");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mean())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mean())));
    g.finish();

    let mut g = c.benchmark_group("uniform/variance");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.variance())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.variance())));
    g.finish();

    let mut g = c.benchmark_group("uniform/entropy");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.entropy())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.entropy())));
    g.finish();

    let mut g = c.benchmark_group("uniform/skewness");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.skewness())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.skewness())));
    g.finish();

    let mut g = c.benchmark_group("uniform/mode");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mode())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mode())));
    g.finish();

    // No statrs equivalent for kurtosis.
    let mut g = c.benchmark_group("uniform/kurtosis");
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
