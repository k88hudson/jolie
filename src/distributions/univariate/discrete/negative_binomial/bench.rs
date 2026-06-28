//! Benchmarks for `NegativeBinomial`, comparing each method against `statrs`
//! where an equivalent exists. `rand_distr` has no NegativeBinomial, and statrs
//! exposes no entropy/kurtosis for it, so those are benched alone.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use jolie::distributions::*;

use rand::SeedableRng;
use rand::rngs::StdRng;

// statrs 0.18 samples through rand 0.8 (aliased `rand08`).
use rand08::rngs::StdRng as StdRng08;
use rand08::{Rng as _, SeedableRng as _};

use statrs::distribution::{Discrete, DiscreteCDF, NegativeBinomial as StatrsNB};
use statrs::statistics::{DiscreteDistribution, Mode};

const R: f64 = 10.0; // canonical shape for analytic methods
const P: f64 = 0.4;
const X: u64 = 12; // an in-support evaluation point
const Q: f64 = 0.65; // a quantile probability

fn jolie_dist() -> NegativeBinomial<f64> {
    NegativeBinomial::new(R, P).unwrap()
}

fn statrs_dist() -> StatrsNB {
    StatrsNB::new(R, P).unwrap()
}

fn sample(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group("negative_binomial/sample");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(Sampleable::sample(&jd, &mut rng)))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| {
            let v: u64 = rng8.sample(sd);
            black_box(v)
        })
    });
    g.finish();
}

fn density(c: &mut Criterion) {
    // pmf across (r, p, k) regimes (ln_gamma + ln_factorial table vs Stirling).
    for (name, r, p, x) in [
        ("negative_binomial/pmf_small", 5.0, 0.5, 3u64),
        ("negative_binomial/pmf_medium", 20.0, 0.3, 50u64),
        ("negative_binomial/pmf_large", 100.0, 0.1, 500u64),
    ] {
        let jd = NegativeBinomial::<f64>::new(r, p).unwrap();
        let sd = StatrsNB::new(r, p).unwrap();
        let mut g = c.benchmark_group(name);
        g.bench_function("jolie", |b| {
            b.iter(|| {
                let x = black_box(x);
                black_box(jd.pdf(&x))
            })
        });
        g.bench_function("statrs", |b| b.iter(|| black_box(sd.pmf(black_box(x)))));
        g.finish();
    }

    let jd = jolie_dist();
    let sd = statrs_dist();
    let mut g = c.benchmark_group("negative_binomial/log_pmf");
    g.bench_function("jolie", |b| {
        b.iter(|| {
            let x = black_box(X);
            black_box(jd.log_pdf(&x))
        })
    });
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.ln_pmf(black_box(X)))));
    g.finish();
}

fn cumulative(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("negative_binomial/cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/ccdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.ccdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.sf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/inverse_cdf");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(Q))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(Q))))
    });
    g.finish();

    // No statrs/rand_distr equivalent for log_cdf.
    let mut g = c.benchmark_group("negative_binomial/log_cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_cdf(black_box(X)))));
    g.finish();
}

fn moments(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("negative_binomial/mean");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mean())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mean())));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/variance");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.variance())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.variance())));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/skewness");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.skewness())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.skewness())));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/mode");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mode())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mode())));
    g.finish();

    // No statrs equivalent for entropy (series sum) or kurtosis.
    let mut g = c.benchmark_group("negative_binomial/entropy");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.entropy())));
    g.finish();

    let mut g = c.benchmark_group("negative_binomial/kurtosis");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.kurtosis())));
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(50);
    targets = sample, density, cumulative, moments
}
criterion_main!(benches);
