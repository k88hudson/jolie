//! Benchmarks for `Poisson`, comparing each trait method against `rand_distr`
//! (sampling) and `statrs` (everything else) where an equivalent exists.
//! Methods with no equivalent (`log_cdf`, `kurtosis`) are benched alone.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use jolie::distributions::*;

use rand::rngs::StdRng;
use rand::{RngExt as _, SeedableRng};
use rand_distr::Poisson as RandPoisson;

// statrs 0.18 samples through rand 0.8 (aliased `rand08`).
use rand08::rngs::StdRng as StdRng08;
use rand08::{Rng as _, SeedableRng as _};

use statrs::distribution::{Discrete, DiscreteCDF, Poisson as StatrsPoisson};
use statrs::statistics::{Distribution as StatrsMoments, Mode};

const LAMBDA: f64 = 10.0; // canonical (Ahrens-Dieter branch) for analytic methods
const X: u64 = 8; // an in-support evaluation point
const P: f64 = 0.65; // a quantile probability

fn jolie_dist() -> Poisson<f64> {
    Poisson::new(LAMBDA).unwrap()
}

fn statrs_dist() -> StatrsPoisson {
    StatrsPoisson::new(LAMBDA).unwrap()
}

// The two sampler paths (Knuth for λ < 10, Ahrens-Dieter rejection for λ >= 10)
// are distinct code, so bench them separately.
fn sample_branch(c: &mut Criterion, name: &str, lambda: f64) {
    let jd = Poisson::<f64>::new(lambda).unwrap();
    let sd = StatrsPoisson::new(lambda).unwrap();
    let rd = RandPoisson::new(lambda).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng8 = StdRng08::seed_from_u64(0xC0FFEE);

    let mut g = c.benchmark_group(name);
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(Sampleable::sample(&jd, &mut rng)))
    });
    g.bench_function("rand_distr", |b| b.iter(|| black_box(rng.sample(rd))));
    g.bench_function("statrs", |b| {
        b.iter(|| {
            let v: u64 = rng8.sample(sd);
            black_box(v)
        })
    });
    g.finish();
}

fn sample(c: &mut Criterion) {
    sample_branch(c, "poisson/sample_knuth", 5.0);
    sample_branch(c, "poisson/sample_rejection", 50.0);
}

fn density(c: &mut Criterion) {
    // pmf across λ regimes (table lookup vs Stirling in ln_factorial).
    for (name, lam, x) in [
        ("poisson/pmf_small", 5.0, 5u64),
        ("poisson/pmf_medium", 50.0, 50u64),
        ("poisson/pmf_large", 500.0, 500u64),
    ] {
        let jd = Poisson::<f64>::new(lam).unwrap();
        let sd = StatrsPoisson::new(lam).unwrap();
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
    let mut g = c.benchmark_group("poisson/log_pmf");
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

    let mut g = c.benchmark_group("poisson/cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.cdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.cdf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("poisson/ccdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.ccdf(black_box(X)))));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.sf(black_box(X)))));
    g.finish();

    let mut g = c.benchmark_group("poisson/inverse_cdf");
    g.bench_function("jolie", |b| {
        b.iter(|| black_box(jd.inverse_cdf(black_box(P))))
    });
    g.bench_function("statrs", |b| {
        b.iter(|| black_box(sd.inverse_cdf(black_box(P))))
    });
    g.finish();

    // No statrs/rand_distr equivalent for log_cdf.
    let mut g = c.benchmark_group("poisson/log_cdf");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.log_cdf(black_box(X)))));
    g.finish();
}

fn moments(c: &mut Criterion) {
    let jd = jolie_dist();
    let sd = statrs_dist();

    let mut g = c.benchmark_group("poisson/mean");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mean())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mean())));
    g.finish();

    let mut g = c.benchmark_group("poisson/variance");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.variance())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.variance())));
    g.finish();

    let mut g = c.benchmark_group("poisson/entropy");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.entropy())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.entropy())));
    g.finish();

    let mut g = c.benchmark_group("poisson/skewness");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.skewness())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.skewness())));
    g.finish();

    let mut g = c.benchmark_group("poisson/mode");
    g.bench_function("jolie", |b| b.iter(|| black_box(jd.mode())));
    g.bench_function("statrs", |b| b.iter(|| black_box(sd.mode())));
    g.finish();

    // No statrs equivalent for kurtosis.
    let mut g = c.benchmark_group("poisson/kurtosis");
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
