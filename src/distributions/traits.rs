use std::fmt::{Debug, Display};

use num_traits::{Float, FromPrimitive, PrimInt, ToPrimitive};
use rand::Rng;

use crate::error::DistributionError;

/// Implement `rand::distr::Distribution` for a jolie distribution type,
/// delegating to [`Sampleable::sample`].
macro_rules! impl_rand_distribution {
    ($dist:ident<F: Float> => $value:ty) => {
        impl<F: num_traits::Float> rand::distr::Distribution<$value> for $dist<F> {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $value {
                crate::distributions::traits::Sampleable::sample(self, rng)
            }
        }
    };
}

pub(crate) use impl_rand_distribution;

pub trait Sampleable {
    type Value;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Value;

    fn sample_fill<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [Self::Value]) {
        for o in out.iter_mut() {
            *o = self.sample(rng);
        }
    }
}

pub trait Distribution<F: Float>: Sampleable {
    fn log_pdf(&self, x: &Self::Value) -> F;

    fn pdf(&self, x: &Self::Value) -> F {
        self.log_pdf(x).exp()
    }
}

pub trait UnivariateContinuous<F: Float>: Distribution<F, Value = F> {
    type Params;

    fn support(&self) -> (F, F);
    fn params(&self) -> Self::Params;
    fn from_params(params: Self::Params) -> Result<Self, DistributionError>
    where
        Self: Sized;

    fn cdf(&self, x: F) -> F;
    fn inverse_cdf(&self, p: F) -> F;

    fn ccdf(&self, x: F) -> F {
        F::one() - self.cdf(x)
    }
    fn log_cdf(&self, x: F) -> F {
        self.cdf(x).ln()
    }
}

/// Trait bound for integer types used as discrete distribution values.
///
/// Built on [`PrimInt`] which provides arithmetic, checked ops, conversions,
/// and bounds.
///
/// Implemented for `i64` and `u64`.
pub trait DiscreteInt: PrimInt + Debug + Display + FromPrimitive + ToPrimitive + 'static {
    /// Iterate over inclusive range [lo, hi]. Avoids the unstable `Step` trait.
    ///
    /// WARNING: Do not call with unbounded supports (e.g. `0..=u64::MAX`).
    /// Use bounded ranges only.
    fn for_each_in_range(lo: Self, hi: Self, f: impl FnMut(Self));
    /// Convert to usize for indexing (saturates negative i64 to 0).
    fn to_usize_saturating(self) -> usize;
    /// Number of integers in [lo, hi], as usize. Returns `usize::MAX` on overflow.
    fn range_size(lo: Self, hi: Self) -> usize;
}

impl DiscreteInt for i64 {
    fn for_each_in_range(lo: Self, hi: Self, mut f: impl FnMut(Self)) {
        for x in lo..=hi {
            f(x);
        }
    }
    fn to_usize_saturating(self) -> usize {
        self.max(0) as usize
    }
    fn range_size(lo: Self, hi: Self) -> usize {
        if hi < lo {
            0
        } else {
            // Use u64 arithmetic to avoid overflow when hi - lo + 1 > i64::MAX
            let diff = (hi as u64).wrapping_sub(lo as u64);
            match diff.checked_add(1) {
                Some(n) => n as usize,
                None => usize::MAX,
            }
        }
    }
}

impl DiscreteInt for u64 {
    fn for_each_in_range(lo: Self, hi: Self, mut f: impl FnMut(Self)) {
        for x in lo..=hi {
            f(x);
        }
    }
    fn to_usize_saturating(self) -> usize {
        self as usize
    }
    fn range_size(lo: Self, hi: Self) -> usize {
        if hi < lo {
            0
        } else {
            match (hi - lo).checked_add(1) {
                Some(n) => n as usize,
                None => usize::MAX,
            }
        }
    }
}

pub trait UnivariateDiscrete<F: Float, K: DiscreteInt>: Distribution<F, Value = K> {
    type Params;

    fn support(&self) -> (K, K);
    fn in_support(&self, x: K) -> bool {
        let (lo, hi) = self.support();
        x >= lo && x <= hi
    }
    fn params(&self) -> Self::Params;
    fn from_params(params: Self::Params) -> Result<Self, DistributionError>
    where
        Self: Sized;

    fn cdf(&self, x: K) -> F;
    fn ccdf(&self, x: K) -> F {
        F::one() - self.cdf(x)
    }
    fn log_cdf(&self, x: K) -> F {
        self.cdf(x).ln()
    }
    fn inverse_cdf(&self, p: F) -> K;
}

// Moment / summary-statistic traits. Each is optional per distribution and
// returns `Option` where a moment may not exist (e.g. heavy-tailed cases).

pub trait HasMean {
    type Value;
    fn mean(&self) -> Option<Self::Value>;
}

pub trait HasVariance: HasMean {
    fn variance(&self) -> Option<Self::Value>;
}

pub trait HasEntropy {
    type Value;
    fn entropy(&self) -> Option<Self::Value>;
}

pub trait HasMode {
    type Value;
    fn mode(&self) -> Option<Self::Value>;
}

pub trait HasSkewness {
    type Value;
    fn skewness(&self) -> Option<Self::Value>;
}

pub trait HasKurtosis {
    type Value;
    fn kurtosis(&self) -> Option<Self::Value>;
}
