//! Type-erased "any distribution" enums, split by kind. Each pairs a runtime
//! distribution enum with a serializable parameter enum, so a config/JSON file
//! can select among the supported distributions.
//!
//! The parameter enums are internally tagged by `"type"`, e.g.
//! `{"type": "Uniform", "a": 0.0, "b": 1.0}`. Adding a distribution means adding
//! one variant and the corresponding match arms below.

use num_traits::Float;
use rand::Rng;

use crate::distributions::traits::*;
use crate::error::DistributionError;

use super::{
    DiscreteUniform, DiscreteUniformParams, Exponential, ExponentialParams, Gamma, GammaParams,
    LogNormal, LogNormalParams, NegativeBinomial, NegativeBinomialParams, Normal, NormalParams,
    Poisson, PoissonParams, Uniform, UniformParams, Weibull, WeibullParams,
};

// ============================== Continuous ==============================

/// Any supported continuous distribution behind a single enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyContinuous<F: Float> {
    Uniform(Uniform<F>),
    Exponential(Exponential<F>),
    Normal(Normal<F>),
    LogNormal(LogNormal<F>),
    Gamma(Gamma<F>),
    Weibull(Weibull<F>),
}

/// Serializable parameters for [`AnyContinuous`], internally tagged by `"type"`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum AnyContinuousParams<F> {
    Uniform(UniformParams<F>),
    Exponential(ExponentialParams<F>),
    Normal(NormalParams<F>),
    LogNormal(LogNormalParams<F>),
    Gamma(GammaParams<F>),
    Weibull(WeibullParams<F>),
}

impl<F: Float> Sampleable for AnyContinuous<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        match self {
            Self::Uniform(d) => d.sample(rng),
            Self::Exponential(d) => d.sample(rng),
            Self::Normal(d) => d.sample(rng),
            Self::LogNormal(d) => d.sample(rng),
            Self::Gamma(d) => d.sample(rng),
            Self::Weibull(d) => d.sample(rng),
        }
    }
}

impl<F: Float> Distribution<F> for AnyContinuous<F> {
    fn log_pdf(&self, x: &F) -> F {
        match self {
            Self::Uniform(d) => d.log_pdf(x),
            Self::Exponential(d) => d.log_pdf(x),
            Self::Normal(d) => d.log_pdf(x),
            Self::LogNormal(d) => d.log_pdf(x),
            Self::Gamma(d) => d.log_pdf(x),
            Self::Weibull(d) => d.log_pdf(x),
        }
    }

    fn pdf(&self, x: &F) -> F {
        match self {
            Self::Uniform(d) => d.pdf(x),
            Self::Exponential(d) => d.pdf(x),
            Self::Normal(d) => d.pdf(x),
            Self::LogNormal(d) => d.pdf(x),
            Self::Gamma(d) => d.pdf(x),
            Self::Weibull(d) => d.pdf(x),
        }
    }
}

impl<F: Float> UnivariateContinuous<F> for AnyContinuous<F> {
    type Params = AnyContinuousParams<F>;

    fn cdf(&self, x: F) -> F {
        match self {
            Self::Uniform(d) => d.cdf(x),
            Self::Exponential(d) => d.cdf(x),
            Self::Normal(d) => d.cdf(x),
            Self::LogNormal(d) => d.cdf(x),
            Self::Gamma(d) => d.cdf(x),
            Self::Weibull(d) => d.cdf(x),
        }
    }

    fn inverse_cdf(&self, p: F) -> F {
        match self {
            Self::Uniform(d) => d.inverse_cdf(p),
            Self::Exponential(d) => d.inverse_cdf(p),
            Self::Normal(d) => d.inverse_cdf(p),
            Self::LogNormal(d) => d.inverse_cdf(p),
            Self::Gamma(d) => d.inverse_cdf(p),
            Self::Weibull(d) => d.inverse_cdf(p),
        }
    }

    fn ccdf(&self, x: F) -> F {
        match self {
            Self::Uniform(d) => d.ccdf(x),
            Self::Exponential(d) => d.ccdf(x),
            Self::Normal(d) => d.ccdf(x),
            Self::LogNormal(d) => d.ccdf(x),
            Self::Gamma(d) => d.ccdf(x),
            Self::Weibull(d) => d.ccdf(x),
        }
    }

    fn support(&self) -> (F, F) {
        match self {
            Self::Uniform(d) => d.support(),
            Self::Exponential(d) => d.support(),
            Self::Normal(d) => d.support(),
            Self::LogNormal(d) => d.support(),
            Self::Gamma(d) => d.support(),
            Self::Weibull(d) => d.support(),
        }
    }

    fn params(&self) -> AnyContinuousParams<F> {
        match self {
            Self::Uniform(d) => AnyContinuousParams::Uniform(d.params()),
            Self::Exponential(d) => AnyContinuousParams::Exponential(d.params()),
            Self::Normal(d) => AnyContinuousParams::Normal(d.params()),
            Self::LogNormal(d) => AnyContinuousParams::LogNormal(d.params()),
            Self::Gamma(d) => AnyContinuousParams::Gamma(d.params()),
            Self::Weibull(d) => AnyContinuousParams::Weibull(d.params()),
        }
    }

    fn from_params(params: AnyContinuousParams<F>) -> Result<Self, DistributionError> {
        Ok(match params {
            AnyContinuousParams::Uniform(p) => Self::Uniform(Uniform::from_params(p)?),
            AnyContinuousParams::Exponential(p) => Self::Exponential(Exponential::from_params(p)?),
            AnyContinuousParams::Normal(p) => Self::Normal(Normal::from_params(p)?),
            AnyContinuousParams::LogNormal(p) => Self::LogNormal(LogNormal::from_params(p)?),
            AnyContinuousParams::Gamma(p) => Self::Gamma(Gamma::from_params(p)?),
            AnyContinuousParams::Weibull(p) => Self::Weibull(Weibull::from_params(p)?),
        })
    }
}

impl<F: Float> From<Uniform<F>> for AnyContinuous<F> {
    fn from(d: Uniform<F>) -> Self {
        Self::Uniform(d)
    }
}

impl<F: Float> From<Exponential<F>> for AnyContinuous<F> {
    fn from(d: Exponential<F>) -> Self {
        Self::Exponential(d)
    }
}

impl<F: Float> From<Normal<F>> for AnyContinuous<F> {
    fn from(d: Normal<F>) -> Self {
        Self::Normal(d)
    }
}

impl<F: Float> From<LogNormal<F>> for AnyContinuous<F> {
    fn from(d: LogNormal<F>) -> Self {
        Self::LogNormal(d)
    }
}

impl<F: Float> From<Gamma<F>> for AnyContinuous<F> {
    fn from(d: Gamma<F>) -> Self {
        Self::Gamma(d)
    }
}

impl<F: Float> From<Weibull<F>> for AnyContinuous<F> {
    fn from(d: Weibull<F>) -> Self {
        Self::Weibull(d)
    }
}

#[cfg(feature = "serde")]
impl<F: Float + serde::de::DeserializeOwned> AnyContinuous<F> {
    /// Parse a tagged JSON object (e.g. `{"type":"Uniform","a":0,"b":1}`) and
    /// construct the validated distribution.
    pub fn from_json_str(s: &str) -> Result<Self, DistributionError> {
        let params: AnyContinuousParams<F> = serde_json::from_str(s)
            .map_err(|_| DistributionError::InvalidParameter("invalid JSON"))?;
        Self::from_params(params)
    }
}

#[cfg(feature = "serde")]
impl<F: Float + serde::Serialize> AnyContinuous<F> {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(&self.params()).expect("params serialization cannot fail")
    }
}

// =============================== Discrete ===============================

/// Any supported discrete distribution behind a single enum.
///
/// Values are `i64`. Count distributions like `Poisson` are u64-native and
/// exposed through this i64 view; a negative argument is treated as out of
/// support. The native typed distribution keeps its u64 API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyDiscrete<F: Float> {
    DiscreteUniform(DiscreteUniform<F>),
    Poisson(Poisson<F>),
    NegativeBinomial(NegativeBinomial<F>),
}

/// Serializable parameters for [`AnyDiscrete`], internally tagged by `"type"`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum AnyDiscreteParams<F> {
    DiscreteUniform(DiscreteUniformParams),
    Poisson(PoissonParams<F>),
    NegativeBinomial(NegativeBinomialParams<F>),
}

impl<F: Float> Sampleable for AnyDiscrete<F> {
    type Value = i64;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        match self {
            Self::DiscreteUniform(d) => d.sample(rng),
            Self::Poisson(d) => d.sample(rng) as i64,
            Self::NegativeBinomial(d) => d.sample(rng) as i64,
        }
    }
}

impl<F: Float> Distribution<F> for AnyDiscrete<F> {
    fn log_pdf(&self, x: &i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.log_pdf(x),
            Self::Poisson(d) => {
                if *x < 0 {
                    F::neg_infinity()
                } else {
                    d.log_pdf(&(*x as u64))
                }
            }
            Self::NegativeBinomial(d) => {
                if *x < 0 {
                    F::neg_infinity()
                } else {
                    d.log_pdf(&(*x as u64))
                }
            }
        }
    }

    fn pdf(&self, x: &i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.pdf(x),
            Self::Poisson(d) => {
                if *x < 0 {
                    F::zero()
                } else {
                    d.pdf(&(*x as u64))
                }
            }
            Self::NegativeBinomial(d) => {
                if *x < 0 {
                    F::zero()
                } else {
                    d.pdf(&(*x as u64))
                }
            }
        }
    }
}

impl<F: Float> UnivariateDiscrete<F, i64> for AnyDiscrete<F> {
    type Params = AnyDiscreteParams<F>;

    fn cdf(&self, x: i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.cdf(x),
            Self::Poisson(d) => {
                if x < 0 {
                    F::zero()
                } else {
                    d.cdf(x as u64)
                }
            }
            Self::NegativeBinomial(d) => {
                if x < 0 {
                    F::zero()
                } else {
                    d.cdf(x as u64)
                }
            }
        }
    }

    fn ccdf(&self, x: i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.ccdf(x),
            Self::Poisson(d) => {
                if x < 0 {
                    F::one()
                } else {
                    d.ccdf(x as u64)
                }
            }
            Self::NegativeBinomial(d) => {
                if x < 0 {
                    F::one()
                } else {
                    d.ccdf(x as u64)
                }
            }
        }
    }

    fn inverse_cdf(&self, p: F) -> i64 {
        match self {
            Self::DiscreteUniform(d) => d.inverse_cdf(p),
            // Counts fit i64; clamp the q=1 sentinel (u64::MAX).
            Self::Poisson(d) => d.inverse_cdf(p).min(i64::MAX as u64) as i64,
            Self::NegativeBinomial(d) => d.inverse_cdf(p).min(i64::MAX as u64) as i64,
        }
    }

    fn support(&self) -> (i64, i64) {
        match self {
            Self::DiscreteUniform(d) => d.support(),
            Self::Poisson(d) => {
                let (lo, hi) = d.support();
                (lo as i64, hi.min(i64::MAX as u64) as i64)
            }
            Self::NegativeBinomial(d) => {
                let (lo, hi) = d.support();
                (lo as i64, hi.min(i64::MAX as u64) as i64)
            }
        }
    }

    fn params(&self) -> AnyDiscreteParams<F> {
        match self {
            Self::DiscreteUniform(d) => AnyDiscreteParams::DiscreteUniform(d.params()),
            Self::Poisson(d) => AnyDiscreteParams::Poisson(d.params()),
            Self::NegativeBinomial(d) => AnyDiscreteParams::NegativeBinomial(d.params()),
        }
    }

    fn from_params(params: AnyDiscreteParams<F>) -> Result<Self, DistributionError> {
        Ok(match params {
            AnyDiscreteParams::DiscreteUniform(p) => {
                Self::DiscreteUniform(DiscreteUniform::from_params(p)?)
            }
            AnyDiscreteParams::Poisson(p) => Self::Poisson(Poisson::from_params(p)?),
            AnyDiscreteParams::NegativeBinomial(p) => {
                Self::NegativeBinomial(NegativeBinomial::from_params(p)?)
            }
        })
    }
}

impl<F: Float> From<DiscreteUniform<F>> for AnyDiscrete<F> {
    fn from(d: DiscreteUniform<F>) -> Self {
        Self::DiscreteUniform(d)
    }
}

impl<F: Float> From<Poisson<F>> for AnyDiscrete<F> {
    fn from(d: Poisson<F>) -> Self {
        Self::Poisson(d)
    }
}

impl<F: Float> From<NegativeBinomial<F>> for AnyDiscrete<F> {
    fn from(d: NegativeBinomial<F>) -> Self {
        Self::NegativeBinomial(d)
    }
}

#[cfg(feature = "serde")]
impl<F: Float + serde::de::DeserializeOwned> AnyDiscrete<F> {
    /// Parse a tagged JSON object (e.g. `{"type":"DiscreteUniform","a":0,"b":9}`)
    /// and construct the validated distribution.
    pub fn from_json_str(s: &str) -> Result<Self, DistributionError> {
        let params: AnyDiscreteParams<F> = serde_json::from_str(s)
            .map_err(|_| DistributionError::InvalidParameter("invalid JSON"))?;
        Self::from_params(params)
    }
}

#[cfg(feature = "serde")]
impl<F: Float + serde::Serialize> AnyDiscrete<F> {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(&self.params()).expect("params serialization cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_from_params_and_delegation() {
        let p = AnyContinuousParams::Uniform(UniformParams { a: 0.0, b: 2.0 });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0.0, 2.0));
        assert!((d.pdf(&1.0) - 0.5).abs() < 1e-12);
        assert!((d.cdf(1.0) - 0.5).abs() < 1e-12);
        assert!((d.inverse_cdf(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_from_params_validates() {
        let p = AnyContinuousParams::Uniform(UniformParams { a: 2.0, b: 1.0 });
        assert!(AnyContinuous::<f64>::from_params(p).is_err());
    }

    #[test]
    fn continuous_exponential_from_params_and_delegation() {
        let p = AnyContinuousParams::Exponential(ExponentialParams::Scale { scale: 2.0 });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0.0, f64::INFINITY));
        assert!((d.pdf(&0.0) - 0.5).abs() < 1e-12);
        assert!((d.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        assert!((d.ccdf(2.0) - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn continuous_exponential_from_rate_params() {
        let p = AnyContinuousParams::Exponential(ExponentialParams::Rate { rate: 4.0 });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        // rate 4 → scale 0.25 → pdf(0) = rate = 4
        assert!((d.pdf(&0.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_normal_from_params_and_delegation() {
        let p = AnyContinuousParams::Normal(NormalParams {
            mean: 0.0,
            std_dev: 1.0,
        });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (f64::NEG_INFINITY, f64::INFINITY));
        assert!((d.cdf(0.0) - 0.5).abs() < 1e-12);
        assert!((d.inverse_cdf(0.5)).abs() < 1e-12);
    }

    #[test]
    fn continuous_lognormal_from_params_and_delegation() {
        let p = AnyContinuousParams::LogNormal(LogNormalParams {
            mu: 0.0,
            sigma: 1.0,
        });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0.0, f64::INFINITY));
        // median = exp(mu) = 1
        assert!((d.cdf(1.0) - 0.5).abs() < 1e-12);
        assert!((d.inverse_cdf(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_gamma_from_params_and_delegation() {
        // Gamma(1, scale) is Exponential(scale): cdf(x) = 1 - e^(-x/scale).
        let p = AnyContinuousParams::Gamma(GammaParams::ShapeScale {
            shape: 1.0,
            scale: 2.0,
        });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0.0, f64::INFINITY));
        assert!((d.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        assert!((d.ccdf(2.0) - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn continuous_gamma_from_rate_params() {
        let p = AnyContinuousParams::Gamma(GammaParams::ShapeRate {
            shape: 2.0,
            rate: 0.5,
        });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        // rate 0.5 -> scale 2 -> mean = shape*scale = 4
        match d {
            AnyContinuous::Gamma(g) => assert!((g.scale() - 2.0).abs() < 1e-12),
            _ => panic!("expected Gamma"),
        }
    }

    #[test]
    fn continuous_weibull_from_params_and_delegation() {
        // Weibull(1, scale) is Exponential(scale): cdf(x) = 1 - e^(-x/scale).
        let p = AnyContinuousParams::Weibull(WeibullParams {
            shape: 1.0,
            scale: 2.0,
        });
        let d = AnyContinuous::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0.0, f64::INFINITY));
        assert!((d.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        assert!((d.ccdf(2.0) - (-1.0_f64).exp()).abs() < 1e-12);
        assert!((d.inverse_cdf(0.5) - 2.0 * 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn discrete_from_params_and_delegation() {
        let p = AnyDiscreteParams::DiscreteUniform(DiscreteUniformParams { a: 0, b: 9 });
        let d = AnyDiscrete::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0, 9));
        assert!((d.pdf(&3) - 0.1).abs() < 1e-12);
        assert!((d.cdf(4) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn discrete_poisson_from_params_and_delegation() {
        // Poisson is u64-native; AnyDiscrete exposes it through the i64 view.
        let p = AnyDiscreteParams::Poisson(PoissonParams { lambda: 4.0 });
        let d = AnyDiscrete::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0, i64::MAX));
        // pmf(2) for Poisson(4) = e^-4 * 4^2 / 2! = 8 e^-4
        assert!((d.pdf(&2) - 8.0 * (-4.0_f64).exp()).abs() < 1e-12);
        // negative argument is out of support
        assert_eq!(d.pdf(&-1), 0.0);
        assert_eq!(d.cdf(-1), 0.0);
        assert_eq!(d.log_pdf(&-3), f64::NEG_INFINITY);
    }

    #[test]
    fn discrete_poisson_via_from_and_sample() {
        // Mirrors abcsmc's `discrete(Poisson::new(..))` path: From + sample + pdf.
        let d: AnyDiscrete<f64> = Poisson::new(5.0).unwrap().into();
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for _ in 0..1000 {
            let k = d.sample(&mut rng);
            assert!(k >= 0);
            assert!(d.pdf(&k) > 0.0);
        }
    }

    #[test]
    fn discrete_negative_binomial_from_params_and_delegation() {
        let p = AnyDiscreteParams::NegativeBinomial(NegativeBinomialParams::RP { r: 5.0, p: 0.5 });
        let d = AnyDiscrete::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0, i64::MAX));
        // pmf(0) for NB(5, 0.5) = p^r = 0.5^5 = 0.03125
        assert!((d.pdf(&0) - 0.03125).abs() < 1e-12);
        assert_eq!(d.pdf(&-1), 0.0);
        assert_eq!(d.cdf(-1), 0.0);
    }

    #[test]
    fn discrete_negative_binomial_via_from_and_sample() {
        let d: AnyDiscrete<f64> = NegativeBinomial::new(5.0, 0.5).unwrap().into();
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for _ in 0..1000 {
            let k = d.sample(&mut rng);
            assert!(k >= 0);
            assert!(d.pdf(&k) > 0.0);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_json_round_trip() {
        let d = AnyContinuous::from(Uniform::<f64>::new(0.0, 1.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Uniform","a":0.0,"b":1.0}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_exponential_json_round_trip() {
        let d = AnyContinuous::from(Exponential::<f64>::from_scale(2.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Exponential","scale":2.0}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_exponential_json_from_rate() {
        let json = r#"{"type":"Exponential","rate":0.5}"#;
        let d = AnyContinuous::<f64>::from_json_str(json).unwrap();
        // rate 0.5 → scale 2.0
        match d {
            AnyContinuous::Exponential(e) => assert!((e.scale() - 2.0).abs() < 1e-12),
            _ => panic!("expected Exponential"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_normal_json_round_trip() {
        let d = AnyContinuous::from(Normal::<f64>::new(1.5, 2.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Normal","mean":1.5,"std_dev":2.0}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_lognormal_json_round_trip() {
        let d = AnyContinuous::from(LogNormal::<f64>::new(1.5, 2.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"LogNormal","mu":1.5,"sigma":2.0}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_gamma_json_round_trip() {
        let d = AnyContinuous::from(Gamma::<f64>::shape_scale(2.0, 1.5).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Gamma","shape":2.0,"scale":1.5}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_gamma_json_from_rate() {
        let json = r#"{"type":"Gamma","shape":2.0,"rate":0.5}"#;
        let d = AnyContinuous::<f64>::from_json_str(json).unwrap();
        match d {
            AnyContinuous::Gamma(g) => assert!((g.scale() - 2.0).abs() < 1e-12),
            _ => panic!("expected Gamma"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_weibull_json_round_trip() {
        let d = AnyContinuous::from(Weibull::<f64>::new(1.5, 2.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Weibull","shape":1.5,"scale":2.0}"#);
        let d2 = AnyContinuous::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn discrete_json_round_trip() {
        let d = AnyDiscrete::from(DiscreteUniform::<f64>::new(0, 9).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"DiscreteUniform","a":0,"b":9}"#);
        let d2 = AnyDiscrete::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn discrete_poisson_json_round_trip() {
        let d = AnyDiscrete::from(Poisson::<f64>::new(4.0).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"Poisson","lambda":4.0}"#);
        let d2 = AnyDiscrete::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn discrete_negative_binomial_json_round_trip() {
        let d = AnyDiscrete::from(NegativeBinomial::<f64>::new(5.0, 0.5).unwrap());
        let s = d.to_json_string();
        let d2 = AnyDiscrete::<f64>::from_json_str(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn continuous_json_from_config() {
        let json = r#"{"type":"Uniform","a":-1.0,"b":1.0}"#;
        let d = AnyContinuous::<f64>::from_json_str(json).unwrap();
        assert_eq!(d.support(), (-1.0, 1.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_rejects_unknown_type() {
        let json = r#"{"type":"Cauchy","x0":0.0,"gamma":1.0}"#;
        assert!(AnyContinuous::<f64>::from_json_str(json).is_err());
    }
}
