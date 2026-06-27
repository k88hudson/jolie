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

use super::{DiscreteUniform, DiscreteUniformParams, Uniform, UniformParams};

// ============================== Continuous ==============================

/// Any supported continuous distribution behind a single enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyContinuous<F: Float> {
    Uniform(Uniform<F>),
}

/// Serializable parameters for [`AnyContinuous`], internally tagged by `"type"`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum AnyContinuousParams<F> {
    Uniform(UniformParams<F>),
}

impl<F: Float> Sampleable for AnyContinuous<F> {
    type Value = F;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        match self {
            Self::Uniform(d) => d.sample(rng),
        }
    }
}

impl<F: Float> Distribution<F> for AnyContinuous<F> {
    fn log_pdf(&self, x: &F) -> F {
        match self {
            Self::Uniform(d) => d.log_pdf(x),
        }
    }

    fn pdf(&self, x: &F) -> F {
        match self {
            Self::Uniform(d) => d.pdf(x),
        }
    }
}

impl<F: Float> UnivariateContinuous<F> for AnyContinuous<F> {
    type Params = AnyContinuousParams<F>;

    fn cdf(&self, x: F) -> F {
        match self {
            Self::Uniform(d) => d.cdf(x),
        }
    }

    fn inverse_cdf(&self, p: F) -> F {
        match self {
            Self::Uniform(d) => d.inverse_cdf(p),
        }
    }

    fn support(&self) -> (F, F) {
        match self {
            Self::Uniform(d) => d.support(),
        }
    }

    fn params(&self) -> AnyContinuousParams<F> {
        match self {
            Self::Uniform(d) => AnyContinuousParams::Uniform(d.params()),
        }
    }

    fn from_params(params: AnyContinuousParams<F>) -> Result<Self, DistributionError> {
        Ok(match params {
            AnyContinuousParams::Uniform(p) => Self::Uniform(Uniform::from_params(p)?),
        })
    }
}

impl<F: Float> From<Uniform<F>> for AnyContinuous<F> {
    fn from(d: Uniform<F>) -> Self {
        Self::Uniform(d)
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
/// Values are `i64`; all current discrete distributions share that value type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyDiscrete<F: Float> {
    DiscreteUniform(DiscreteUniform<F>),
}

/// Serializable parameters for [`AnyDiscrete`], internally tagged by `"type"`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum AnyDiscreteParams {
    DiscreteUniform(DiscreteUniformParams),
}

impl<F: Float> Sampleable for AnyDiscrete<F> {
    type Value = i64;

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        match self {
            Self::DiscreteUniform(d) => d.sample(rng),
        }
    }
}

impl<F: Float> Distribution<F> for AnyDiscrete<F> {
    fn log_pdf(&self, x: &i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.log_pdf(x),
        }
    }

    fn pdf(&self, x: &i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.pdf(x),
        }
    }
}

impl<F: Float> UnivariateDiscrete<F, i64> for AnyDiscrete<F> {
    type Params = AnyDiscreteParams;

    fn cdf(&self, x: i64) -> F {
        match self {
            Self::DiscreteUniform(d) => d.cdf(x),
        }
    }

    fn inverse_cdf(&self, p: F) -> i64 {
        match self {
            Self::DiscreteUniform(d) => d.inverse_cdf(p),
        }
    }

    fn support(&self) -> (i64, i64) {
        match self {
            Self::DiscreteUniform(d) => d.support(),
        }
    }

    fn params(&self) -> AnyDiscreteParams {
        match self {
            Self::DiscreteUniform(d) => AnyDiscreteParams::DiscreteUniform(d.params()),
        }
    }

    fn from_params(params: AnyDiscreteParams) -> Result<Self, DistributionError> {
        Ok(match params {
            AnyDiscreteParams::DiscreteUniform(p) => {
                Self::DiscreteUniform(DiscreteUniform::from_params(p)?)
            }
        })
    }
}

impl<F: Float> From<DiscreteUniform<F>> for AnyDiscrete<F> {
    fn from(d: DiscreteUniform<F>) -> Self {
        Self::DiscreteUniform(d)
    }
}

#[cfg(feature = "serde")]
impl<F: Float> AnyDiscrete<F> {
    /// Parse a tagged JSON object (e.g. `{"type":"DiscreteUniform","a":0,"b":9}`)
    /// and construct the validated distribution.
    pub fn from_json_str(s: &str) -> Result<Self, DistributionError> {
        let params: AnyDiscreteParams = serde_json::from_str(s)
            .map_err(|_| DistributionError::InvalidParameter("invalid JSON"))?;
        Self::from_params(params)
    }

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
    fn discrete_from_params_and_delegation() {
        let p = AnyDiscreteParams::DiscreteUniform(DiscreteUniformParams { a: 0, b: 9 });
        let d = AnyDiscrete::<f64>::from_params(p).unwrap();
        assert_eq!(d.support(), (0, 9));
        assert!((d.pdf(&3) - 0.1).abs() < 1e-12);
        assert!((d.cdf(4) - 0.5).abs() < 1e-12);
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
    fn discrete_json_round_trip() {
        let d = AnyDiscrete::from(DiscreteUniform::<f64>::new(0, 9).unwrap());
        let s = d.to_json_string();
        assert_eq!(s, r#"{"type":"DiscreteUniform","a":0,"b":9}"#);
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
