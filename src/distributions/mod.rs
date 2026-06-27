pub mod any;
pub mod traits;
pub mod univariate;

pub use traits::*;

pub use any::{AnyContinuous, AnyContinuousParams, AnyDiscrete, AnyDiscreteParams};
pub use univariate::continuous::exponential::{Exponential, ExponentialParams};
pub use univariate::continuous::normal::{Normal, NormalParams};
pub use univariate::continuous::uniform::{Uniform, UniformParams};
pub use univariate::discrete::discrete_uniform::{DiscreteUniform, DiscreteUniformParams};
