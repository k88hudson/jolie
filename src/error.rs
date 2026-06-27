use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionError {
    InvalidParameter(&'static str),
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for DistributionError {}
