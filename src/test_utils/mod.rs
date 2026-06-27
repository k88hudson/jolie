// Based on Julia's Distributions.jl test infrastructure:
// https://github.com/JuliaStats/Distributions.jl/blob/master/test/testutils.jl

// Many helpers target distributions not yet ported to jolie (discrete, etc.),
// so they are unused until those distributions land.
#![allow(dead_code)]

mod consistency;
mod edge_cases;
mod evaluation;
mod reference_data;
mod sampling;

pub use consistency::*;
pub use edge_cases::*;
pub use evaluation::*;
pub use reference_data::*;
pub use sampling::*;
