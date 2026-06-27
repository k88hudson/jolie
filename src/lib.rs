// Each distribution lives in a folder whose impl file shares the folder name
// (e.g. `uniform/uniform.rs`); allow that deliberate layout crate-wide.
#![allow(clippy::module_inception)]

pub mod distributions;
pub mod error;
pub(crate) mod special;
pub mod unchecked;

#[cfg(test)]
pub(crate) mod test_utils;
