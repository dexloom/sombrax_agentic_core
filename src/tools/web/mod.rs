//! Web tools
//!
//! Tools for HTTP requests and web content retrieval.

mod env_expand;
mod fetch;

pub use env_expand::expand_env_vars;
pub use fetch::{FetchArgs, FetchOutput, FetchTool};
