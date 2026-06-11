//! MiniMax provider module (Anthropic-compatible API).
//!
//! Provides access to MiniMax M2.x models via the Anthropic Messages API.

pub mod client;
pub mod types;

pub use client::{MinimaxClient, MinimaxClientBuilder, MinimaxCompletionModel};
pub use types::*;
