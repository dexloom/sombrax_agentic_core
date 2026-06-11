//! Anthropic (Claude) provider
//!
//! This module provides a client for Anthropic's Claude models via the Messages API.
//!
//! ## Features
//!
//! - Claude 3 family support (Opus, Sonnet, Haiku)
//! - Tool/function calling
//! - Streaming support (planned)
//!
//! ## Example
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::AnthropicClientBuilder;
//!
//! let client = AnthropicClientBuilder::new("your-api-key")
//!     .temperature(0.7)
//!     .max_tokens(4096)
//!     .build();
//!
//! let model = client.completion_model("claude-3-5-sonnet-20241022");
//! ```

pub mod client;
pub mod types;

pub use client::{AnthropicClient, AnthropicClientBuilder, AnthropicCompletionModel};
pub use types::*;
