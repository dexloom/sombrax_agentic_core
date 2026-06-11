//! OpenAI provider
//!
//! This module provides a client for OpenAI's GPT models via the Chat Completions API.
//!
//! ## Features
//!
//! - GPT-4 family support (GPT-4, GPT-4 Turbo, GPT-4o)
//! - GPT-3.5 Turbo support
//! - Tool/function calling
//! - Organization support
//! - Azure OpenAI compatible (via base_url)
//!
//! ## Example
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::OpenAIClientBuilder;
//!
//! let client = OpenAIClientBuilder::new("your-api-key")
//!     .temperature(0.7)
//!     .max_tokens(4096)
//!     .build();
//!
//! let model = client.completion_model("gpt-4o");
//! ```

pub mod client;
pub mod types;

pub use client::{OpenAIClient, OpenAIClientBuilder, OpenAICompletionModel};
pub use types::*;
