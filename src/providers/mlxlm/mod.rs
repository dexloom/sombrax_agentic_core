//! MLX-LM provider module
//!
//! Provides access to local MLX-LM server with support for custom chat templates.
//! MLX-LM is Apple's machine learning framework for running LLMs on Apple Silicon.
//!
//! # Chat Template Support
//!
//! MLX-LM servers can use custom Jinja chat templates. This provider supports:
//! - Standard OpenAI-compatible message format (default)
//! - Minimax chat template format with special delimiters
//!
//! # Usage
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::{MlxLmClientBuilder, ChatTemplate};
//!
//! // Basic usage with local server
//! let client = MlxLmClientBuilder::new()
//!     .base_url("http://localhost:8080")
//!     .build();
//!
//! let model = client.completion_model("mlx-community/model");
//!
//! // With Minimax chat template
//! let client = MlxLmClientBuilder::new()
//!     .base_url("http://localhost:8080")
//!     .chat_template(ChatTemplate::Minimax)
//!     .build();
//! ```

pub mod client;
pub mod types;

pub use client::*;
pub use types::*;
