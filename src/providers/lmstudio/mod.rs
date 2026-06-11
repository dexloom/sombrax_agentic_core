//! LMStudio provider module
//!
//! Provides access to LMStudio server with anti-repetition controls.
//! LMStudio handles chat templates server-side, making this provider
//! simpler than MLX-LM (no client-side template rendering needed).
//!
//! # Anti-Repetition Features
//!
//! LMStudio supports parameters that prevent models from entering repetition loops:
//! - `repeat_penalty` — penalizes tokens from prompt+output (primary control)
//! - `repetition_context_size` — how many recent tokens to check for repeats
//! - `frequency_penalty` — penalizes based on token frequency in output
//! - `presence_penalty` — penalizes based on token presence in output
//! - `min_p` — minimum probability floor for sampling
//!
//! # Usage
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::LmStudioClientBuilder;
//!
//! // Basic usage with local server
//! let client = LmStudioClientBuilder::new()
//!     .base_url("http://localhost:1234/v1")
//!     .repeat_penalty(1.15)
//!     .repetition_context_size(256)
//!     .build();
//!
//! let model = client.completion_model("MiniMax-M2.1-Q8");
//!
//! // With anti-loop preset
//! let client = LmStudioClientBuilder::new()
//!     .with_anti_loop_config()
//!     .with_anti_repetition_stops()
//!     .build();
//! ```

pub mod client;
pub mod types;

pub use client::*;
pub use types::*;
