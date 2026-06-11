//! Extractor module for structured data extraction from LLM responses.
//!
//! This module provides `ExtractorWrapper` for extracting structured data
//! from text using LLM providers with JSON schema-based tool calling.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sombrax_agentic_core::extractor::{build_extractor, ExtractorWrapper};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Debug, JsonSchema, Deserialize)]
//! struct ReviewResponse {
//!     rating: u8,
//!     note: String,
//! }
//!
//! let extractor = build_extractor(&llm_config)?;
//! let review: ReviewResponse = extractor
//!     .extract(&llm_config.model(), preamble, prompt)
//!     .await?;
//! ```

mod builder;
mod wrapper;

pub use builder::{build_extractor, ExtractorBuildError};
pub use wrapper::{ExtractorError, ExtractorWrapper};
