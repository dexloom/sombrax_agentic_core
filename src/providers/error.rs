//! Provider error types

use thiserror::Error;

/// Errors that can occur during provider operations
#[derive(Debug, Error)]
pub enum ProviderError {
    /// HTTP error with status code and message
    #[error("HTTP error: {status} - {message}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Error message body.
        message: String,
    },

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Rate limited by the provider
    #[error("Rate limited")]
    RateLimited {
        /// Optional retry delay in milliseconds
        retry_after_ms: Option<u64>,
    },

    /// Invalid response from the provider
    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    /// HTTP request error
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Environment variable not set
    #[error("Environment variable not set: {0}")]
    EnvVarNotSet(String),
}

/// Completion-level error (wraps ProviderError)
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Provider error
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    /// Invalid request configuration
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}
