//! Tool error types

use thiserror::Error;

/// Errors that can occur during tool execution
#[derive(Debug, Error)]
pub enum ToolError {
    /// Validation error for tool arguments
    #[error("Validation error: {0}")]
    Validation(String),

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Path is outside the allowed workspace
    #[error("Path outside workspace: {0}")]
    PathOutsideWorkspace(String),

    /// Command was rejected by safety validation
    #[error("Command rejected: {0}")]
    CommandRejected(String),

    /// Operation timed out
    #[error("Timeout after {0}ms")]
    Timeout(u64),

    /// Maximum recursion depth exceeded
    #[error("Max recursion depth exceeded")]
    MaxRecursionDepth,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Regex error
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    /// HTTP error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
