//! Error types for the Agent Hook Library
//!
//! Provides structured error types with context for debugging (per SC-008).

use std::fmt;

/// Stage where a hook error occurred (for debugging per SC-008)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookStage {
    /// Error occurred during pre-completion hook
    PreCompletion,
    /// Error occurred during post-completion hook
    PostCompletion,
    /// Error occurred during pre-tool-call hook
    PreToolCall,
    /// Error occurred during post-tool-call hook
    PostToolCall,
    /// Error occurred during filter-tools hook
    FilterTools,
    /// Error occurred during on-assistant-message hook
    OnAssistantMessage,
}

impl fmt::Display for HookStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HookStage::PreCompletion => write!(f, "pre_completion"),
            HookStage::PostCompletion => write!(f, "post_completion"),
            HookStage::PreToolCall => write!(f, "pre_tool_call"),
            HookStage::PostToolCall => write!(f, "post_tool_call"),
            HookStage::FilterTools => write!(f, "filter_tools"),
            HookStage::OnAssistantMessage => write!(f, "on_assistant_message"),
        }
    }
}

/// Error type for hook operations (includes context per FR-017)
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    /// A hook failed during execution
    #[error("Hook '{hook_name}' failed at {stage}: {message}")]
    HookFailed {
        /// Name of the hook that failed
        hook_name: String,
        /// Stage where the failure occurred
        stage: HookStage,
        /// Error message
        message: String,
        /// Source error if available
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Hook chain was cancelled
    #[error("Hook chain cancelled")]
    Cancelled,

    /// Serialization error during hook processing
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl HookError {
    /// Create a new hook failed error
    pub fn hook_failed(
        hook_name: impl Into<String>,
        stage: HookStage,
        message: impl Into<String>,
    ) -> Self {
        Self::HookFailed {
            hook_name: hook_name.into(),
            stage,
            message: message.into(),
            source: None,
        }
    }

    /// Create a new hook failed error with a source error
    pub fn hook_failed_with_source(
        hook_name: impl Into<String>,
        stage: HookStage,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::HookFailed {
            hook_name: hook_name.into(),
            stage,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

/// Error type for completion operations
#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    /// HTTP error during API call
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Provider-specific error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Rate limited by provider
    #[error("Rate limited: retry after {retry_after_secs:?} seconds")]
    RateLimited {
        /// Number of seconds to wait before retrying
        retry_after_secs: Option<u64>,
    },

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication failed
    #[error("Authentication failed")]
    AuthenticationFailed,

    /// Request cancelled
    #[error("Request cancelled")]
    Cancelled,

    /// Hook error during request processing
    #[error("Hook error: {0}")]
    HookError(#[from] HookError),

    /// Tool error during execution
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),
}

impl From<reqwest::Error> for CompletionError {
    fn from(err: reqwest::Error) -> Self {
        CompletionError::HttpError(err.to_string())
    }
}

impl CompletionError {
    /// Check if this error is retryable (transient error).
    ///
    /// Retryable errors include:
    /// - Rate limiting (HTTP 429)
    /// - HTTP errors (network issues, timeouts)
    /// - Provider errors that appear transient
    ///
    /// Non-retryable errors include:
    /// - Authentication failures
    /// - Invalid requests
    /// - Tool execution errors (side effects may have occurred)
    /// - Hook errors
    pub fn is_retryable(&self) -> bool {
        match self {
            // Rate limiting is always retryable
            CompletionError::RateLimited { .. } => true,
            // HTTP errors are generally transient
            CompletionError::HttpError(msg) => {
                let lower = msg.to_lowercase();
                // Check for common transient patterns
                lower.contains("timeout")
                    || lower.contains("connection")
                    || lower.contains("network")
                    || lower.contains("500")
                    || lower.contains("502")
                    || lower.contains("503")
                    || lower.contains("504")
                    || lower.contains("529")
                    || lower.contains("overloaded")
                    || lower.contains("temporarily unavailable")
                    // reqwest errors for network issues during send/receive
                    || lower.contains("error sending request")
                    || lower.contains("error receiving")
                    || lower.contains("broken pipe")
                    || lower.contains("reset by peer")
                    || lower.contains("timed out")
                    // Response body decoding errors (partial/malformed responses)
                    || lower.contains("error decoding")
            }
            // Provider errors may be transient
            CompletionError::ProviderError(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("timeout")
                    || lower.contains("overload")
                    || lower.contains("capacity")
                    || lower.contains("unavailable")
            }
            // These are NOT retryable
            CompletionError::JsonError(_) => false,
            CompletionError::InvalidRequest(_) => false,
            CompletionError::AuthenticationFailed => false,
            CompletionError::Cancelled => false,
            CompletionError::HookError(_) => false,
            CompletionError::ToolError(_) => false,
        }
    }

    /// Get the suggested retry delay in seconds, if available.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            CompletionError::RateLimited { retry_after_secs } => *retry_after_secs,
            _ => None,
        }
    }
}

/// Error type for tool operations
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Invalid arguments
    #[error("Invalid arguments: {0}")]
    InvalidArguments(#[from] serde_json::Error),

    /// Tool call was interrupted
    #[error("Tool call interrupted")]
    Interrupted,

    /// MCP protocol error
    #[error("MCP error: {0}")]
    McpError(String),

    /// Tool call was blocked by a hook
    #[error("Tool call blocked: {0}")]
    Blocked(String),
}

/// Error type for registry operations
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Agent already registered
    #[error("Agent '{0}' already registered")]
    AlreadyRegistered(String),

    /// Agent not found
    #[error("Agent '{0}' not found")]
    NotFound(String),

    /// Maximum invocation depth exceeded
    #[error("Maximum invocation depth ({0}) exceeded - possible cycle detected")]
    MaxDepthExceeded(usize),

    /// Invocation failed
    #[error("Invocation failed: {0}")]
    InvocationFailed(String),
}

/// Result type for hook operations
pub type HookResult<T> = Result<T, HookError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_stage_display() {
        assert_eq!(HookStage::PreCompletion.to_string(), "pre_completion");
        assert_eq!(HookStage::PostCompletion.to_string(), "post_completion");
        assert_eq!(HookStage::PreToolCall.to_string(), "pre_tool_call");
        assert_eq!(HookStage::PostToolCall.to_string(), "post_tool_call");
        assert_eq!(HookStage::FilterTools.to_string(), "filter_tools");
        assert_eq!(
            HookStage::OnAssistantMessage.to_string(),
            "on_assistant_message"
        );
    }

    #[test]
    fn test_hook_error_display() {
        let err = HookError::hook_failed("TestHook", HookStage::PreCompletion, "test error");
        assert!(err.to_string().contains("TestHook"));
        assert!(err.to_string().contains("pre_completion"));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_completion_error_from_reqwest() {
        // Test that we can create completion errors from strings
        let err = CompletionError::HttpError("connection refused".to_string());
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_http_error_retryable_patterns() {
        // Classic patterns
        assert!(CompletionError::HttpError("connection refused".to_string()).is_retryable());
        assert!(CompletionError::HttpError("timeout occurred".to_string()).is_retryable());
        assert!(CompletionError::HttpError("network unreachable".to_string()).is_retryable());
        assert!(CompletionError::HttpError("502 bad gateway".to_string()).is_retryable());
        assert!(CompletionError::HttpError("503 service unavailable".to_string()).is_retryable());
        assert!(
            CompletionError::HttpError("HTTP 529: overloaded_error".to_string()).is_retryable()
        );
        assert!(
            CompletionError::HttpError("anthropic overloaded_error (529)".to_string())
                .is_retryable()
        );

        // reqwest-style errors that should now be retryable
        assert!(CompletionError::HttpError(
            "error sending request for url (https://api.example.com/v1/completions)".to_string()
        )
        .is_retryable());
        assert!(
            CompletionError::HttpError("error receiving response body".to_string()).is_retryable()
        );
        assert!(CompletionError::HttpError("broken pipe".to_string()).is_retryable());
        assert!(CompletionError::HttpError("reset by peer".to_string()).is_retryable());
        assert!(CompletionError::HttpError("operation timed out".to_string()).is_retryable());
        assert!(
            CompletionError::HttpError("error decoding response body".to_string()).is_retryable()
        );

        // Non-retryable HTTP errors (client errors, etc.)
        assert!(!CompletionError::HttpError("400 bad request".to_string()).is_retryable());
        assert!(!CompletionError::HttpError("401 unauthorized".to_string()).is_retryable());
        assert!(!CompletionError::HttpError("invalid json response".to_string()).is_retryable());
    }
}
