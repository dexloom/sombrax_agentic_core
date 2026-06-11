//! Retry configuration for agent execution with exponential backoff.
//!
//! This module provides retry functionality for agent completion requests.
//! Only **transient errors** are retried to avoid re-running operations that
//! may have side effects.
//!
//! # Retry Behavior
//!
//! The retry logic only retries errors classified as retryable:
//!
//! - Rate limiting (HTTP 429)
//! - Network timeouts and connection errors
//! - Server errors (HTTP 500, 502, 503, 504)
//! - Temporary provider overload
//! - Empty or too-short responses (when validation is enabled)
//! - Tool execution errors (when validation is enabled)
//!
//! # Non-Retryable Errors
//!
//! Permanent errors are NOT retried:
//!
//! - Invalid parameters or configuration
//! - Authentication failures
//! - Permission denied
//!
//! # Usage
//!
//! ```rust,ignore
//! use sombrax_agentic_core::retry::{RetryConfig, ResponseValidation};
//!
//! let agent = Agent::builder(model)
//!     .retry_config(RetryConfig::default())  // 10 retries with exponential backoff
//!     .response_validation(ResponseValidation::min_length(100))  // Retry if response < 100 chars
//!     .build();
//!
//! // Or disable retries
//! let agent = Agent::builder(model)
//!     .retry_config(RetryConfig::no_retries())
//!     .build();
//! ```

use std::time::Duration;

/// Default retry delays in seconds (exponential backoff pattern).
pub const DEFAULT_RETRY_DELAYS_SECS: [u64; 5] = [1, 3, 10, 20, 60];

/// Default maximum number of retry attempts.
pub const DEFAULT_MAX_RETRIES: usize = 10;

/// Configuration for retry behavior during agent completion requests.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (default: 10).
    pub max_retries: usize,
    /// Delay between retries in seconds (uses index for exponential backoff).
    pub delays_secs: Vec<u64>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            delays_secs: DEFAULT_RETRY_DELAYS_SECS.to_vec(),
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with custom max retries.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::RetryConfig;
    ///
    /// let config = RetryConfig::with_max_retries(3);
    /// assert_eq!(config.max_retries, 3);
    /// ```
    pub fn with_max_retries(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Create a retry config with custom delays.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::RetryConfig;
    ///
    /// let config = RetryConfig::with_delays(3, vec![2, 5, 10]);
    /// assert_eq!(config.max_retries, 3);
    /// ```
    pub fn with_delays(max_retries: usize, delays_secs: Vec<u64>) -> Self {
        Self {
            max_retries,
            delays_secs,
        }
    }

    /// Create a retry config with no retries.
    ///
    /// Use this for operations that should fail immediately on error.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::RetryConfig;
    ///
    /// let config = RetryConfig::no_retries();
    /// assert_eq!(config.max_retries, 0);
    /// ```
    pub fn no_retries() -> Self {
        Self {
            max_retries: 0,
            delays_secs: vec![],
        }
    }

    /// Get the delay for a specific retry attempt.
    ///
    /// If the attempt index exceeds the delays array length,
    /// the last delay value is used.
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let secs = self
            .delays_secs
            .get(attempt)
            .copied()
            .unwrap_or_else(|| *self.delays_secs.last().unwrap_or(&60));
        Duration::from_secs(secs)
    }

    /// Check if retries are enabled.
    pub fn retries_enabled(&self) -> bool {
        self.max_retries > 0
    }
}

/// Validation configuration for response quality.
///
/// When enabled, the agent will retry if the response doesn't meet
/// the validation criteria.
#[derive(Debug, Clone, Default)]
pub struct ResponseValidation {
    /// Minimum response content length (in characters).
    /// If set, responses shorter than this will trigger a retry.
    pub min_length: Option<usize>,
    /// Maximum allowed tool errors before retrying the entire request.
    /// If set, when tool execution errors exceed this count, the agent
    /// will retry from the beginning.
    pub max_tool_errors: Option<usize>,
    /// Maximum retries specifically for validation failures.
    /// Uses the same delays as RetryConfig.
    /// Default: 3
    pub max_validation_retries: usize,
}

impl ResponseValidation {
    /// Create validation with minimum response length requirement.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::ResponseValidation;
    ///
    /// // Retry if response is less than 100 characters
    /// let validation = ResponseValidation::min_length(100);
    /// ```
    pub fn min_length(min_length: usize) -> Self {
        Self {
            min_length: Some(min_length),
            max_tool_errors: None,
            max_validation_retries: 3,
        }
    }

    /// Create validation with max tool errors limit.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::ResponseValidation;
    ///
    /// // Retry if more than 5 tool execution errors occur
    /// let validation = ResponseValidation::max_tool_errors(5);
    /// ```
    pub fn max_tool_errors(max_errors: usize) -> Self {
        Self {
            min_length: None,
            max_tool_errors: Some(max_errors),
            max_validation_retries: 3,
        }
    }

    /// Create validation with both min length and max tool errors.
    ///
    /// # Example
    ///
    /// ```
    /// use sombrax_agentic_core::retry::ResponseValidation;
    ///
    /// // Retry if response < 100 chars OR more than 3 tool errors
    /// let validation = ResponseValidation::new(100, 3);
    /// ```
    pub fn new(min_length: usize, max_tool_errors: usize) -> Self {
        Self {
            min_length: Some(min_length),
            max_tool_errors: Some(max_tool_errors),
            max_validation_retries: 3,
        }
    }

    /// Set the maximum number of validation retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_validation_retries = max_retries;
        self
    }

    /// Check if validation is enabled.
    pub fn is_enabled(&self) -> bool {
        self.min_length.is_some() || self.max_tool_errors.is_some()
    }

    /// Check if the response content passes the minimum length requirement.
    pub fn check_length(&self, content: &str) -> bool {
        match self.min_length {
            Some(min) => content.len() >= min,
            None => true,
        }
    }

    /// Check if the tool error count is acceptable.
    pub fn check_tool_errors(&self, error_count: usize) -> bool {
        match self.max_tool_errors {
            Some(max) => error_count <= max,
            None => true,
        }
    }

    /// Validate the response and return a validation result.
    pub fn validate(&self, content: &str, tool_error_count: usize) -> ValidationResult {
        if !self.check_length(content) {
            return ValidationResult::TooShort {
                actual: content.len(),
                minimum: self.min_length.unwrap_or(0),
            };
        }
        if !self.check_tool_errors(tool_error_count) {
            return ValidationResult::TooManyToolErrors {
                actual: tool_error_count,
                maximum: self.max_tool_errors.unwrap_or(0),
            };
        }
        ValidationResult::Valid
    }

    /// Validate the response, skipping length check.
    ///
    /// Use this for tool call responses where empty content is valid.
    pub fn validate_skip_length(&self, tool_error_count: usize) -> ValidationResult {
        if !self.check_tool_errors(tool_error_count) {
            return ValidationResult::TooManyToolErrors {
                actual: tool_error_count,
                maximum: self.max_tool_errors.unwrap_or(0),
            };
        }
        ValidationResult::Valid
    }
}

/// Result of response validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Response passed validation.
    Valid,
    /// Response content is too short.
    TooShort {
        /// Actual response length in characters.
        actual: usize,
        /// Minimum required length in characters.
        minimum: usize,
    },
    /// Too many tool execution errors occurred.
    TooManyToolErrors {
        /// Actual number of tool errors.
        actual: usize,
        /// Maximum allowed tool errors.
        maximum: usize,
    },
}

impl ValidationResult {
    /// Check if the validation passed.
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    /// Check if the validation failed and should trigger a retry.
    pub fn should_retry(&self) -> bool {
        !self.is_valid()
    }
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationResult::Valid => write!(f, "valid"),
            ValidationResult::TooShort { actual, minimum } => {
                write!(
                    f,
                    "response too short ({} chars, minimum {})",
                    actual, minimum
                )
            }
            ValidationResult::TooManyToolErrors { actual, maximum } => {
                write!(
                    f,
                    "too many tool errors ({} errors, maximum {})",
                    actual, maximum
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 10);
        assert_eq!(config.delays_secs.len(), 5);
        assert!(config.retries_enabled());
    }

    #[test]
    fn test_retry_config_delay_for_attempt() {
        let config = RetryConfig::default();

        // First attempt uses first delay
        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));

        // Second attempt
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(3));

        // Third attempt
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(10));

        // Beyond array bounds uses last value
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(60));
    }

    #[test]
    fn test_retry_config_no_retries() {
        let config = RetryConfig::no_retries();
        assert_eq!(config.max_retries, 0);
        assert!(!config.retries_enabled());
    }

    #[test]
    fn test_retry_config_custom() {
        let config = RetryConfig::with_max_retries(3);
        assert_eq!(config.max_retries, 3);
        assert!(config.retries_enabled());
    }

    #[test]
    fn test_retry_config_with_delays() {
        let config = RetryConfig::with_delays(2, vec![5, 10]);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(5));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(10));
        // Beyond bounds uses last value
        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(10));
    }

    #[test]
    fn test_response_validation_default() {
        let validation = ResponseValidation::default();
        assert!(!validation.is_enabled());
        assert!(validation.check_length(""));
        assert!(validation.check_tool_errors(100));
    }

    #[test]
    fn test_response_validation_min_length() {
        let validation = ResponseValidation::min_length(100);
        assert!(validation.is_enabled());

        // Short content fails
        assert!(!validation.check_length("short"));
        assert_eq!(
            validation.validate("short", 0),
            ValidationResult::TooShort {
                actual: 5,
                minimum: 100
            }
        );

        // Long enough content passes
        let long_content = "x".repeat(100);
        assert!(validation.check_length(&long_content));
        assert_eq!(
            validation.validate(&long_content, 0),
            ValidationResult::Valid
        );
    }

    #[test]
    fn test_response_validation_max_tool_errors() {
        let validation = ResponseValidation::max_tool_errors(3);
        assert!(validation.is_enabled());

        // Few errors pass
        assert!(validation.check_tool_errors(2));
        assert_eq!(validation.validate("content", 2), ValidationResult::Valid);

        // Too many errors fail
        assert!(!validation.check_tool_errors(5));
        assert_eq!(
            validation.validate("content", 5),
            ValidationResult::TooManyToolErrors {
                actual: 5,
                maximum: 3
            }
        );
    }

    #[test]
    fn test_response_validation_combined() {
        let validation = ResponseValidation::new(50, 3);
        assert!(validation.is_enabled());

        // Both pass
        let content = "x".repeat(50);
        assert_eq!(validation.validate(&content, 2), ValidationResult::Valid);

        // Length fails first
        assert_eq!(
            validation.validate("short", 2),
            ValidationResult::TooShort {
                actual: 5,
                minimum: 50
            }
        );

        // Tool errors fail (length ok)
        assert_eq!(
            validation.validate(&content, 5),
            ValidationResult::TooManyToolErrors {
                actual: 5,
                maximum: 3
            }
        );
    }

    #[test]
    fn test_validation_result_display() {
        assert_eq!(format!("{}", ValidationResult::Valid), "valid");
        assert_eq!(
            format!(
                "{}",
                ValidationResult::TooShort {
                    actual: 10,
                    minimum: 100
                }
            ),
            "response too short (10 chars, minimum 100)"
        );
        assert_eq!(
            format!(
                "{}",
                ValidationResult::TooManyToolErrors {
                    actual: 5,
                    maximum: 3
                }
            ),
            "too many tool errors (5 errors, maximum 3)"
        );
    }
}
