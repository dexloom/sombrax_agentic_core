//! Environment variable expansion for string values.
//!
//! Supports `${VAR_NAME}` syntax only (braced form).
//! Bare `$VAR_NAME` is intentionally NOT supported to avoid conflicts
//! with legitimate URL patterns (e.g., OData `$select`, `$filter`).
//! Returns an error if a referenced variable is not set.

use std::sync::LazyLock;

use regex::Regex;

use crate::tools::error::ToolError;

/// Regex matching `${VAR_NAME}` only (braced form).
/// Bare `$VAR` is not matched to avoid breaking URLs with legitimate `$` usage.
static ENV_VAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").unwrap());

/// Expand environment variable references in the input string.
///
/// Supports `${VAR_NAME}` syntax (braced form only).
/// Bare `$VAR_NAME` is intentionally not expanded to avoid conflicts
/// with URLs that use `$` as part of their API syntax (e.g., OData).
///
/// Returns `ToolError::Validation` if any referenced variable is not set.
pub fn expand_env_vars(input: &str) -> Result<String, ToolError> {
    if !input.contains("${") {
        return Ok(input.to_string());
    }

    let mut last_error: Option<ToolError> = None;

    let result = ENV_VAR_RE.replace_all(input, |caps: &regex::Captures| {
        let var_name = caps.get(1).map(|m| m.as_str()).unwrap_or("");

        match std::env::var(var_name) {
            Ok(value) => value,
            Err(_) => {
                last_error = Some(ToolError::Validation(format!(
                    "Environment variable not set: ${}",
                    var_name
                )));
                caps[0].to_string()
            }
        }
    });

    if let Some(err) = last_error {
        return Err(err);
    }

    Ok(result.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_vars() {
        let result = expand_env_vars("https://example.com/api/v1").unwrap();
        assert_eq!(result, "https://example.com/api/v1");
    }

    #[test]
    fn test_bare_var_not_expanded() {
        // Bare $VAR is intentionally NOT expanded to avoid OData conflicts
        unsafe { std::env::set_var("TEST_EXPAND_KEY", "abc123") };
        let result = expand_env_vars("https://api.example.com?key=$TEST_EXPAND_KEY").unwrap();
        assert_eq!(result, "https://api.example.com?key=$TEST_EXPAND_KEY");
        unsafe { std::env::remove_var("TEST_EXPAND_KEY") };
    }

    #[test]
    fn test_braced_var() {
        unsafe { std::env::set_var("TEST_EXPAND_BRACED", "xyz") };
        let result = expand_env_vars("https://api.example.com/${TEST_EXPAND_BRACED}/path").unwrap();
        assert_eq!(result, "https://api.example.com/xyz/path");
        unsafe { std::env::remove_var("TEST_EXPAND_BRACED") };
    }

    #[test]
    fn test_multiple_braced_vars() {
        unsafe { std::env::set_var("TEST_HOST", "api.example.com") };
        unsafe { std::env::set_var("TEST_KEY", "secret") };
        let result = expand_env_vars("https://${TEST_HOST}/v1?key=${TEST_KEY}").unwrap();
        assert_eq!(result, "https://api.example.com/v1?key=secret");
        unsafe { std::env::remove_var("TEST_HOST") };
        unsafe { std::env::remove_var("TEST_KEY") };
    }

    #[test]
    fn test_missing_var_returns_error() {
        unsafe { std::env::remove_var("DEFINITELY_NOT_SET_VAR_12345") };
        let result = expand_env_vars("https://example.com?key=${DEFINITELY_NOT_SET_VAR_12345}");
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("DEFINITELY_NOT_SET_VAR_12345"));
    }

    #[test]
    fn test_empty_string() {
        let result = expand_env_vars("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_dollar_sign_not_followed_by_identifier() {
        let result = expand_env_vars("price is $100").unwrap();
        assert_eq!(result, "price is $100");
    }

    #[test]
    fn test_odata_params_pass_through() {
        // OData-style query parameters should not trigger expansion
        let result =
            expand_env_vars("https://api.example.com/items?$select=name&$filter=active").unwrap();
        assert_eq!(
            result,
            "https://api.example.com/items?$select=name&$filter=active"
        );
    }

    #[test]
    fn test_adjacent_text_with_braces() {
        unsafe { std::env::set_var("TEST_PREFIX", "api") };
        let result = expand_env_vars("${TEST_PREFIX}key=value").unwrap();
        assert_eq!(result, "apikey=value");
        unsafe { std::env::remove_var("TEST_PREFIX") };
    }
}
