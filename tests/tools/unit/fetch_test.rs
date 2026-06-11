//! Unit tests for FetchTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use sombrax_agentic_core::tools::web::{FetchArgs, FetchTool};
use std::collections::HashMap;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_fetch_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "fetch");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_fetch_tool_invalid_method() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    let result = tool
        .call(FetchArgs {
            url: "https://example.com".to_string(),
            method: "INVALID".to_string(),
            body: None,
            headers: None,
            timeout: None,
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_fetch_tool_valid_methods() {
    // Just verify that valid methods don't cause validation errors
    let valid_methods = vec!["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];

    for method in valid_methods {
        let (_temp_dir, ctx) = create_test_context();
        let tool = FetchTool::new(ctx);

        // We can't actually make HTTP requests in unit tests without a mock server
        // so we just validate that the method validation passes
        // The actual HTTP call would fail with a connection error which is expected
        let result = tool
            .call(FetchArgs {
                url: "http://localhost:99999/nonexistent".to_string(), // Use a port that's likely not in use
                method: method.to_string(),
                body: None,
                headers: None,
                timeout: Some(100), // Very short timeout to fail fast
            })
            .await;

        // Should fail with HTTP error, not validation error
        if result.is_err() {
            let err = format!("{:?}", result.err().unwrap());
            assert!(
                !err.contains("Invalid HTTP method"),
                "Method {} should be valid",
                method
            );
        }
    }
}

#[tokio::test]
async fn test_fetch_tool_case_insensitive_method() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    // lowercase should work
    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "get".to_string(),
            body: None,
            headers: None,
            timeout: Some(100),
        })
        .await;

    // Should fail with connection error, not validation error
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(!err.contains("Invalid HTTP method"));
    }
}

#[tokio::test]
async fn test_fetch_tool_with_headers() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    let mut headers = HashMap::new();
    headers.insert("X-Custom-Header".to_string(), "test-value".to_string());
    headers.insert("Authorization".to_string(), "Bearer token".to_string());

    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: Some(headers),
            timeout: Some(100),
        })
        .await;

    // Headers should be accepted without validation error
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(!err.contains("header"));
    }
}

#[tokio::test]
async fn test_fetch_tool_with_body() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    let body = serde_json::json!({
        "key": "value",
        "number": 42
    });

    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "POST".to_string(),
            body: Some(body),
            headers: None,
            timeout: Some(100),
        })
        .await;

    // Body should be accepted without validation error
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(!err.contains("body"));
    }
}

#[tokio::test]
async fn test_fetch_tool_timeout_clamping() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    // Very large timeout should be accepted (clamped internally)
    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: None,
            timeout: Some(999999999), // Will be clamped to max
        })
        .await;

    // Should not fail on timeout validation
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(!err.contains("timeout") || err.contains("Timeout")); // Timeout error is ok, but not validation
    }
}

#[tokio::test]
async fn test_fetch_tool_env_var_expansion_in_url() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    unsafe { std::env::set_var("TEST_FETCH_HOST", "localhost") };

    let result = tool
        .call(FetchArgs {
            url: "http://${TEST_FETCH_HOST}:99999/nonexistent".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: None,
            timeout: Some(100),
        })
        .await;

    // Should fail with HTTP/connection error, not a validation error about env vars
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(
            !err.contains("Environment variable not set"),
            "Env var should have been expanded"
        );
    }

    unsafe { std::env::remove_var("TEST_FETCH_HOST") };
}

#[tokio::test]
async fn test_fetch_tool_missing_env_var_returns_error() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    unsafe { std::env::remove_var("NONEXISTENT_VAR_FOR_FETCH_TEST") };

    let result = tool
        .call(FetchArgs {
            url: "https://api.example.com?key=${NONEXISTENT_VAR_FOR_FETCH_TEST}".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: None,
            timeout: Some(100),
        })
        .await;

    assert!(result.is_err());
    let err = format!("{:?}", result.err().unwrap());
    assert!(err.contains("NONEXISTENT_VAR_FOR_FETCH_TEST"));
}

#[tokio::test]
async fn test_fetch_tool_env_var_in_headers() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    unsafe { std::env::set_var("TEST_FETCH_TOKEN", "my-secret-token") };

    let mut headers = HashMap::new();
    headers.insert(
        "Authorization".to_string(),
        "Bearer ${TEST_FETCH_TOKEN}".to_string(),
    );

    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: Some(headers),
            timeout: Some(100),
        })
        .await;

    // Should fail with HTTP error, not env var validation error
    if result.is_err() {
        let err = format!("{:?}", result.err().unwrap());
        assert!(
            !err.contains("Environment variable not set"),
            "Env var in header should have been expanded"
        );
    }

    unsafe { std::env::remove_var("TEST_FETCH_TOKEN") };
}

#[tokio::test]
async fn test_fetch_tool_missing_env_var_in_headers() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);

    unsafe { std::env::remove_var("NONEXISTENT_HEADER_VAR") };

    let mut headers = HashMap::new();
    headers.insert(
        "Authorization".to_string(),
        "Bearer ${NONEXISTENT_HEADER_VAR}".to_string(),
    );

    let result = tool
        .call(FetchArgs {
            url: "http://localhost:99999/nonexistent".to_string(),
            method: "GET".to_string(),
            body: None,
            headers: Some(headers),
            timeout: Some(100),
        })
        .await;

    assert!(result.is_err());
    let err = format!("{:?}", result.err().unwrap());
    assert!(err.contains("NONEXISTENT_HEADER_VAR"));
}
