//! Mocked integration tests for ZAI provider

use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test ZAI client creation with mock server
#[tokio::test]
async fn test_zai_client_creation_with_mock() {
    let mock_server = MockServer::start().await;

    let response_body = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "zai-001",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    // Create client with mock server URL
    let _client = sombrax_agentic_core::providers::ZaiClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    // Client was created successfully
}

/// Test ZAI client with thinking mode
#[tokio::test]
async fn test_zai_client_with_thinking_mode() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-456",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::ZaiClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .enable_thinking(true)
        .build();
}

/// Test ZAI rate limit error setup
#[tokio::test]
async fn test_zai_rate_limit_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::ZaiClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test ZAI auth error setup
#[tokio::test]
async fn test_zai_auth_error_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "Invalid API key", "type": "authentication_error"}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::ZaiClientBuilder::new("invalid-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test ZAI tool call response setup
#[tokio::test]
async fn test_zai_tool_call_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-789",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::ZaiClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();
}
