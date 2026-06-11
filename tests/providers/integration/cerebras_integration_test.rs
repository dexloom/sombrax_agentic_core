//! Mocked integration tests for Cerebras provider

use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test Cerebras client creation with mock server
#[tokio::test]
async fn test_cerebras_client_creation_with_mock() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-cerebras-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::CerebrasClientBuilder::new("test-cerebras-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test Cerebras tool call response setup
#[tokio::test]
async fn test_cerebras_tool_call_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-456",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_xyz", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 25, "total_tokens": 45}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::CerebrasClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test Cerebras string tool content extraction
#[tokio::test]
async fn test_cerebras_string_tool_content() {
    use sombrax_agentic_core::providers::cerebras::extract_tool_result_content;

    let result = extract_tool_result_content("plain text result");
    assert_eq!(result, "plain text result");

    let json_str = r#"{"result": "success"}"#;
    let result = extract_tool_result_content(json_str);
    assert_eq!(result, json_str);
}

/// Test Cerebras rate limit mock setup
#[tokio::test]
async fn test_cerebras_rate_limit_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(json!({"error": {"message": "Too many requests"}})),
        )
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::CerebrasClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test Cerebras with sampling parameters
#[tokio::test]
async fn test_cerebras_with_sampling_params() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-789",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::CerebrasClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .temperature(0.9)
        .top_p(0.95)
        .build();
}
