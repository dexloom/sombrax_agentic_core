//! Mocked integration tests for OpenRouter provider

use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test OpenRouter client creation with mock server
#[tokio::test]
async fn test_openrouter_client_creation_with_mock() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-openrouter-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gen-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        })))
        .mount(&mock_server)
        .await;

    let _client =
        sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-openrouter-key")
            .base_url(&mock_server.uri())
            .build();
}

/// Test OpenRouter with whitelist routing
#[tokio::test]
async fn test_openrouter_with_whitelist_routing() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gen-456",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .whitelist(vec!["anthropic".to_string(), "openai".to_string()])
        .build();
}

/// Test OpenRouter with blacklist routing
#[tokio::test]
async fn test_openrouter_with_blacklist_routing() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gen-789",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .blacklist(vec!["azure".to_string()])
        .build();
}

/// Test OpenRouter XML tool call parsing
#[tokio::test]
async fn test_openrouter_xml_tool_calls() {
    use sombrax_agentic_core::providers::openrouter::parse_minimax_xml_tool_calls;

    let content = r#"<minimax:tool_call><invoke name="get_weather"><parameter name="location">NYC</parameter></invoke></minimax:tool_call>"#;
    let tool_calls = parse_minimax_xml_tool_calls(content);

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
}

/// Test OpenRouter duplicate JSON extraction
#[tokio::test]
async fn test_openrouter_json_extraction() {
    use sombrax_agentic_core::providers::openrouter::extract_first_json_object;

    let content = r#"{"name": "test"} extra {"name": "test"}"#;
    let result = extract_first_json_object(content);

    assert!(result.is_some());
    let json_str = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["name"], "test");
}

/// Test OpenRouter with fallbacks disabled
#[tokio::test]
async fn test_openrouter_no_fallbacks() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gen-no-fb",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .allow_fallbacks(false)
        .build();
}

/// Test OpenRouter tool call response setup
#[tokio::test]
async fn test_openrouter_tool_call_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gen-tools",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 25, "completion_tokens": 30, "total_tokens": 55}
        })))
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .build();
}

/// Test OpenRouter provider error mock setup
#[tokio::test]
async fn test_openrouter_provider_error_mock_setup() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503)
                .set_body_json(json!({"error": {"message": "Upstream unavailable"}})),
        )
        .mount(&mock_server)
        .await;

    let _client = sombrax_agentic_core::providers::OpenRouterClientBuilder::new("test-key")
        .base_url(&mock_server.uri())
        .build();
}
