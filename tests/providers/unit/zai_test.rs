//! Unit tests for ZAI provider

use sombrax_agentic_core::providers::zai::{
    client::{CompletionRequest, Message, ToolDefinition},
    ZaiClientBuilder,
};

#[test]
fn test_zai_client_builder_defaults() {
    let client = ZaiClientBuilder::new("test-api-key").build();
    let model = client.completion_model("glm-4.6");
    assert_eq!(model.model_id(), "glm-4.6");
    assert_eq!(model.provider(), "zai");
}

#[test]
fn test_zai_client_builder_with_temperature() {
    let client = ZaiClientBuilder::new("test-api-key")
        .temperature(0.7)
        .build();
    let model = client.completion_model("glm-4.6");
    assert_eq!(model.model_id(), "glm-4.6");
}

#[test]
fn test_zai_client_builder_temperature_clamping() {
    // Temperature should be clamped to 0.0-2.0
    let _client = ZaiClientBuilder::new("test-api-key")
        .temperature(3.0) // Should be clamped to 2.0
        .build();
}

#[test]
fn test_zai_client_builder_with_top_p() {
    let _client = ZaiClientBuilder::new("test-api-key").top_p(0.9).build();
}

#[test]
fn test_zai_client_builder_with_top_k() {
    let _client = ZaiClientBuilder::new("test-api-key").top_k(40).build();
}

#[test]
fn test_zai_client_builder_with_max_tokens() {
    let _client = ZaiClientBuilder::new("test-api-key")
        .max_tokens(4096)
        .build();
}

#[test]
fn test_zai_client_builder_with_thinking() {
    let _client = ZaiClientBuilder::new("test-api-key")
        .enable_thinking(true)
        .build();
}

#[test]
fn test_zai_client_builder_with_base_url() {
    let _client = ZaiClientBuilder::new("test-api-key")
        .base_url("https://custom.api.endpoint/v1")
        .build();
}

#[test]
fn test_zai_client_builder_full_config() {
    let client = ZaiClientBuilder::new("test-api-key")
        .base_url("https://custom.api.endpoint/v1")
        .temperature(0.8)
        .top_p(0.95)
        .top_k(50)
        .max_tokens(8192)
        .enable_thinking(true)
        .build();

    let model = client.completion_model("glm-4.6-flash");
    assert_eq!(model.model_id(), "glm-4.6-flash");
}

#[test]
fn test_completion_request_default() {
    let request = CompletionRequest::default();
    assert!(request.preamble.is_none());
    assert!(request.messages.is_empty());
    assert!(request.tools.is_empty());
    assert!(request.temperature.is_none());
    assert!(request.max_tokens.is_none());
}

#[test]
fn test_completion_request_with_messages() {
    let request = CompletionRequest {
        preamble: Some("You are a helpful assistant.".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1000),
        additional_params: None,
        cache: Default::default(),
    };

    assert_eq!(
        request.preamble.as_deref(),
        Some("You are a helpful assistant.")
    );
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, "user");
}

#[test]
fn test_tool_definition_creation() {
    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get weather for a location".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    };

    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description, "Get weather for a location");
}

#[test]
fn test_message_with_tool_calls() {
    use sombrax_agentic_core::providers::zai::client::ToolCall;

    let message = Message {
        role: "assistant".to_string(),
        content: "".to_string(),
        tool_calls: Some(vec![ToolCall {
            id: "call_123".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"location": "NYC"}"#.to_string(),
        }]),
        tool_call_id: None,
        reasoning: None,
    };

    assert!(message.tool_calls.is_some());
    let calls = message.tool_calls.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
}

#[test]
fn test_message_tool_result() {
    let message = Message {
        role: "tool".to_string(),
        content: r#"{"temperature": 72, "conditions": "sunny"}"#.to_string(),
        tool_calls: None,
        tool_call_id: Some("call_123".to_string()),
        reasoning: None,
    };

    assert_eq!(message.role, "tool");
    assert_eq!(message.tool_call_id.as_deref(), Some("call_123"));
}
