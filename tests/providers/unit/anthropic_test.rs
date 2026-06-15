//! Unit tests for Anthropic provider

use sombrax_agentic_core::providers::anthropic::types::*;
use sombrax_agentic_core::providers::AnthropicClientBuilder;

#[test]
fn test_anthropic_client_builder() {
    let client = AnthropicClientBuilder::new("test-api-key")
        .temperature(0.7)
        .top_p(0.9)
        .top_k(50)
        .max_tokens(4096)
        .build();

    let model = client.completion_model("claude-3-5-sonnet-20241022");
    assert_eq!(model.model_id(), "claude-3-5-sonnet-20241022");
    assert_eq!(model.provider(), "anthropic");
}

#[test]
fn test_anthropic_client_builder_base_url() {
    let client = AnthropicClientBuilder::new("test-api-key")
        .base_url("https://custom.api.com")
        .build();

    let model = client.completion_model("claude-3-opus-20240229");
    assert_eq!(model.model_id(), "claude-3-opus-20240229");
}

#[test]
fn test_anthropic_content_text() {
    let content = AnthropicContent::Text("Hello, world!".to_string());
    assert_eq!(content.text(), "Hello, world!");
}

#[test]
fn test_anthropic_content_blocks() {
    let content = AnthropicContent::Blocks(vec![
        AnthropicContentBlock::Text {
            text: "Part 1".to_string(),
            cache_control: None,
        },
        AnthropicContentBlock::Text {
            text: "Part 2".to_string(),
            cache_control: None,
        },
    ]);
    assert_eq!(content.text(), "Part 1Part 2");
}

#[test]
fn test_anthropic_content_blocks_with_tool_use() {
    let content = AnthropicContent::Blocks(vec![
        AnthropicContentBlock::Text {
            text: "Let me use a tool".to_string(),
            cache_control: None,
        },
        AnthropicContentBlock::ToolUse {
            id: "tool_123".to_string(),
            name: "search".to_string(),
            input: serde_json::json!({"query": "test"}),
            cache_control: None,
        },
    ]);
    // text() should only extract text blocks
    assert_eq!(content.text(), "Let me use a tool");
}

#[test]
fn test_anthropic_message_serialization() {
    let message = AnthropicMessage {
        role: "user".to_string(),
        content: AnthropicContent::Text("Hello".to_string()),
    };

    let json = serde_json::to_string(&message).unwrap();
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("Hello"));
}

#[test]
fn test_anthropic_tool_definition() {
    let tool = AnthropicTool {
        name: "get_weather".to_string(),
        description: "Get weather for a location".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        cache_control: None,
    };

    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains("get_weather"));
    assert!(json.contains("input_schema"));
}

#[test]
fn test_anthropic_tool_choice_auto() {
    let choice = AnthropicToolChoice::Auto;
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("auto"));
}

#[test]
fn test_anthropic_tool_choice_any() {
    let choice = AnthropicToolChoice::Any;
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("any"));
}

#[test]
fn test_anthropic_tool_choice_specific() {
    let choice = AnthropicToolChoice::Tool {
        name: "search".to_string(),
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("tool"));
    assert!(json.contains("search"));
}

#[test]
fn test_anthropic_request_serialization() {
    let request = AnthropicRequest {
        model: "claude-3-5-sonnet-20241022".to_string(),
        messages: vec![AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContent::Text("Hello".to_string()),
        }],
        max_tokens: 4096,
        system: Some(AnthropicSystem::Text("You are helpful".to_string())),
        temperature: Some(0.7),
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        metadata: None,
        thinking: None,
        stream: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("claude-3-5-sonnet"));
    assert!(json.contains("max_tokens"));
    assert!(json.contains("system"));
}

#[test]
fn test_anthropic_response_deserialization() {
    let json = r#"{
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }"#;

    let response: AnthropicResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.id, "msg_123");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.output_tokens, 5);
}

#[test]
fn test_anthropic_response_with_tool_use() {
    let json = r#"{
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me search for that"},
            {"type": "tool_use", "id": "tool_123", "name": "search", "input": {"query": "test"}}
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {"input_tokens": 15, "output_tokens": 20}
    }"#;

    let response: AnthropicResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.content.len(), 2);
    assert_eq!(response.stop_reason, Some("tool_use".to_string()));
}
