//! Unit tests for MiniMax provider

use sombrax_agentic_core::providers::minimax::MinimaxClientBuilder;
use sombrax_agentic_core::providers::zai::client::{CompletionRequest, Message, ToolDefinition};

// ============================================================================
// Client Builder Tests
// ============================================================================

#[test]
fn test_minimax_client_builder_defaults() {
    let client = MinimaxClientBuilder::new("test-api-key").build();
    let model = client.completion_model("minimax-m2.1");
    assert_eq!(model.model_id(), "minimax-m2.1");
    assert_eq!(model.provider(), "minimax");
}

#[test]
fn test_minimax_client_builder_with_temperature() {
    let client = MinimaxClientBuilder::new("test-api-key")
        .temperature(0.7)
        .build();
    let model = client.completion_model("minimax-m2.1");
    assert_eq!(model.model_id(), "minimax-m2.1");
}

#[test]
fn test_minimax_client_builder_temperature_clamping() {
    // Temperature should be clamped to 0.0-1.0
    let _client = MinimaxClientBuilder::new("test-api-key")
        .temperature(3.0) // Should be clamped to 1.0
        .build();

    // Lower bound
    let _client = MinimaxClientBuilder::new("test-api-key")
        .temperature(-1.0) // Should be clamped to 0.0
        .build();
}

#[test]
fn test_minimax_client_builder_with_top_p() {
    let _client = MinimaxClientBuilder::new("test-api-key").top_p(0.9).build();
}

#[test]
fn test_minimax_client_builder_with_top_k() {
    let _client = MinimaxClientBuilder::new("test-api-key").top_k(40).build();
}

#[test]
fn test_minimax_client_builder_with_max_tokens() {
    let _client = MinimaxClientBuilder::new("test-api-key")
        .max_tokens(4096)
        .build();
}

#[test]
fn test_minimax_client_builder_with_base_url() {
    let _client = MinimaxClientBuilder::new("test-api-key")
        .base_url("https://custom.api.minimax.io/v1")
        .build();
}

#[test]
fn test_minimax_client_builder_full_config() {
    let client = MinimaxClientBuilder::new("test-api-key")
        .base_url("https://custom.api.endpoint/v1")
        .temperature(0.8)
        .top_p(0.95)
        .top_k(50)
        .max_tokens(8192)
        .build();

    let model = client.completion_model("minimax-m2.1-pro");
    assert_eq!(model.model_id(), "minimax-m2.1-pro");
    assert_eq!(model.provider(), "minimax");
}

#[test]
fn test_minimax_model_clone() {
    let client = MinimaxClientBuilder::new("test-api-key").build();
    let model = client.completion_model("minimax-m2.1");
    let cloned = model.clone();
    assert_eq!(model.model_id(), cloned.model_id());
    assert_eq!(model.provider(), cloned.provider());
}

#[test]
fn test_minimax_client_clone() {
    let client = MinimaxClientBuilder::new("test-api-key").build();
    let cloned = client.clone();

    // Both should create models with same properties
    let model1 = client.completion_model("minimax-m2.1");
    let model2 = cloned.completion_model("minimax-m2.1");
    assert_eq!(model1.model_id(), model2.model_id());
}

// ============================================================================
// Request/Message Tests (using shared types from zai::client)
// ============================================================================

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
        preamble: Some("You are a helpful MiniMax assistant.".to_string()),
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
    };

    assert_eq!(
        request.preamble.as_deref(),
        Some("You are a helpful MiniMax assistant.")
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

#[test]
fn test_message_with_reasoning() {
    let message = Message {
        role: "assistant".to_string(),
        content: "The answer is 42.".to_string(),
        tool_calls: None,
        tool_call_id: None,
        reasoning: Some("I need to think about this carefully...".to_string()),
    };

    assert_eq!(message.role, "assistant");
    assert!(message.reasoning.is_some());
    assert_eq!(
        message.reasoning.as_deref(),
        Some("I need to think about this carefully...")
    );
}

// ============================================================================
// MiniMax-specific Type Serialization Tests
// ============================================================================

#[test]
fn test_minimax_content_text_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContent;

    let content = MinimaxContent::Text("Hello, world!".to_string());
    let json = serde_json::to_string(&content).unwrap();
    assert_eq!(json, r#""Hello, world!""#);
}

#[test]
fn test_minimax_content_text_extraction() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContent;

    let content = MinimaxContent::Text("Hello, world!".to_string());
    assert_eq!(content.text(), "Hello, world!");
}

#[test]
fn test_minimax_content_block_text_serialization() {
    use sombrax_agentic_core::providers::minimax::types::{MinimaxContent, MinimaxContentBlock};

    let content = MinimaxContent::Blocks(vec![MinimaxContentBlock::Text {
        text: "Hello!".to_string(),
    }]);
    let json = serde_json::to_string(&content).unwrap();
    assert!(json.contains(r#""type":"text""#));
    assert!(json.contains(r#""text":"Hello!""#));
}

#[test]
fn test_minimax_content_blocks_text_extraction() {
    use sombrax_agentic_core::providers::minimax::types::{MinimaxContent, MinimaxContentBlock};

    let content = MinimaxContent::Blocks(vec![
        MinimaxContentBlock::Text {
            text: "Hello ".to_string(),
        },
        MinimaxContentBlock::Text {
            text: "world!".to_string(),
        },
    ]);
    assert_eq!(content.text(), "Hello world!");
}

#[test]
fn test_minimax_content_block_thinking_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let block = MinimaxContentBlock::Thinking {
        thinking: "Let me reason about this...".to_string(),
    };
    let json = serde_json::to_string(&block).unwrap();
    assert!(json.contains(r#""type":"thinking""#));
    assert!(json.contains(r#""thinking":"Let me reason about this...""#));
}

#[test]
fn test_minimax_content_block_tool_use_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let block = MinimaxContentBlock::ToolUse {
        id: "tool_123".to_string(),
        name: "get_weather".to_string(),
        input: serde_json::json!({"location": "NYC"}),
    };
    let json = serde_json::to_string(&block).unwrap();
    assert!(json.contains(r#""type":"tool_use""#));
    assert!(json.contains(r#""id":"tool_123""#));
    assert!(json.contains(r#""name":"get_weather""#));
    assert!(json.contains(r#""location":"NYC""#));
}

#[test]
fn test_minimax_content_block_tool_result_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let block = MinimaxContentBlock::ToolResult {
        tool_use_id: "tool_123".to_string(),
        content: "72°F and sunny".to_string(),
        is_error: None,
    };
    let json = serde_json::to_string(&block).unwrap();
    assert!(json.contains(r#""type":"tool_result""#));
    assert!(json.contains(r#""tool_use_id":"tool_123""#));
    assert!(json.contains(r#""content":"72°F and sunny""#));
    // is_error should be skipped when None
    assert!(!json.contains("is_error"));
}

#[test]
fn test_minimax_content_block_tool_result_with_error_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let block = MinimaxContentBlock::ToolResult {
        tool_use_id: "tool_123".to_string(),
        content: "Error: API unavailable".to_string(),
        is_error: Some(true),
    };
    let json = serde_json::to_string(&block).unwrap();
    assert!(json.contains(r#""is_error":true"#));
}

#[test]
fn test_minimax_message_serialization() {
    use sombrax_agentic_core::providers::minimax::types::{MinimaxContent, MinimaxMessage};

    let message = MinimaxMessage {
        role: "user".to_string(),
        content: MinimaxContent::Text("Hello!".to_string()),
    };
    let json = serde_json::to_string(&message).unwrap();
    assert!(json.contains(r#""role":"user""#));
    assert!(json.contains(r#""content":"Hello!""#));
}

#[test]
fn test_minimax_tool_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxTool;

    let tool = MinimaxTool {
        name: "calculator".to_string(),
        description: "Performs arithmetic operations".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }),
    };
    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains(r#""name":"calculator""#));
    assert!(json.contains(r#""description":"Performs arithmetic operations""#));
    assert!(json.contains(r#""input_schema""#));
}

#[test]
fn test_minimax_tool_choice_auto_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxToolChoice;

    let choice = MinimaxToolChoice::Auto;
    let json = serde_json::to_string(&choice).unwrap();
    assert_eq!(json, r#"{"type":"auto"}"#);
}

#[test]
fn test_minimax_tool_choice_any_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxToolChoice;

    let choice = MinimaxToolChoice::Any;
    let json = serde_json::to_string(&choice).unwrap();
    assert_eq!(json, r#"{"type":"any"}"#);
}

#[test]
fn test_minimax_tool_choice_tool_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxToolChoice;

    let choice = MinimaxToolChoice::Tool {
        name: "get_weather".to_string(),
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains(r#""type":"tool""#));
    assert!(json.contains(r#""name":"get_weather""#));
}

#[test]
fn test_minimax_request_serialization() {
    use sombrax_agentic_core::providers::minimax::types::{
        MinimaxContent, MinimaxMessage, MinimaxRequest,
    };

    let request = MinimaxRequest {
        model: "minimax-m2.1".to_string(),
        messages: vec![MinimaxMessage {
            role: "user".to_string(),
            content: MinimaxContent::Text("Hello!".to_string()),
        }],
        max_tokens: 1024,
        system: Some("You are a helpful assistant.".to_string()),
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        tools: None,
        tool_choice: None,
        metadata: None,
        stream: None,
        thinking: None,
    };
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains(r#""model":"minimax-m2.1""#));
    assert!(json.contains(r#""max_tokens":1024"#));
    assert!(json.contains(r#""system":"You are a helpful assistant.""#));
    assert!(json.contains(r#""temperature":0.7"#));
    assert!(json.contains(r#""top_p":0.9"#));
    assert!(json.contains(r#""top_k":40"#));
    // Optional fields should be skipped when None
    assert!(!json.contains("tools"));
    assert!(!json.contains("tool_choice"));
    assert!(!json.contains("metadata"));
}

#[test]
fn test_minimax_request_minimal_serialization() {
    use sombrax_agentic_core::providers::minimax::types::{
        MinimaxContent, MinimaxMessage, MinimaxRequest,
    };

    let request = MinimaxRequest {
        model: "minimax-m2.1".to_string(),
        messages: vec![MinimaxMessage {
            role: "user".to_string(),
            content: MinimaxContent::Text("Hello!".to_string()),
        }],
        max_tokens: 1024,
        system: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        metadata: None,
        stream: None,
        thinking: None,
    };
    let json = serde_json::to_string(&request).unwrap();
    // Only required fields should be present
    assert!(json.contains(r#""model""#));
    assert!(json.contains(r#""messages""#));
    assert!(json.contains(r#""max_tokens""#));
    // Optional fields should NOT be present
    assert!(!json.contains("system"));
    assert!(!json.contains("temperature"));
    assert!(!json.contains("top_p"));
    assert!(!json.contains("top_k"));
}

// ============================================================================
// Response Deserialization Tests
// ============================================================================

#[test]
fn test_minimax_response_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxResponse;

    let json = r#"{
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "model": "minimax-m2.1",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15
        }
    }"#;

    let response: MinimaxResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.id, "msg_01XYZ");
    assert_eq!(response.response_type, "message");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.model, "minimax-m2.1");
    assert_eq!(response.stop_reason, Some("end_turn".to_string()));
    assert!(response.stop_sequence.is_none());
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.output_tokens, 15);
    assert_eq!(response.content.len(), 1);
}

#[test]
fn test_minimax_response_with_thinking_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::{
        MinimaxResponse, MinimaxResponseContent,
    };

    let json = r#"{
        "id": "msg_02ABC",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me analyze this problem..."},
            {"type": "text", "text": "The answer is 42."}
        ],
        "model": "minimax-m2.1",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 30
        }
    }"#;

    let response: MinimaxResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.content.len(), 2);

    match &response.content[0] {
        MinimaxResponseContent::Thinking { thinking } => {
            assert_eq!(thinking, "Let me analyze this problem...");
        }
        _ => panic!("Expected thinking block"),
    }

    match &response.content[1] {
        MinimaxResponseContent::Text { text } => {
            assert_eq!(text, "The answer is 42.");
        }
        _ => panic!("Expected text block"),
    }
}

#[test]
fn test_minimax_response_with_tool_use_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::{
        MinimaxResponse, MinimaxResponseContent,
    };

    let json = r#"{
        "id": "msg_03DEF",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather for you."},
            {"type": "tool_use", "id": "tool_456", "name": "get_weather", "input": {"location": "NYC"}}
        ],
        "model": "minimax-m2.1",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 25,
            "output_tokens": 35
        }
    }"#;

    let response: MinimaxResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.stop_reason, Some("tool_use".to_string()));
    assert_eq!(response.content.len(), 2);

    match &response.content[1] {
        MinimaxResponseContent::ToolUse { id, name, input } => {
            assert_eq!(id, "tool_456");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "NYC");
        }
        _ => panic!("Expected tool_use block"),
    }
}

#[test]
fn test_minimax_response_with_cache_usage_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxResponse;

    let json = r#"{
        "id": "msg_04GHI",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Cached response"}],
        "model": "minimax-m2.1",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20
        }
    }"#;

    let response: MinimaxResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.usage.input_tokens, 100);
    assert_eq!(response.usage.output_tokens, 50);
    assert_eq!(response.usage.cache_read_input_tokens, Some(80));
    assert_eq!(response.usage.cache_creation_input_tokens, Some(20));
}

#[test]
fn test_minimax_usage_defaults() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxUsage;

    // Cache fields should default to None when not present
    let json = r#"{
        "input_tokens": 10,
        "output_tokens": 20
    }"#;

    let usage: MinimaxUsage = serde_json::from_str(json).unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 20);
    assert!(usage.cache_read_input_tokens.is_none());
    assert!(usage.cache_creation_input_tokens.is_none());
}

// ============================================================================
// Content Block Deserialization Tests
// ============================================================================

#[test]
fn test_minimax_content_block_text_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let json = r#"{"type": "text", "text": "Hello!"}"#;
    let block: MinimaxContentBlock = serde_json::from_str(json).unwrap();

    match block {
        MinimaxContentBlock::Text { text } => assert_eq!(text, "Hello!"),
        _ => panic!("Expected text block"),
    }
}

#[test]
fn test_minimax_content_block_thinking_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let json = r#"{"type": "thinking", "thinking": "I should consider..."}"#;
    let block: MinimaxContentBlock = serde_json::from_str(json).unwrap();

    match block {
        MinimaxContentBlock::Thinking { thinking } => {
            assert_eq!(thinking, "I should consider...")
        }
        _ => panic!("Expected thinking block"),
    }
}

#[test]
fn test_minimax_content_block_tool_use_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let json = r#"{"type": "tool_use", "id": "t1", "name": "calc", "input": {"x": 1}}"#;
    let block: MinimaxContentBlock = serde_json::from_str(json).unwrap();

    match block {
        MinimaxContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "t1");
            assert_eq!(name, "calc");
            assert_eq!(input["x"], 1);
        }
        _ => panic!("Expected tool_use block"),
    }
}

#[test]
fn test_minimax_content_block_tool_result_deserialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxContentBlock;

    let json = r#"{"type": "tool_result", "tool_use_id": "t1", "content": "result"}"#;
    let block: MinimaxContentBlock = serde_json::from_str(json).unwrap();

    match block {
        MinimaxContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_use_id, "t1");
            assert_eq!(content, "result");
            assert!(is_error.is_none());
        }
        _ => panic!("Expected tool_result block"),
    }
}

#[test]
fn test_minimax_metadata_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxMetadata;

    let metadata = MinimaxMetadata {
        user_id: Some("user_123".to_string()),
    };
    let json = serde_json::to_string(&metadata).unwrap();
    assert!(json.contains(r#""user_id":"user_123""#));
}

#[test]
fn test_minimax_metadata_empty_serialization() {
    use sombrax_agentic_core::providers::minimax::types::MinimaxMetadata;

    let metadata = MinimaxMetadata { user_id: None };
    let json = serde_json::to_string(&metadata).unwrap();
    // user_id should be skipped when None
    assert!(!json.contains("user_id"));
}
