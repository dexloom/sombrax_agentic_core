//! Unit tests for OpenAI provider

use sombrax_agentic_core::providers::openai::types::*;
use sombrax_agentic_core::providers::OpenAIClientBuilder;

#[test]
fn test_openai_client_builder() {
    let client = OpenAIClientBuilder::new("test-api-key")
        .temperature(0.7)
        .top_p(0.9)
        .max_tokens(4096)
        .frequency_penalty(0.5)
        .presence_penalty(0.5)
        .build();

    let model = client.completion_model("gpt-4o");
    assert_eq!(model.model_id(), "gpt-4o");
    assert_eq!(model.provider(), "openai");
}

#[test]
fn test_openai_client_builder_organization() {
    let client = OpenAIClientBuilder::new("test-api-key")
        .organization("org-123")
        .build();

    let model = client.completion_model("gpt-4");
    assert_eq!(model.model_id(), "gpt-4");
}

#[test]
fn test_openai_client_builder_base_url() {
    let client = OpenAIClientBuilder::new("test-api-key")
        .base_url("https://api.azure.com/openai")
        .build();

    let model = client.completion_model("gpt-35-turbo");
    assert_eq!(model.model_id(), "gpt-35-turbo");
}

#[test]
fn test_openai_client_builder_parallel_tool_calls() {
    let client = OpenAIClientBuilder::new("test-api-key")
        .parallel_tool_calls(false)
        .build();

    let model = client.completion_model("gpt-4o");
    assert_eq!(model.provider(), "openai");
}

#[test]
fn test_openai_message_serialization() {
    let message = OpenAIMessage {
        role: "user".to_string(),
        content: Some("Hello".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning: None,
    };

    let json = serde_json::to_string(&message).unwrap();
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("Hello"));
}

#[test]
fn test_openai_message_with_tool_calls() {
    let message = OpenAIMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(vec![OpenAIToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunctionCall {
                name: "search".to_string(),
                arguments: r#"{"query": "test"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
        name: None,
        reasoning: None,
    };

    let json = serde_json::to_string(&message).unwrap();
    assert!(json.contains("tool_calls"));
    assert!(json.contains("call_123"));
    assert!(json.contains("search"));
}

#[test]
fn test_openai_tool_definition() {
    let tool = OpenAITool {
        tool_type: "function".to_string(),
        function: OpenAIFunction {
            name: "get_weather".to_string(),
            description: Some("Get weather for a location".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })),
        },
    };

    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains("function"));
    assert!(json.contains("get_weather"));
}

#[test]
fn test_openai_tool_choice_string() {
    let choice = OpenAIToolChoice::String("auto".to_string());
    let json = serde_json::to_string(&choice).unwrap();
    assert_eq!(json, "\"auto\"");
}

#[test]
fn test_openai_tool_choice_object() {
    let choice = OpenAIToolChoice::Object {
        choice_type: "function".to_string(),
        function: OpenAIToolChoiceFunction {
            name: "search".to_string(),
        },
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(json.contains("function"));
    assert!(json.contains("search"));
}

#[test]
fn test_openai_request_serialization() {
    let request = OpenAIRequest {
        model: "gpt-4o".to_string(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        }],
        temperature: Some(0.7),
        max_tokens: Some(4096),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: Some(true),
        user: None,
        chat_template_kwargs: None,
        stream: None,
        stream_options: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("gpt-4o"));
    assert!(json.contains("temperature"));
    assert!(json.contains("parallel_tool_calls"));
}

#[test]
fn test_openai_response_deserialization() {
    let json = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }"#;

    let response: OpenAIResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.id, "chatcmpl-123");
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.usage.prompt_tokens, 10);
    assert_eq!(response.usage.total_tokens, 15);
}

#[test]
fn test_openai_response_with_tool_calls() {
    let json = r#"{
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\": \"test\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}
    }"#;

    let response: OpenAIResponse = serde_json::from_str(json).unwrap();
    assert_eq!(
        response.choices[0].finish_reason,
        Some("tool_calls".to_string())
    );
    let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "search");
}

#[test]
fn test_openai_response_with_system_fingerprint() {
    let json = r#"{
        "id": "chatcmpl-789",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "system_fingerprint": "fp_abc123",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }"#;

    let response: OpenAIResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.system_fingerprint, Some("fp_abc123".to_string()));
}
