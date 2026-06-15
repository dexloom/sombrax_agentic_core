//! Mocked integration tests for MiniMax provider

use serde_json::json;
use sombrax_agentic_core::providers::zai::client::{
    CompletionRequest, Message, ToolCall, ToolDefinition,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Convert a (non-streaming) response body into the SSE event stream the MiniMax
/// client now consumes. MiniMax speaks the same Anthropic-style event frames
/// (`message_start` → `content_block_*` → `message_delta` → `message_stop`), so
/// each mock keeps its single-JSON spec and this re-emits it as the stream the
/// parser reassembles — keeping the tests' assertions unchanged.
fn sse_body(resp: &serde_json::Value) -> String {
    let usage = resp.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut start_usage = json!({ "input_tokens": input_tokens, "output_tokens": 0 });
    if let Some(v) = usage.get("cache_read_input_tokens") {
        start_usage["cache_read_input_tokens"] = v.clone();
    }
    if let Some(v) = usage.get("cache_creation_input_tokens") {
        start_usage["cache_creation_input_tokens"] = v.clone();
    }

    let mut events: Vec<serde_json::Value> = vec![json!({
        "type": "message_start",
        "message": {
            "id": resp.get("id").cloned().unwrap_or_else(|| json!("msg_test")),
            "type": "message",
            "role": resp.get("role").cloned().unwrap_or_else(|| json!("assistant")),
            "model": resp.get("model").cloned().unwrap_or_else(|| json!("")),
            "usage": start_usage,
        }
    })];

    if let Some(blocks) = resp.get("content").and_then(|c| c.as_array()) {
        for (i, block) in blocks.iter().enumerate() {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    events.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}}));
                    events.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text": block.get("text").cloned().unwrap_or_else(|| json!(""))}}));
                    events.push(json!({"type":"content_block_stop","index":i}));
                }
                Some("thinking") => {
                    events.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"thinking","thinking":""}}));
                    events.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"thinking_delta","thinking": block.get("thinking").cloned().unwrap_or_else(|| json!(""))}}));
                    events.push(json!({"type":"content_block_stop","index":i}));
                }
                Some("tool_use") => {
                    events.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"tool_use","id": block.get("id").cloned().unwrap_or_else(|| json!("")),"name": block.get("name").cloned().unwrap_or_else(|| json!(""))}}));
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    events.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json": serde_json::to_string(&input).unwrap()}}));
                    events.push(json!({"type":"content_block_stop","index":i}));
                }
                _ => {}
            }
        }
    }

    events.push(json!({
        "type": "message_delta",
        "delta": { "stop_reason": resp.get("stop_reason").cloned().unwrap_or(serde_json::Value::Null) },
        "usage": { "output_tokens": output_tokens }
    }));
    events.push(json!({ "type": "message_stop" }));

    events
        .iter()
        .map(|e| format!("data: {}\n\n", serde_json::to_string(e).unwrap()))
        .collect()
}

/// Build a 200 SSE `ResponseTemplate` from a MiniMax/Anthropic-style response body.
fn sse_response(body: serde_json::Value) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(sse_body(&body))
}

/// Test MiniMax completion request and response parsing
#[tokio::test]
async fn test_minimax_completion_request_response() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("Content-Type", "application/json"))
        .respond_with(sse_response(json!({
            "id": "msg_completion",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! I'm MiniMax M2.1"}],
            "model": "minimax-m2.1",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .temperature(0.7)
        .build();

    let model = client.completion_model("minimax-m2.1");

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
        temperature: None,
        max_tokens: Some(1024),
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert_eq!(response.message.role, "assistant");
    assert_eq!(response.message.content, "Hello! I'm MiniMax M2.1");
    assert_eq!(response.usage.prompt_tokens, 12);
    assert_eq!(response.usage.completion_tokens, 8);
    assert_eq!(response.usage.total_tokens, 20);
}

/// Test MiniMax completion with sampling parameters
#[tokio::test]
async fn test_minimax_with_sampling_params() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-api-key"))
        .respond_with(sse_response(json!({
            "id": "msg_sampling",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Response with sampling params"}],
            "model": "minimax-m2.1",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .temperature(0.7)
        .top_p(0.9)
        .top_k(40)
        .max_tokens(2048)
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Test".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert_eq!(response.message.content, "Response with sampling params");
}

/// Test MiniMax completion with tool calls
#[tokio::test]
async fn test_minimax_completion_with_tool_calls() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(json!({
            "id": "msg_tools",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "id": "call_weather", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "model": "minimax-m2.1",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {"input_tokens": 30, "output_tokens": 25}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "What's the weather in NYC?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        }],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert!(response.message.tool_calls.is_some());
    let tool_calls = response.message.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_weather");
    assert_eq!(tool_calls[0].name, "get_weather");
}

/// Test MiniMax completion with thinking/reasoning content
#[tokio::test]
async fn test_minimax_completion_with_thinking() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(json!({
            "id": "msg_thinking",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "The user wants to know about quantum physics. I should explain it simply."},
                {"type": "text", "text": "Quantum physics is the study of matter and energy at the smallest scales."}
            ],
            "model": "minimax-m2.1",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 15, "output_tokens": 40}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Explain quantum physics".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert!(response.reasoning_content.is_some());
    assert!(response
        .reasoning_content
        .unwrap()
        .contains("quantum physics"));
    assert!(response.message.content.contains("Quantum physics"));
}

/// Test MiniMax response with cache usage metrics
#[tokio::test]
async fn test_minimax_completion_with_cache_usage() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(json!({
            "id": "msg_cached",
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
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Test cache".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert_eq!(response.message.content, "Cached response");
    assert_eq!(response.usage.prompt_tokens, 100);
    assert_eq!(response.usage.completion_tokens, 50);
}

/// Test MiniMax rate limit error handling
#[tokio::test]
async fn test_minimax_rate_limit_error() {
    use sombrax_agentic_core::providers::error::ProviderError;

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let result = model.completion(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        sombrax_agentic_core::providers::error::CompletionError::Provider(
            ProviderError::RateLimited { .. },
        ) => {}
        _ => panic!("Expected rate limit error, got: {:?}", err),
    }
}

/// Test MiniMax authentication error handling
#[tokio::test]
async fn test_minimax_auth_error() {
    use sombrax_agentic_core::providers::error::ProviderError;

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "Invalid API key", "type": "authentication_error"}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("invalid-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let result = model.completion(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        sombrax_agentic_core::providers::error::CompletionError::Provider(
            ProviderError::Authentication(_),
        ) => {}
        _ => panic!("Expected authentication error, got: {:?}", err),
    }
}

/// Test MiniMax server error handling
#[tokio::test]
async fn test_minimax_server_error() {
    use sombrax_agentic_core::providers::error::ProviderError;

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": {"message": "Internal server error", "type": "server_error"}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    let request = CompletionRequest {
        preamble: None,
        messages: vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let result = model.completion(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        sombrax_agentic_core::providers::error::CompletionError::Provider(
            ProviderError::Http { status, .. },
        ) => {
            assert_eq!(status, 500);
        }
        _ => panic!("Expected HTTP 500 error, got: {:?}", err),
    }
}

/// Test that multiple tool results are merged into a single user message.
/// The Anthropic-compatible API requires all tool_result blocks for a multi-tool-call
/// assistant message to appear in one user message immediately after.
#[tokio::test]
async fn test_minimax_merges_consecutive_tool_results() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(sse_response(json!({
            "id": "msg_after_tools",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Based on the results..."}],
            "model": "minimax-m2.1",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 50, "output_tokens": 10}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = sombrax_agentic_core::providers::MinimaxClientBuilder::new("test-api-key")
        .base_url(&mock_server.uri())
        .build();

    let model = client.completion_model("minimax-m2.1");

    // Simulate: user asks, assistant calls 2 tools, then 2 separate tool result messages
    let request = CompletionRequest {
        preamble: None,
        messages: vec![
            Message {
                role: "user".to_string(),
                content: "Find files".to_string(),
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            },
            Message {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: Some(vec![
                    ToolCall {
                        id: "call_1".to_string(),
                        name: "glob".to_string(),
                        arguments: r#"{"pattern":"*.sol"}"#.to_string(),
                    },
                    ToolCall {
                        id: "call_2".to_string(),
                        name: "glob".to_string(),
                        arguments: r#"{"pattern":"*.rs"}"#.to_string(),
                    },
                ]),
                tool_call_id: None,
                reasoning: None,
            },
            Message {
                role: "user".to_string(),
                content: "file1.sol".to_string(),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning: None,
            },
            Message {
                role: "user".to_string(),
                content: "file1.rs".to_string(),
                tool_calls: None,
                tool_call_id: Some("call_2".to_string()),
                reasoning: None,
            },
        ],
        tools: vec![ToolDefinition {
            name: "glob".to_string(),
            description: "Find files".to_string(),
            parameters: json!({"type": "object", "properties": {"pattern": {"type": "string"}}}),
        }],
        temperature: None,
        max_tokens: None,
        additional_params: None,
        cache: Default::default(),
    };

    let response = model.completion(request).await.unwrap();
    assert_eq!(response.message.content, "Based on the results...");

    // Verify the request body has merged tool results
    let requests = mock_server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let msgs = body["messages"].as_array().unwrap();

    // Should be 3 messages: user, assistant (2 tool_use), user (2 tool_results merged)
    assert_eq!(
        msgs.len(),
        3,
        "Expected 3 messages (user, assistant, merged tool results), got {}",
        msgs.len()
    );

    // The last message should have 2 tool_result blocks
    let last_msg = &msgs[2];
    assert_eq!(last_msg["role"], "user");
    let content_blocks = last_msg["content"].as_array().unwrap();
    assert_eq!(
        content_blocks.len(),
        2,
        "Expected 2 tool_result blocks in merged message"
    );
    assert_eq!(content_blocks[0]["type"], "tool_result");
    assert_eq!(content_blocks[0]["tool_use_id"], "call_1");
    assert_eq!(content_blocks[1]["type"], "tool_result");
    assert_eq!(content_blocks[1]["tool_use_id"], "call_2");
}
