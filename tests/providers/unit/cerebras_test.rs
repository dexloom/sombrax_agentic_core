//! Unit tests for Cerebras provider

use sombrax_agentic_core::providers::cerebras::{
    extract_tool_result_content, CerebrasClientBuilder,
};

#[test]
fn test_cerebras_client_builder_defaults() {
    let client = CerebrasClientBuilder::new("test-api-key").build();
    let model = client.completion_model("llama-3.3-70b");
    assert_eq!(model.model_id(), "llama-3.3-70b");
    assert_eq!(model.provider(), "cerebras");
}

#[test]
fn test_cerebras_client_builder_with_temperature() {
    let client = CerebrasClientBuilder::new("test-api-key")
        .temperature(0.5)
        .build();
    let model = client.completion_model("llama-3.3-70b");
    assert_eq!(model.model_id(), "llama-3.3-70b");
}

#[test]
fn test_cerebras_client_builder_temperature_clamping() {
    // Cerebras temperature is clamped to 0.0-1.0
    let _client = CerebrasClientBuilder::new("test-api-key")
        .temperature(1.5) // Should be clamped to 1.0
        .build();
}

#[test]
fn test_cerebras_client_builder_with_top_p() {
    let _client = CerebrasClientBuilder::new("test-api-key")
        .top_p(0.9)
        .build();
}

#[test]
fn test_cerebras_client_builder_with_top_k() {
    let _client = CerebrasClientBuilder::new("test-api-key").top_k(40).build();
}

#[test]
fn test_cerebras_client_builder_with_max_tokens() {
    let _client = CerebrasClientBuilder::new("test-api-key")
        .max_tokens(4096)
        .build();
}

#[test]
fn test_cerebras_client_builder_with_base_url() {
    let _client = CerebrasClientBuilder::new("test-api-key")
        .base_url("https://custom.cerebras.endpoint/v1")
        .build();
}

#[test]
fn test_cerebras_client_builder_full_config() {
    let client = CerebrasClientBuilder::new("test-api-key")
        .base_url("https://custom.cerebras.endpoint/v1")
        .temperature(0.6)
        .top_p(0.85)
        .top_k(30)
        .max_tokens(2048)
        .build();

    let model = client.completion_model("llama-3.1-8b");
    assert_eq!(model.model_id(), "llama-3.1-8b");
}

#[test]
fn test_extract_tool_result_content_simple() {
    let content = "Hello, world!";
    let result = extract_tool_result_content(content);
    assert_eq!(result, "Hello, world!");
}

#[test]
fn test_extract_tool_result_content_json() {
    let content = r#"{"result": "success", "data": [1, 2, 3]}"#;
    let result = extract_tool_result_content(content);
    assert_eq!(result, content);
}

#[test]
fn test_extract_tool_result_content_empty() {
    let content = "";
    let result = extract_tool_result_content(content);
    assert_eq!(result, "");
}

#[test]
fn test_extract_tool_result_content_multiline() {
    let content = "Line 1\nLine 2\nLine 3";
    let result = extract_tool_result_content(content);
    assert_eq!(result, "Line 1\nLine 2\nLine 3");
}

#[test]
fn test_cerebras_model_clone() {
    let client = CerebrasClientBuilder::new("test-api-key").build();
    let model1 = client.completion_model("llama-3.3-70b");
    let model2 = model1.clone();
    assert_eq!(model1.model_id(), model2.model_id());
}
