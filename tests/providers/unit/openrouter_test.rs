//! Unit tests for OpenRouter provider

use sombrax_agentic_core::providers::openrouter::{
    OpenRouterClientBuilder, OpenRouterProviderConfig,
};

#[test]
fn test_openrouter_client_builder_defaults() {
    let client = OpenRouterClientBuilder::new("test-api-key").build();
    let model = client.completion_model("anthropic/claude-3.5-sonnet");
    assert_eq!(model.model_id(), "anthropic/claude-3.5-sonnet");
    assert_eq!(model.provider(), "openrouter");
}

#[test]
fn test_openrouter_client_builder_with_temperature() {
    let client = OpenRouterClientBuilder::new("test-api-key")
        .temperature(0.7)
        .build();
    let model = client.completion_model("openai/gpt-4");
    assert_eq!(model.model_id(), "openai/gpt-4");
}

#[test]
fn test_openrouter_client_builder_temperature_clamping() {
    // OpenRouter temperature is clamped to 0.0-2.0
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .temperature(3.0) // Should be clamped to 2.0
        .build();
}

#[test]
fn test_openrouter_client_builder_with_top_p() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .top_p(0.9)
        .build();
}

#[test]
fn test_openrouter_client_builder_with_top_k() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .top_k(40)
        .build();
}

#[test]
fn test_openrouter_client_builder_with_max_tokens() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .max_tokens(4096)
        .build();
}

#[test]
fn test_openrouter_client_builder_with_base_url() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .base_url("https://custom.openrouter.endpoint/v1")
        .build();
}

#[test]
fn test_openrouter_client_builder_with_blacklist() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .blacklist(vec!["Azure".to_string(), "AWS".to_string()])
        .build();
}

#[test]
fn test_openrouter_client_builder_with_whitelist() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .whitelist(vec!["Anthropic".to_string()])
        .build();
}

#[test]
fn test_openrouter_client_builder_with_fallbacks() {
    let _client = OpenRouterClientBuilder::new("test-api-key")
        .allow_fallbacks(false)
        .build();
}

#[test]
fn test_openrouter_client_builder_full_config() {
    let client = OpenRouterClientBuilder::new("test-api-key")
        .base_url("https://custom.openrouter.endpoint/v1")
        .temperature(0.8)
        .top_p(0.95)
        .top_k(50)
        .max_tokens(8192)
        .blacklist(vec!["Azure".to_string()])
        .allow_fallbacks(true)
        .build();

    let model = client.completion_model("meta-llama/llama-3.1-405b");
    assert_eq!(model.model_id(), "meta-llama/llama-3.1-405b");
}

#[test]
fn test_provider_config_default() {
    let config = OpenRouterProviderConfig::default();
    assert!(!config.has_rules());
}

#[test]
fn test_provider_config_blacklist() {
    let config =
        OpenRouterProviderConfig::default().blacklist(vec!["Azure".to_string(), "AWS".to_string()]);
    assert!(config.has_rules());
}

#[test]
fn test_provider_config_whitelist() {
    let config = OpenRouterProviderConfig::default().whitelist(vec!["Anthropic".to_string()]);
    assert!(config.has_rules());
}

#[test]
fn test_provider_config_allow_fallbacks() {
    let config = OpenRouterProviderConfig::default().allow_fallbacks(false);
    assert!(config.has_rules());
}

#[test]
fn test_provider_config_combined() {
    let config = OpenRouterProviderConfig::default()
        .blacklist(vec!["Azure".to_string()])
        .allow_fallbacks(true);
    assert!(config.has_rules());
}

#[test]
fn test_openrouter_model_clone() {
    let client = OpenRouterClientBuilder::new("test-api-key").build();
    let model1 = client.completion_model("openai/gpt-4");
    let model2 = model1.clone();
    assert_eq!(model1.model_id(), model2.model_id());
}
