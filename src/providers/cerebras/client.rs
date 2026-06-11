//! Cerebras client and completion model

use std::env;
use std::sync::Arc;

use reqwest::Client;
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default Cerebras API base URL
const DEFAULT_BASE_URL: &str = "https://api.cerebras.ai/v1";

/// Default max tokens for Cerebras
const DEFAULT_MAX_TOKENS: u64 = 8192;

/// Cerebras client configuration
#[derive(Clone)]
pub struct CerebrasClient {
    inner: Arc<CerebrasClientInner>,
}

struct CerebrasClientInner {
    http_client: Client,
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
}

impl CerebrasClient {
    /// Create a new Cerebras client from environment variable
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("CEREBRAS_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("CEREBRAS_API_KEY".to_string()))?;

        Ok(CerebrasClientBuilder::new(&api_key).build())
    }

    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> CerebrasCompletionModel {
        CerebrasCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for Cerebras client configuration
pub struct CerebrasClientBuilder {
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
}

impl CerebrasClientBuilder {
    /// Create a new builder with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
        }
    }

    /// Set custom base URL
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set temperature (clamped to 0.0-1.0)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp.clamp(0.0, 1.0));
        self
    }

    /// Set top_p sampling parameter
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set top_k sampling parameter
    pub fn top_k(mut self, k: u64) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Build the client
    pub fn build(self) -> CerebrasClient {
        CerebrasClient {
            inner: Arc::new(CerebrasClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                base_url: self.base_url,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
            }),
        }
    }
}

/// Cerebras completion model
#[derive(Clone)]
pub struct CerebrasCompletionModel {
    client: CerebrasClient,
    model_id: String,
}

/// Extract tool result content as a simple string
/// Cerebras requires content to be a string, not an array
pub fn extract_tool_result_content(content: &str) -> String {
    // Content is already a string in our simplified model
    content.to_string()
}

impl CerebrasCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "cerebras"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "cerebras"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<CerebrasResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Build messages
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(CerebrasMessage {
                role: "system".to_string(),
                content: preamble.clone(),
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.push(CerebrasMessage {
                role: msg.role.clone(),
                // Cerebras requires content as simple string
                content: extract_tool_result_content(&msg.content),
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| CerebrasToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: CerebrasFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: msg.tool_call_id.clone(),
                reasoning: msg.reasoning.clone(),
            });
        }

        // Build tools
        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .tools
                    .iter()
                    .map(|t| CerebrasTool {
                        tool_type: "function".to_string(),
                        function: CerebrasFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        let cerebras_request = CerebrasRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            tools,
            tool_choice: None,
        };

        let url = format!("{}/chat/completions", inner.base_url);

        let response = inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", inner.api_key))
            .header("Content-Type", "application/json")
            .json(&cerebras_request)
            .send()
            .instrument(info_span!("cerebras_http_request"))
            .await
            .map_err(ProviderError::Request)?;

        let status = response.status();
        let response_text = response.text().await.map_err(ProviderError::Request)?;

        if !status.is_success() {
            return Err(CompletionError::Provider(ProviderError::Http {
                status: status.as_u16(),
                message: response_text,
            }));
        }

        let cerebras_response: CerebrasResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                tracing::error!(
                    "Failed to deserialize Cerebras response: {}\nRaw response: {}",
                    e,
                    &response_text[..response_text.len().min(2000)]
                );
                ProviderError::InvalidResponse(format!(
                    "JSON deserialization failed: {}. Response preview: {}",
                    e,
                    &response_text[..response_text.len().min(500)]
                ))
            })?;

        // Extract first choice
        let choice = cerebras_response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        // Convert tool calls
        let tool_calls = choice.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                })
                .collect()
        });

        let message = Message {
            role: choice.message.role.clone(),
            content: choice.message.content.clone(),
            tool_calls,
            tool_call_id: None,
            reasoning: choice.message.reasoning.clone(),
        };

        // Extract cache tokens if available
        let cache_read_tokens = cerebras_response
            .usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: cerebras_response.usage.prompt_tokens,
                completion_tokens: cerebras_response.usage.completion_tokens,
                total_tokens: cerebras_response.usage.total_tokens,
                cache_read_tokens,
                cache_creation_tokens: 0,
            },
            raw: cerebras_response.clone(),
            reasoning_content: None,
            finish_reason: choice.finish_reason.clone(),
        })
    }
}
