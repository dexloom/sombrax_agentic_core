//! ZAI client and completion model

use std::env;
use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;

/// Default ZAI API base URL
const DEFAULT_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";

/// Default max tokens for ZAI
const DEFAULT_MAX_TOKENS: u64 = 8192;

/// ZAI client configuration
#[derive(Clone)]
pub struct ZaiClient {
    inner: Arc<ZaiClientInner>,
}

struct ZaiClientInner {
    http_client: Client,
    api_key: String,
    base_url: String,
    enable_thinking: bool,
    #[allow(dead_code)] // Reserved for future ZAI thinking budget support
    thinking_budget_tokens: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    parallel_tool_calls: Option<bool>,
}

impl ZaiClient {
    /// Create a new ZAI client from environment variable
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("ZAI_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("ZAI_API_KEY".to_string()))?;

        Ok(ZaiClientBuilder::new(&api_key).build())
    }

    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> ZaiCompletionModel {
        ZaiCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for ZAI client configuration
pub struct ZaiClientBuilder {
    api_key: String,
    base_url: String,
    enable_thinking: bool,
    thinking_budget_tokens: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    parallel_tool_calls: Option<bool>,
}

impl ZaiClientBuilder {
    /// Create a new builder with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            enable_thinking: true,
            thinking_budget_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            parallel_tool_calls: Some(true),
        }
    }

    /// Set custom base URL
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Enable/disable thinking mode (default: true)
    pub fn enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = enabled;
        self
    }

    /// Set thinking budget in tokens (reserved for future ZAI support)
    pub fn thinking_budget_tokens(mut self, tokens: u64) -> Self {
        self.thinking_budget_tokens = Some(tokens);
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

    /// Enable/disable parallel tool calls (default: true)
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Build the client
    pub fn build(self) -> ZaiClient {
        ZaiClient {
            inner: Arc::new(ZaiClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                base_url: self.base_url,
                enable_thinking: self.enable_thinking,
                thinking_budget_tokens: self.thinking_budget_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
                parallel_tool_calls: self.parallel_tool_calls,
            }),
        }
    }
}

/// ZAI completion model
#[derive(Clone)]
pub struct ZaiCompletionModel {
    client: ZaiClient,
    model_id: String,
}

/// Completion request (simplified from rig-core)
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// System prompt / preamble.
    pub preamble: Option<String>,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tool definitions.
    pub tools: Vec<ToolDefinition>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens in the response.
    pub max_tokens: Option<u64>,
    /// Provider-specific parameters.
    pub additional_params: Option<serde_json::Value>,
    /// Provider-independent prompt-cache hints (translated to wire markers by
    /// clients with an explicit cache protocol; ignored by implicit-cache
    /// clients). This is an intermediate, non-serialized request type, so the
    /// field never appears on any provider's wire body.
    pub cache: crate::provider::CacheHints,
}

/// Message in a completion request
#[derive(Debug, Clone)]
pub struct Message {
    /// Message role.
    pub role: String,
    /// Message content.
    pub content: String,
    /// Optional tool calls.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional tool call id for tool results.
    pub tool_call_id: Option<String>,
    /// Optional reasoning/thinking content (for models that return it as separate field).
    pub reasoning: Option<String>,
}

impl Message {
    /// Create a user message
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }
    }

    /// Create a system message
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }
    }
}

/// Tool definition for LLM function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON schema for tool parameters.
    pub parameters: serde_json::Value,
}

/// Tool call from LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool call identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// Completion response
#[derive(Debug, Clone)]
pub struct CompletionResponse<R> {
    /// Assistant message.
    pub message: Message,
    /// Token usage.
    pub usage: Usage,
    /// Raw provider response.
    pub raw: R,
    /// Optional reasoning content.
    pub reasoning_content: Option<String>,
    /// The reason the model stopped generating.
    pub finish_reason: Option<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Prompt token count.
    pub prompt_tokens: u64,
    /// Completion token count.
    pub completion_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Cache read tokens (tokens served from cache).
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Cache creation tokens (tokens written to cache).
    #[serde(default)]
    pub cache_creation_tokens: u64,
}

impl ZaiCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "zai"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "zai"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<ZaiResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Build messages
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(ZaiMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            // Skip empty user messages (ZAI quirk)
            if msg.role == "user" && msg.content.is_empty() {
                continue;
            }

            messages.push(ZaiMessage {
                role: msg.role.clone(),
                content: Some(msg.content.clone()),
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| ZaiToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: ZaiFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: msg.tool_call_id.clone(),
                reasoning_content: None,
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
                    .map(|t| ZaiTool {
                        tool_type: "function".to_string(),
                        function: ZaiFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        // Build thinking config
        let thinking = if inner.enable_thinking {
            Some(ZaiThinkingConfig::enabled())
        } else {
            None
        };

        let zai_request = ZaiRequest {
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
            thinking,
            parallel_tool_calls: inner.parallel_tool_calls,
        };

        let url = format!("{}/chat/completions", inner.base_url);

        let response = inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", inner.api_key))
            .header("Content-Type", "application/json")
            .json(&zai_request)
            .send()
            .instrument(info_span!("zai_http_request"))
            .await
            .map_err(ProviderError::Request)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CompletionError::Provider(ProviderError::Http {
                status: status.as_u16(),
                message: error_text,
            }));
        }

        let zai_response: ZaiResponse = response.json().await.map_err(ProviderError::Request)?;

        // Extract first choice
        let choice = zai_response.choices.first().ok_or_else(|| {
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
            content: choice.message.content.clone().unwrap_or_default(),
            tool_calls,
            tool_call_id: None,
            reasoning: None,
        };

        // Extract cache tokens from various possible locations
        let cache_read_tokens = zai_response
            .usage
            .cached_tokens
            .or_else(|| {
                zai_response
                    .usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
            })
            .unwrap_or(0);

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: zai_response.usage.prompt_tokens,
                completion_tokens: zai_response.usage.completion_tokens,
                total_tokens: zai_response.usage.total_tokens,
                cache_read_tokens,
                cache_creation_tokens: 0, // ZAI doesn't report cache creation separately
            },
            raw: zai_response.clone(),
            reasoning_content: choice.message.reasoning_content.clone(),
            finish_reason: choice.finish_reason.clone(),
        })
    }
}
