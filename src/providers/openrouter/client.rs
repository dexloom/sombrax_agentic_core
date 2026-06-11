//! OpenRouter client and completion model

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use regex::Regex;
use reqwest::Client;
use tracing::{info_span, instrument, warn, Instrument};

use super::routing::OpenRouterProviderConfig;
use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default OpenRouter API base URL
const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default max tokens for OpenRouter
const DEFAULT_MAX_TOKENS: u64 = 16384;

/// OpenRouter client configuration
#[derive(Clone)]
pub struct OpenRouterClient {
    inner: Arc<OpenRouterClientInner>,
}

struct OpenRouterClientInner {
    http_client: Client,
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    provider_config: Option<OpenRouterProviderConfig>,
}

impl OpenRouterClient {
    /// Create a new OpenRouter client from environment variable
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("OPENROUTER_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("OPENROUTER_API_KEY".to_string()))?;

        Ok(OpenRouterClientBuilder::new(&api_key).build())
    }

    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> OpenRouterCompletionModel {
        OpenRouterCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for OpenRouter client configuration
pub struct OpenRouterClientBuilder {
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    provider_config: Option<OpenRouterProviderConfig>,
}

impl OpenRouterClientBuilder {
    /// Create a new builder with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            provider_config: None,
        }
    }

    /// Set custom base URL
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set temperature (clamped to 0.0-2.0 for OpenRouter)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
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

    /// Blacklist specific providers
    pub fn blacklist(mut self, providers: Vec<String>) -> Self {
        let config = self.provider_config.take().unwrap_or_default();
        self.provider_config = Some(config.blacklist(providers));
        self
    }

    /// Whitelist specific providers (only use these)
    pub fn whitelist(mut self, providers: Vec<String>) -> Self {
        let config = self.provider_config.take().unwrap_or_default();
        self.provider_config = Some(config.whitelist(providers));
        self
    }

    /// Enable/disable fallback routing
    pub fn allow_fallbacks(mut self, enabled: bool) -> Self {
        let config = self.provider_config.take().unwrap_or_default();
        self.provider_config = Some(config.allow_fallbacks(enabled));
        self
    }

    /// Build the client
    pub fn build(self) -> OpenRouterClient {
        OpenRouterClient {
            inner: Arc::new(OpenRouterClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                base_url: self.base_url,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
                provider_config: self.provider_config,
            }),
        }
    }
}

/// OpenRouter completion model
#[derive(Clone)]
pub struct OpenRouterCompletionModel {
    client: OpenRouterClient,
    model_id: String,
}

/// Atomic counter for generating Minimax tool call IDs
static MINIMAX_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Parse Minimax XML tool calls from reasoning content
/// Some models (e.g., Minimax) return tool calls in XML format within reasoning
pub fn parse_minimax_xml_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();

    // Match <minimax:tool_call> blocks
    let tool_call_regex = Regex::new(
        r#"<minimax:tool_call>\s*<invoke\s+name="([^"]+)">(.*?)</invoke>\s*</minimax:tool_call>"#,
    )
    .unwrap();

    let param_regex = Regex::new(r#"<parameter\s+name="([^"]+)">([^<]*)</parameter>"#).unwrap();

    for cap in tool_call_regex.captures_iter(content) {
        let name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let params_block = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        let mut params = serde_json::Map::new();
        for param_cap in param_regex.captures_iter(params_block) {
            let param_name = param_cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let param_value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");
            params.insert(
                param_name.to_string(),
                serde_json::Value::String(param_value.to_string()),
            );
        }

        tool_calls.push(ToolCall {
            id: format!(
                "call_minimax_{}",
                MINIMAX_CALL_COUNTER.fetch_add(1, Ordering::SeqCst)
            ),
            name: name.to_string(),
            arguments: serde_json::to_string(&params).unwrap_or_default(),
        });
    }

    tool_calls
}

/// Extract first valid JSON object from a string with potential duplicates.
///
/// Properly handles braces inside JSON string values (e.g., code snippets
/// containing `{` and `}` characters) by tracking quoted regions and
/// backslash escapes.
pub fn extract_first_json_object(s: &str) -> Option<String> {
    let mut depth = 0;
    let mut start = None;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if in_string {
            match c {
                '\\' => escape_next = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match c {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start_idx) = start {
                        return Some(s[start_idx..=i].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

impl OpenRouterCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "openrouter"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "openrouter"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<OpenRouterResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Build messages
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(OpenRouterMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                ..Default::default()
            });
        }

        // Convert messages.
        //
        // Echo `reasoning` back as `reasoning_content` on assistant turns.
        // Some providers (e.g. Moonshot AI) reject the request with HTTP 400
        // — `thinking is enabled but reasoning_content is missing in
        // assistant tool call message at index N` — when an assistant
        // message in history carries `tool_calls` but no `reasoning_content`
        // and the request has thinking enabled. Carry it through whenever
        // we have it; providers that don't care just ignore the field.
        for msg in &request.messages {
            messages.push(OpenRouterMessage {
                role: msg.role.clone(),
                content: Some(msg.content.clone()),
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| OpenRouterToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: OpenRouterFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: msg.tool_call_id.clone(),
                reasoning_content: msg.reasoning.clone(),
                ..Default::default()
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
                    .map(|t| OpenRouterTool {
                        tool_type: "function".to_string(),
                        function: OpenRouterFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        // Include provider routing if configured
        let provider = inner
            .provider_config
            .as_ref()
            .filter(|c| c.has_rules())
            .cloned();

        let openrouter_request = OpenRouterRequest {
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
            provider,
        };

        let url = format!("{}/chat/completions", inner.base_url);

        let response = inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", inner.api_key))
            .header("Content-Type", "application/json")
            .json(&openrouter_request)
            .send()
            .instrument(info_span!("openrouter_http_request"))
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

        let openrouter_response: OpenRouterResponse =
            response.json().await.map_err(ProviderError::Request)?;

        // Check for error in 200 response body
        if let Some(error) = &openrouter_response.error {
            return Err(CompletionError::Provider(ProviderError::InvalidResponse(
                error
                    .message
                    .clone()
                    .unwrap_or_else(|| "Unknown error in response body".to_string()),
            )));
        }

        // Extract first choice
        let choice = openrouter_response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        // Check for choice-level error
        if let Some(error) = &choice.error {
            warn!("Choice-level error: {:?}", error);
        }

        // Get tool calls from response, checking for XML format in reasoning
        let mut tool_calls: Option<Vec<ToolCall>> =
            choice.message.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|tc| {
                        // Handle potential duplicate JSON in arguments
                        let arguments = extract_first_json_object(&tc.function.arguments)
                            .unwrap_or_else(|| tc.function.arguments.clone());

                        ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments,
                        }
                    })
                    .collect()
            });

        // Check for Minimax XML tool calls in reasoning content
        let reasoning_content = choice
            .message
            .reasoning
            .clone()
            .or_else(|| choice.message.reasoning_content.clone());

        if tool_calls.is_none()
            || tool_calls
                .as_ref()
                .map(|v: &Vec<ToolCall>| v.is_empty())
                .unwrap_or(true)
        {
            if let Some(reasoning) = &reasoning_content {
                let xml_tool_calls = parse_minimax_xml_tool_calls(reasoning);
                if !xml_tool_calls.is_empty() {
                    tool_calls = Some(xml_tool_calls);
                }
            }
        }

        let message = Message {
            role: choice.message.role.clone(),
            content: choice.message.content.clone().unwrap_or_default(),
            tool_calls,
            tool_call_id: None,
            // Carry reasoning forward on the message itself so the next
            // request can echo it back as `reasoning_content`. Required by
            // providers like Moonshot AI when thinking is enabled.
            reasoning: reasoning_content.clone(),
        };

        // Use default usage if not provided
        let usage = openrouter_response.usage.clone().unwrap_or_default();

        // Extract cache tokens from various possible locations
        // OpenAI format: prompt_tokens_details.cached_tokens
        // Anthropic format: cache_read_input_tokens
        let cache_read_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .or(usage.cache_read_input_tokens)
            .unwrap_or(0);
        let cache_creation_tokens = usage.cache_creation_input_tokens.unwrap_or(0);

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            },
            raw: openrouter_response.clone(),
            reasoning_content,
            finish_reason: choice.finish_reason.clone(),
        })
    }
}
