//! Ollama native client and completion model.
//!
//! Speaks Ollama's own `/api/chat` protocol and serves both:
//!
//! - **Local**: `http://localhost:11434`, no API key.
//! - **Cloud**: `https://ollama.com`, `Authorization: Bearer $OLLAMA_API_KEY`.
//!
//! Cloud vs local is purely `base_url` + presence of a key — the wire format
//! is identical.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use reqwest::Client;
use tracing::{info_span, instrument, warn, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default Ollama base URL (local server).
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Default max output tokens.
const DEFAULT_MAX_TOKENS: u64 = 8192;

/// Sentinel `build_agent` passes when a config has no API key.
const NO_KEY_SENTINEL: &str = "none";

/// Normalize a configured base URL into a full `/api/chat` endpoint.
///
/// Strips a trailing `/v1` or `/v1/` (legacy OpenAI-compat config that
/// predates the native provider) and any trailing slash, then appends
/// `/api/chat`. Returns `(endpoint, stripped_v1)` so the caller can warn.
fn normalize_chat_endpoint(base_url: &str) -> (String, bool) {
    let trimmed = base_url.trim_end_matches('/');
    let (root, stripped) = if let Some(stripped) = trimmed.strip_suffix("/v1") {
        (stripped.trim_end_matches('/'), true)
    } else {
        (trimmed, false)
    };
    (format!("{root}/api/chat"), stripped)
}

/// Convert a JSON-string arguments blob (sac's shared shape) into the JSON
/// **object** Ollama requires. Fails fast on invalid JSON or any non-object
/// value — Ollama's `function.arguments` is specifically an object.
fn args_string_to_object(args: &str) -> Result<serde_json::Value, CompletionError> {
    let value: serde_json::Value = serde_json::from_str(args).map_err(|e| {
        CompletionError::InvalidRequest(format!(
            "ollama: tool-call arguments are not valid JSON: {e}"
        ))
    })?;
    if !value.is_object() {
        return Err(CompletionError::InvalidRequest(format!(
            "ollama: tool-call arguments must be a JSON object, got: {value}"
        )));
    }
    Ok(value)
}

/// Convert Ollama's object arguments back into sac's JSON-string shape.
fn args_object_to_string(value: &serde_json::Value) -> String {
    if value.is_null() {
        "{}".to_string()
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Ollama client configuration.
#[derive(Clone)]
pub struct OllamaClient {
    inner: Arc<OllamaClientInner>,
}

struct OllamaClientInner {
    http_client: Client,
    /// `None` for keyless local use; `Some` adds a Bearer header (cloud).
    api_key: Option<String>,
    /// Fully-resolved `…/api/chat` endpoint.
    endpoint: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    max_tokens: Option<u64>,
    num_ctx: Option<u64>,
    keep_alive: Option<String>,
    enable_thinking: bool,
    think_level: Option<String>,
}

impl OllamaClient {
    /// Create a client from the `OLLAMA_API_KEY` environment variable
    /// (cloud usage). For local use, build with [`OllamaClientBuilder`].
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("OLLAMA_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("OLLAMA_API_KEY".to_string()))?;
        Ok(OllamaClientBuilder::new()
            .base_url("https://ollama.com")
            .api_key(&api_key)
            .build())
    }

    /// Create a completion model for a specific model ID.
    pub fn completion_model(&self, model_id: &str) -> OllamaCompletionModel {
        OllamaCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for [`OllamaClient`].
pub struct OllamaClientBuilder {
    base_url: String,
    api_key: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    max_tokens: Option<u64>,
    num_ctx: Option<u64>,
    keep_alive: Option<String>,
    enable_thinking: bool,
    think_level: Option<String>,
}

impl Default for OllamaClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaClientBuilder {
    /// Create a new builder (local defaults, no API key).
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            max_tokens: None,
            num_ctx: None,
            keep_alive: None,
            enable_thinking: false,
            think_level: None,
        }
    }

    /// Set the base URL (e.g. `https://ollama.com` for cloud). A trailing
    /// `/v1` is stripped automatically for legacy-config compatibility.
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set the API key. Ignored when empty or the `"none"` sentinel so
    /// keyless local use stays headerless.
    pub fn api_key(mut self, key: &str) -> Self {
        if !key.is_empty() && key != NO_KEY_SENTINEL {
            self.api_key = Some(key.to_string());
        }
        self
    }

    /// Set temperature (clamped to 0.0-2.0).
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }

    /// Set top_p (clamped to 0.0-1.0).
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set top_k.
    pub fn top_k(mut self, k: u64) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set min_p (clamped to 0.0-1.0).
    pub fn min_p(mut self, p: f64) -> Self {
        self.min_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set max output tokens (mapped to Ollama `num_predict`).
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the context window size (`num_ctx`).
    pub fn num_ctx(mut self, ctx: u64) -> Self {
        self.num_ctx = Some(ctx);
        self
    }

    /// Set the `keep_alive` model-residency hint (e.g. `"5m"`, `"0"`).
    pub fn keep_alive(mut self, value: &str) -> Self {
        self.keep_alive = Some(value.to_string());
        self
    }

    /// Enable/disable native thinking traces.
    pub fn enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = enabled;
        self
    }

    /// Set a thinking effort level (`"high"` / `"medium"` / `"low"`).
    /// Implies thinking enabled.
    pub fn think_level(mut self, level: &str) -> Self {
        self.think_level = Some(level.to_string());
        self.enable_thinking = true;
        self
    }

    /// Build the client.
    pub fn build(self) -> OllamaClient {
        let (endpoint, stripped_v1) = normalize_chat_endpoint(&self.base_url);
        if stripped_v1 {
            warn!(
                "Ollama: stripped legacy '/v1' suffix from base_url; \
                 using native endpoint '{}'. Update your config to drop '/v1'.",
                endpoint
            );
        }
        OllamaClient {
            inner: Arc::new(OllamaClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                endpoint,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                min_p: self.min_p,
                max_tokens: self.max_tokens,
                num_ctx: self.num_ctx,
                keep_alive: self.keep_alive,
                enable_thinking: self.enable_thinking,
                think_level: self.think_level,
            }),
        }
    }
}

/// Ollama completion model.
#[derive(Clone)]
pub struct OllamaCompletionModel {
    client: OllamaClient,
    model_id: String,
}

impl OllamaCompletionModel {
    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name.
    pub fn provider(&self) -> &str {
        "ollama"
    }

    /// Build the Ollama message list from the shared request, recovering
    /// `tool_name` for tool-result turns by tracking assistant tool-call
    /// ids as we iterate (the agent has already remapped ids in history).
    fn build_messages(request: &CompletionRequest) -> Result<Vec<OllamaMessage>, CompletionError> {
        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: preamble.clone(),
                ..Default::default()
            });
        }

        // Running id -> tool name map, populated from assistant tool calls
        // and consumed by the following `role: "tool"` messages.
        let mut tool_names: HashMap<String, String> = HashMap::new();

        for msg in &request.messages {
            let tool_calls = match &msg.tool_calls {
                Some(calls) => {
                    let mut out = Vec::with_capacity(calls.len());
                    for tc in calls {
                        tool_names.insert(tc.id.clone(), tc.name.clone());
                        out.push(OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: tc.name.clone(),
                                arguments: args_string_to_object(&tc.arguments)?,
                            },
                        });
                    }
                    Some(out)
                }
                None => None,
            };

            let tool_name = if msg.role == "tool" {
                let name = msg
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| tool_names.get(id).cloned());
                if name.is_none() {
                    warn!(
                        "Ollama: no matching tool name for tool result (id={:?}); \
                         sending tool_name=None",
                        msg.tool_call_id
                    );
                }
                name
            } else {
                None
            };

            messages.push(OllamaMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
                // Replay assistant reasoning so follow-up tool turns keep
                // the native thinking context the model expects echoed.
                thinking: if msg.role == "assistant" {
                    msg.reasoning.clone()
                } else {
                    None
                },
                tool_calls,
                tool_name,
            });
        }

        Ok(messages)
    }

    /// Send a completion request.
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "ollama"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<OllamaResponse>, CompletionError> {
        let inner = &self.client.inner;

        let messages = Self::build_messages(&request)?;

        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .tools
                    .iter()
                    .map(|t| OllamaTool {
                        tool_type: "function".to_string(),
                        function: OllamaFunction {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.parameters.clone(),
                        },
                    })
                    .collect(),
            )
        };

        let num_predict = request
            .max_tokens
            .or(inner.max_tokens)
            .or(Some(DEFAULT_MAX_TOKENS))
            .map(|t| t as i64);

        let options = OllamaOptions {
            temperature: request.temperature.or(inner.temperature),
            top_p: inner.top_p,
            top_k: inner.top_k,
            min_p: inner.min_p,
            num_predict,
            num_ctx: inner.num_ctx,
            seed: None,
            stop: None,
        };

        // Send `think` EXPLICITLY in both directions. Omitting it lets the
        // model fall back to its own default — and cloud reasoning models
        // (glm-5.1, minimax, qwen3.5, …) default thinking ON, which would
        // silently contaminate "no-thinking" runs. Explicit `think:false`
        // actually suppresses reasoning so think/no-think is a real A/B.
        let think = Some(if inner.enable_thinking {
            match &inner.think_level {
                Some(level) => OllamaThink::Level(level.clone()),
                None => OllamaThink::Enabled(true),
            }
        } else {
            OllamaThink::Enabled(false)
        });

        let ollama_request = OllamaRequest {
            model: self.model_id.clone(),
            messages,
            tools,
            stream: false,
            think,
            keep_alive: inner.keep_alive.clone(),
            options: if options.is_empty() {
                None
            } else {
                Some(options)
            },
        };

        let mut req = inner
            .http_client
            .post(&inner.endpoint)
            .header("Content-Type", "application/json");
        if let Some(key) = &inner.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let response = req
            .json(&ollama_request)
            .send()
            .instrument(info_span!("ollama_http_request"))
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

        let ollama_response: OllamaResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                tracing::error!(
                    "Failed to deserialize Ollama response: {}\nRaw response: {}",
                    e,
                    &response_text[..response_text.len().min(2000)]
                );
                ProviderError::InvalidResponse(format!(
                    "JSON deserialization failed: {}. Response preview: {}",
                    e,
                    &response_text[..response_text.len().min(500)]
                ))
            })?;

        // Synthesize stable tool-call ids (Ollama emits none); the agent
        // remaps these before they re-enter history.
        let tool_calls = ollama_response.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(idx, tc)| ToolCall {
                    id: format!("call_{idx}_{}", tc.function.name),
                    name: tc.function.name.clone(),
                    arguments: args_object_to_string(&tc.function.arguments),
                })
                .collect()
        });

        let reasoning = ollama_response
            .message
            .thinking
            .clone()
            .filter(|t| !t.is_empty());

        let message = Message {
            role: ollama_response.message.role.clone(),
            content: ollama_response.message.content.clone(),
            tool_calls,
            tool_call_id: None,
            reasoning: reasoning.clone(),
        };

        let prompt_tokens = ollama_response.prompt_eval_count.unwrap_or(0);
        let completion_tokens = ollama_response.eval_count.unwrap_or(0);

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
            finish_reason: ollama_response.done_reason.clone(),
            reasoning_content: reasoning,
            raw: ollama_response,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_endpoint_variants() {
        assert_eq!(
            normalize_chat_endpoint("http://localhost:11434"),
            ("http://localhost:11434/api/chat".to_string(), false)
        );
        assert_eq!(
            normalize_chat_endpoint("http://localhost:11434/"),
            ("http://localhost:11434/api/chat".to_string(), false)
        );
        assert_eq!(
            normalize_chat_endpoint("http://localhost:11434/v1"),
            ("http://localhost:11434/api/chat".to_string(), true)
        );
        assert_eq!(
            normalize_chat_endpoint("http://localhost:11434/v1/"),
            ("http://localhost:11434/api/chat".to_string(), true)
        );
        assert_eq!(
            normalize_chat_endpoint("https://ollama.com"),
            ("https://ollama.com/api/chat".to_string(), false)
        );
    }

    #[test]
    fn args_string_to_object_valid() {
        let v = args_string_to_object(r#"{"city":"Paris"}"#).unwrap();
        assert_eq!(v, serde_json::json!({"city": "Paris"}));
        let empty = args_string_to_object("{}").unwrap();
        assert_eq!(empty, serde_json::json!({}));
        let nested = args_string_to_object(r#"{"a":{"b":[1,2]}}"#).unwrap();
        assert_eq!(nested, serde_json::json!({"a":{"b":[1,2]}}));
    }

    #[test]
    fn args_string_to_object_rejects_non_object() {
        for bad in [r#"not json"#, r#"[]"#, r#"42"#, r#""s""#] {
            let err = args_string_to_object(bad).unwrap_err();
            assert!(matches!(err, CompletionError::InvalidRequest(_)), "{bad}");
        }
    }

    #[test]
    fn args_object_to_string_roundtrip() {
        let s = args_object_to_string(&serde_json::json!({"x": 1}));
        assert_eq!(s, r#"{"x":1}"#);
        assert_eq!(args_object_to_string(&serde_json::Value::Null), "{}");
    }

    #[test]
    fn api_key_ignores_sentinel_and_empty() {
        assert!(OllamaClientBuilder::new()
            .api_key("none")
            .build()
            .inner
            .api_key
            .is_none());
        assert!(OllamaClientBuilder::new()
            .api_key("")
            .build()
            .inner
            .api_key
            .is_none());
        assert_eq!(
            OllamaClientBuilder::new()
                .api_key("sk-real")
                .build()
                .inner
                .api_key
                .as_deref(),
            Some("sk-real")
        );
    }

    #[test]
    fn build_messages_recovers_tool_name_and_replays_thinking() {
        let request = CompletionRequest {
            preamble: Some("sys".into()),
            messages: vec![
                Message {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: Some(vec![
                        ToolCall {
                            id: "id-a".into(),
                            name: "get_weather".into(),
                            arguments: r#"{"city":"Paris"}"#.into(),
                        },
                        ToolCall {
                            id: "id-b".into(),
                            name: "get_time".into(),
                            arguments: "{}".into(),
                        },
                    ]),
                    tool_call_id: None,
                    reasoning: Some("thinking trace".into()),
                },
                Message {
                    role: "tool".into(),
                    content: "sunny".into(),
                    tool_calls: None,
                    tool_call_id: Some("id-a".into()),
                    reasoning: None,
                },
                Message {
                    role: "tool".into(),
                    content: "12:00".into(),
                    tool_calls: None,
                    tool_call_id: Some("id-b".into()),
                    reasoning: None,
                },
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };

        let msgs = OllamaCompletionModel::build_messages(&request).unwrap();
        assert_eq!(msgs[0].role, "system");
        // assistant turn replays reasoning into `thinking`
        assert_eq!(msgs[1].thinking.as_deref(), Some("thinking trace"));
        assert_eq!(msgs[1].tool_calls.as_ref().unwrap().len(), 2);
        // tool results recover their names from the running map
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].tool_name.as_deref(), Some("get_weather"));
        assert_eq!(msgs[3].tool_name.as_deref(), Some("get_time"));
    }

    #[test]
    fn build_messages_unmatched_tool_id_is_none() {
        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message {
                role: "tool".into(),
                content: "orphan".into(),
                tool_calls: None,
                tool_call_id: Some("missing".into()),
                reasoning: None,
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };
        let msgs = OllamaCompletionModel::build_messages(&request).unwrap();
        assert_eq!(msgs[0].role, "tool");
        assert!(msgs[0].tool_name.is_none());
    }

    #[test]
    fn build_messages_invalid_args_fails_fast() {
        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: "x".into(),
                    name: "f".into(),
                    arguments: "not json".into(),
                }]),
                tool_call_id: None,
                reasoning: None,
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };
        assert!(matches!(
            OllamaCompletionModel::build_messages(&request),
            Err(CompletionError::InvalidRequest(_))
        ));
    }
}
