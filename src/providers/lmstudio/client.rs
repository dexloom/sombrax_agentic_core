//! LMStudio client and completion model
//!
//! Provides access to LMStudio server with anti-repetition controls.
//! LMStudio handles chat templates server-side, so this provider uses
//! plain OpenAI-compatible format without client-side template rendering.

use std::sync::Arc;

use reqwest::Client;
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default LMStudio server base URL
const DEFAULT_BASE_URL: &str = "http://localhost:1234/v1";

/// Default max tokens for LMStudio
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Default repetition context size (last N tokens checked for repeats)
const DEFAULT_REPETITION_CONTEXT_SIZE: i64 = 64;

/// LMStudio client configuration
#[derive(Clone)]
pub struct LmStudioClient {
    inner: Arc<LmStudioClientInner>,
}

struct LmStudioClientInner {
    http_client: Client,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    repeat_penalty: Option<f64>,
    repetition_context_size: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    min_p: Option<f64>,
    stop_sequences: Option<Vec<String>>,
}

impl LmStudioClient {
    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> LmStudioCompletionModel {
        LmStudioCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for LMStudio client configuration
pub struct LmStudioClientBuilder {
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    repeat_penalty: Option<f64>,
    repetition_context_size: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    min_p: Option<f64>,
    stop_sequences: Option<Vec<String>>,
}

impl Default for LmStudioClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LmStudioClientBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            repeat_penalty: None,
            repetition_context_size: None,
            frequency_penalty: None,
            presence_penalty: None,
            min_p: None,
            stop_sequences: None,
        }
    }

    /// Set custom base URL for the LMStudio server
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set temperature (clamped to 0.0-2.0)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }

    /// Set top_p sampling parameter (clamped to 0.0-1.0)
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

    /// Set repeat penalty (clamped to 1.0-2.0)
    ///
    /// Controls how much the model penalizes repeating tokens from the prompt and output.
    /// This is the primary anti-repetition parameter in LMStudio.
    /// - `1.0` = no penalty (default behavior)
    /// - `1.1` = mild penalty, good default for most use cases
    /// - `1.15` = moderate, recommended for code generation
    /// - `1.3+` = strong, may affect output quality
    pub fn repeat_penalty(mut self, penalty: f64) -> Self {
        self.repeat_penalty = Some(penalty.clamp(1.0, 2.0));
        self
    }

    /// Set repetition context size (how many recent tokens to check for repeats)
    ///
    /// Controls the window of tokens considered for `repeat_penalty`.
    /// Serialized as `repeat_last_n` in the API request (llama.cpp backend name).
    /// - `-1` = full context (check all tokens)
    /// - `0` = disabled
    /// - positive = last N tokens (e.g., 64, 256)
    ///
    /// Default: 64
    pub fn repetition_context_size(mut self, size: i64) -> Self {
        self.repetition_context_size = Some(size);
        self
    }

    /// Set frequency penalty (clamped to -2.0 to 2.0)
    ///
    /// Penalizes tokens based on their frequency in the output only.
    /// Positive values reduce repetition, negative values encourage it.
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Set presence penalty (clamped to -2.0 to 2.0)
    ///
    /// Penalizes tokens based on whether they appear in the output at all.
    /// Positive values encourage topic diversity, negative values encourage staying on topic.
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Set minimum probability floor for sampling (clamped to 0.0-1.0)
    ///
    /// Tokens with probability below this threshold are filtered out.
    /// Helps prevent low-probability garbage tokens.
    pub fn min_p(mut self, p: f64) -> Self {
        self.min_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Add stop sequences (appends to existing sequences)
    pub fn add_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        match &mut self.stop_sequences {
            Some(existing) => existing.extend(sequences),
            None => self.stop_sequences = Some(sequences),
        }
        self
    }

    /// Configure recommended anti-loop settings
    ///
    /// Sets multiple parameters to prevent the model from entering repetition loops:
    /// - `temperature`: 0.7 (increased randomness)
    /// - `top_p`: 0.95 (nucleus sampling)
    /// - `top_k`: 40 (limits token pool)
    /// - `repeat_penalty`: 1.15 (mild penalty)
    /// - `repetition_context_size`: 256 (check last 256 tokens)
    ///
    /// These values are optimized for code generation with local models.
    ///
    /// **Note:** This does NOT add stop sequences. Use in combination with
    /// `.with_anti_repetition_stops()`.
    pub fn with_anti_loop_config(mut self) -> Self {
        self.temperature = Some(0.7);
        self.top_p = Some(0.95);
        self.top_k = Some(40);
        self.repeat_penalty = Some(1.15);
        self.repetition_context_size = Some(256);
        self
    }

    /// Add anti-repetition stop sequences
    ///
    /// Adds patterns that commonly appear when models enter repetition loops.
    /// Includes detection for:
    /// - Repeated characters (EEEE, ====, ----, etc.)
    /// - Excessive whitespace (spaces, newlines, tabs)
    pub fn with_anti_repetition_stops(mut self) -> Self {
        let sequences = vec![
            // Excessive whitespace
            "\n\n\n\n".to_string(),
            "        ".to_string(),
            "            ".to_string(),
            "\t\t\t\t".to_string(),
            // Repetitive characters
            "EEEE".to_string(),
            "====".to_string(),
            "----".to_string(),
            "####".to_string(),
            "****".to_string(),
            "....".to_string(),
            ",,,,".to_string(),
        ];

        match &mut self.stop_sequences {
            Some(existing) => existing.extend(sequences),
            None => self.stop_sequences = Some(sequences),
        }
        self
    }

    /// Build the client
    pub fn build(self) -> LmStudioClient {
        LmStudioClient {
            inner: Arc::new(LmStudioClientInner {
                http_client: build_http_client(),
                base_url: self.base_url,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
                repeat_penalty: self.repeat_penalty,
                repetition_context_size: self.repetition_context_size,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                min_p: self.min_p,
                stop_sequences: self.stop_sequences,
            }),
        }
    }
}

/// LMStudio completion model
#[derive(Clone)]
pub struct LmStudioCompletionModel {
    client: LmStudioClient,
    model_id: String,
}

impl LmStudioCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "lmstudio"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "lmstudio"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<LmStudioResponse>, CompletionError> {
        let inner = &self.client.inner;

        let lmstudio_request = self.build_request(&request, inner);

        let url = format!("{}/chat/completions", inner.base_url);

        let response = inner
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&lmstudio_request)
            .send()
            .instrument(info_span!("lmstudio_http_request"))
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

        let lmstudio_response: LmStudioResponse =
            response.json().await.map_err(ProviderError::Request)?;

        self.parse_response(lmstudio_response)
    }

    /// Build request in OpenAI-compatible format with LMStudio extensions
    fn build_request(
        &self,
        request: &CompletionRequest,
        inner: &LmStudioClientInner,
    ) -> LmStudioRequest {
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(LmStudioMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                ..Default::default()
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.push(LmStudioMessage {
                role: msg.role.clone(),
                content: if msg.content.is_empty() {
                    None
                } else {
                    Some(msg.content.clone())
                },
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| LmStudioToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: LmStudioFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: msg.tool_call_id.clone(),
                reasoning: msg.reasoning.clone(),
                ..Default::default()
            });
        }

        // Build tools — LMStudio strictly requires "properties" in every tool's
        // parameters schema. Ensure it exists even if the original schema omits it.
        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .tools
                    .iter()
                    .map(|t| {
                        let params = ensure_properties(&t.parameters);
                        LmStudioTool {
                            tool_type: "function".to_string(),
                            function: LmStudioFunction {
                                name: t.name.clone(),
                                description: Some(t.description.clone()),
                                parameters: Some(params),
                            },
                        }
                    })
                    .collect(),
            )
        };

        LmStudioRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repeat_penalty: inner.repeat_penalty,
            repetition_context_size: inner
                .repetition_context_size
                .or(Some(DEFAULT_REPETITION_CONTEXT_SIZE)),
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools,
            tool_choice: None,
        }
    }

    /// Parse OpenAI-compatible response with `<think>` tag extraction
    fn parse_response(
        &self,
        response: LmStudioResponse,
    ) -> Result<CompletionResponse<LmStudioResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        let finish_reason = choice.finish_reason.clone();
        let content = choice.message.content.clone().unwrap_or_default();

        // Check for reasoning content in dedicated JSON fields
        let json_reasoning = choice
            .message
            .reasoning
            .clone()
            .or_else(|| choice.message.reasoning_content.clone());

        // Extract thinking blocks from content (<think> tags)
        let (tag_reasoning, content_without_thinking) = extract_thinking_blocks(&content);

        // Combine reasoning from JSON field and <think> tags
        let reasoning_content = match (&json_reasoning, &tag_reasoning) {
            (Some(json), Some(tag)) => Some(format!("{}\n\n{}", json, tag)),
            (Some(json), None) => Some(json.clone()),
            (None, Some(tag)) => Some(tag.clone()),
            (None, None) => None,
        };

        // Check for native tool calls first (standard OpenAI format)
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                tracing::debug!(
                    native_tool_calls_count = native_tool_calls.len(),
                    "Using native tool calls from response (LMStudio)"
                );
                let tool_calls: Vec<ToolCall> = native_tool_calls
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    })
                    .collect();

                let message = Message {
                    role: "assistant".to_string(),
                    content: content_without_thinking.clone(),
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                    reasoning: None,
                };

                return Ok(CompletionResponse {
                    message,
                    usage: Usage {
                        prompt_tokens: response.usage.prompt_tokens,
                        completion_tokens: response.usage.completion_tokens,
                        total_tokens: response.usage.total_tokens,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                    },
                    raw: response,
                    reasoning_content,
                    finish_reason,
                });
            }
        }

        // Fallback: try to parse JSON tool calls from content
        let (text_content, tool_calls) = parse_json_tool_calls(&content_without_thinking);

        let message = Message {
            role: "assistant".to_string(),
            content: text_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            reasoning: None,
        };

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
            },
            raw: response,
            reasoning_content,
            finish_reason,
        })
    }
}

/// Ensure a JSON schema object has a `"properties"` field.
///
/// LMStudio strictly validates that every tool's `parameters` schema contains
/// `"properties"`. Some tools generate schemas like `{"type": "object"}` without
/// properties, which causes a 400 error. This function adds an empty `"properties": {}`
/// if the field is missing.
fn ensure_properties(schema: &serde_json::Value) -> serde_json::Value {
    if let serde_json::Value::Object(map) = schema {
        let mut result = map.clone();
        if !result.contains_key("properties") {
            result.insert(
                "properties".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
        serde_json::Value::Object(result)
    } else {
        schema.clone()
    }
}

/// Extract `<think>...</think>` blocks from content
///
/// Returns (reasoning_content, content_without_thinking)
fn extract_thinking_blocks(content: &str) -> (Option<String>, String) {
    let mut thinking_parts = Vec::new();
    let mut text_parts = Vec::new();
    let mut remaining = content;

    while let Some(start) = remaining.find("<think>") {
        let before = &remaining[..start];
        if !before.trim().is_empty() {
            text_parts.push(before.trim().to_string());
        }

        if let Some(end) = remaining[start..].find("</think>") {
            let think_content = &remaining[start + "<think>".len()..start + end];
            if !think_content.trim().is_empty() {
                thinking_parts.push(think_content.trim().to_string());
            }
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> tag - treat rest as thinking content
            let think_content = &remaining[start + "<think>".len()..];
            if !think_content.trim().is_empty() {
                thinking_parts.push(think_content.trim().to_string());
            }
            remaining = "";
            break;
        }
    }

    if !remaining.trim().is_empty() {
        text_parts.push(remaining.trim().to_string());
    }

    let reasoning_content = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n\n"))
    };

    (reasoning_content, text_parts.join("\n"))
}

/// Parse JSON-formatted tool calls from text content
///
/// Looks for tool calls in various JSON formats:
/// - `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
/// - `{"name": "...", "arguments": {...}}`
fn parse_json_tool_calls(content: &str) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();
    let mut remaining = content;
    let mut call_index = 0;

    // Try to find <tool_call>...</tool_call> wrapped JSON
    while let Some(start) = remaining.find("<tool_call>") {
        let before = &remaining[..start];
        if !before.trim().is_empty() {
            text_parts.push(before.trim().to_string());
        }

        if let Some(end) = remaining.find("</tool_call>") {
            let json_str = &remaining[start + "<tool_call>".len()..end];
            if let Some(tc) = parse_single_json_tool_call(json_str.trim(), call_index) {
                tool_calls.push(tc);
                call_index += 1;
            }
            remaining = &remaining[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    if !remaining.trim().is_empty() {
        text_parts.push(remaining.trim().to_string());
    }

    // If no <tool_call> tags found, try standalone JSON objects
    if tool_calls.is_empty() {
        let mut search_pos = 0;
        while search_pos < content.len() {
            if let Some(obj_start) = content[search_pos..].find("{\"name\"") {
                let abs_start = search_pos + obj_start;
                if let Some(tc) =
                    try_parse_json_object_as_tool_call(&content[abs_start..], call_index)
                {
                    tool_calls.push(tc);
                    call_index += 1;
                }
                search_pos = abs_start + 1;
            } else {
                break;
            }
        }
    }

    (text_parts.join("\n"), tool_calls)
}

/// Try to parse a JSON string as a tool call
fn parse_single_json_tool_call(json_str: &str, index: usize) -> Option<ToolCall> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Check for OpenAI format: {"type": "function", "function": {"name": ..., "arguments": ...}}
    if let Some(func) = parsed.get("function") {
        let name = func.get("name")?.as_str()?.to_string();
        let arguments = match func.get("arguments") {
            Some(args) if args.is_string() => args.as_str()?.to_string(),
            Some(args) => serde_json::to_string(args).ok()?,
            None => "{}".to_string(),
        };
        let id = parsed
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("call_{}", index));
        return Some(ToolCall {
            id,
            name,
            arguments,
        });
    }

    // Check for simple format: {"name": "...", "arguments": {...}}
    let name = parsed.get("name")?.as_str()?.to_string();
    let arguments = match parsed.get("arguments") {
        Some(args) if args.is_string() => args.as_str()?.to_string(),
        Some(args) => serde_json::to_string(args).ok()?,
        None => "{}".to_string(),
    };
    let id = parsed
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("call_{}", index));
    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

/// Try to parse a JSON object starting at the given position
fn try_parse_json_object_as_tool_call(content: &str, index: usize) -> Option<ToolCall> {
    let mut depth = 0;
    let mut end_pos = 0;
    for (i, c) in content.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_pos = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    if end_pos == 0 {
        return None;
    }

    let json_str = &content[..end_pos];
    parse_single_json_tool_call(json_str, index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let client = LmStudioClientBuilder::new().build();
        assert_eq!(client.inner.base_url, "http://localhost:1234/v1");
        assert!(client.inner.temperature.is_none());
        assert!(client.inner.repeat_penalty.is_none());
        assert!(client.inner.repetition_context_size.is_none());
        assert!(client.inner.frequency_penalty.is_none());
        assert!(client.inner.presence_penalty.is_none());
        assert!(client.inner.min_p.is_none());
    }

    #[test]
    fn test_builder_custom_url() {
        let client = LmStudioClientBuilder::new()
            .base_url("http://localhost:1234/v1")
            .build();
        assert_eq!(client.inner.base_url, "http://localhost:1234/v1");
    }

    #[test]
    fn test_temperature_clamping() {
        let client = LmStudioClientBuilder::new().temperature(3.0).build();
        assert_eq!(client.inner.temperature, Some(2.0));

        let client = LmStudioClientBuilder::new().temperature(-1.0).build();
        assert_eq!(client.inner.temperature, Some(0.0));

        let client = LmStudioClientBuilder::new().temperature(0.7).build();
        assert_eq!(client.inner.temperature, Some(0.7));
    }

    #[test]
    fn test_top_p_clamping() {
        let client = LmStudioClientBuilder::new().top_p(1.5).build();
        assert_eq!(client.inner.top_p, Some(1.0));

        let client = LmStudioClientBuilder::new().top_p(-0.5).build();
        assert_eq!(client.inner.top_p, Some(0.0));
    }

    #[test]
    fn test_repeat_penalty_clamping() {
        let client = LmStudioClientBuilder::new().repeat_penalty(0.5).build();
        assert_eq!(client.inner.repeat_penalty, Some(1.0));

        let client = LmStudioClientBuilder::new().repeat_penalty(3.0).build();
        assert_eq!(client.inner.repeat_penalty, Some(2.0));

        let client = LmStudioClientBuilder::new().repeat_penalty(1.15).build();
        assert_eq!(client.inner.repeat_penalty, Some(1.15));
    }

    #[test]
    fn test_frequency_penalty_clamping() {
        let client = LmStudioClientBuilder::new().frequency_penalty(3.0).build();
        assert_eq!(client.inner.frequency_penalty, Some(2.0));

        let client = LmStudioClientBuilder::new().frequency_penalty(-3.0).build();
        assert_eq!(client.inner.frequency_penalty, Some(-2.0));

        let client = LmStudioClientBuilder::new().frequency_penalty(0.5).build();
        assert_eq!(client.inner.frequency_penalty, Some(0.5));
    }

    #[test]
    fn test_presence_penalty_clamping() {
        let client = LmStudioClientBuilder::new().presence_penalty(3.0).build();
        assert_eq!(client.inner.presence_penalty, Some(2.0));

        let client = LmStudioClientBuilder::new().presence_penalty(-3.0).build();
        assert_eq!(client.inner.presence_penalty, Some(-2.0));
    }

    #[test]
    fn test_min_p_clamping() {
        let client = LmStudioClientBuilder::new().min_p(1.5).build();
        assert_eq!(client.inner.min_p, Some(1.0));

        let client = LmStudioClientBuilder::new().min_p(-0.5).build();
        assert_eq!(client.inner.min_p, Some(0.0));

        let client = LmStudioClientBuilder::new().min_p(0.05).build();
        assert_eq!(client.inner.min_p, Some(0.05));
    }

    #[test]
    fn test_repetition_context_size() {
        let client = LmStudioClientBuilder::new()
            .repetition_context_size(256)
            .build();
        assert_eq!(client.inner.repetition_context_size, Some(256));

        let client = LmStudioClientBuilder::new()
            .repetition_context_size(-1)
            .build();
        assert_eq!(client.inner.repetition_context_size, Some(-1));
    }

    #[test]
    fn test_anti_loop_config() {
        let client = LmStudioClientBuilder::new().with_anti_loop_config().build();

        assert_eq!(client.inner.temperature, Some(0.7));
        assert_eq!(client.inner.top_p, Some(0.95));
        assert_eq!(client.inner.top_k, Some(40));
        assert_eq!(client.inner.repeat_penalty, Some(1.15));
        assert_eq!(client.inner.repetition_context_size, Some(256));
    }

    #[test]
    fn test_anti_loop_config_can_override() {
        let client = LmStudioClientBuilder::new()
            .with_anti_loop_config()
            .temperature(0.8)
            .build();

        assert_eq!(client.inner.temperature, Some(0.8));
        assert_eq!(client.inner.top_p, Some(0.95)); // Still from preset
    }

    #[test]
    fn test_anti_repetition_stops() {
        let client = LmStudioClientBuilder::new()
            .with_anti_repetition_stops()
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        assert!(stops.contains(&"EEEE".to_string()));
        assert!(stops.contains(&"====".to_string()));
        assert!(stops.contains(&"----".to_string()));
        assert!(stops.contains(&"\n\n\n\n".to_string()));
    }

    #[test]
    fn test_stop_sequences() {
        let client = LmStudioClientBuilder::new()
            .stop_sequences(vec!["STOP".to_string()])
            .add_stop_sequences(vec!["END".to_string()])
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        assert_eq!(stops.len(), 2);
        assert!(stops.contains(&"STOP".to_string()));
        assert!(stops.contains(&"END".to_string()));
    }

    #[test]
    fn test_request_serialization() {
        let client = LmStudioClientBuilder::new()
            .repeat_penalty(1.15)
            .repetition_context_size(256)
            .frequency_penalty(0.1)
            .presence_penalty(0.1)
            .min_p(0.05)
            .build();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: Some("You are helpful.".to_string()),
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
        };

        let lmstudio_request = model.build_request(&request, &model.client.inner);

        // Verify the request serializes correctly
        let json = serde_json::to_value(&lmstudio_request).unwrap();

        // Check field names are correct for LMStudio
        assert_eq!(json["repeat_penalty"], 1.15);
        assert_eq!(json["repeat_last_n"], 256); // serialized as repeat_last_n
        assert_eq!(json["frequency_penalty"], 0.1);
        assert_eq!(json["presence_penalty"], 0.1);
        assert_eq!(json["min_p"], 0.05);

        // Ensure we don't accidentally use mlxlm field name
        assert!(json.get("repetition_penalty").is_none());
    }

    #[test]
    fn test_request_default_repetition_context_size() {
        let client = LmStudioClientBuilder::new().repeat_penalty(1.1).build();
        let model = client.completion_model("test-model");

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
        };

        let lmstudio_request = model.build_request(&request, &model.client.inner);
        let json = serde_json::to_value(&lmstudio_request).unwrap();

        // Default repetition_context_size should be 64
        assert_eq!(json["repeat_last_n"], 64);
    }

    #[test]
    fn test_ensure_properties_added_when_missing() {
        let schema = serde_json::json!({"type": "object"});
        let fixed = ensure_properties(&schema);
        assert!(fixed.get("properties").is_some());
        assert_eq!(fixed["properties"], serde_json::json!({}));
        assert_eq!(fixed["type"], "object");
    }

    #[test]
    fn test_ensure_properties_preserved_when_present() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        let fixed = ensure_properties(&schema);
        assert_eq!(fixed["properties"]["name"]["type"], "string");
    }

    #[test]
    fn test_tools_get_properties_in_request() {
        use crate::providers::zai::client::ToolDefinition;

        let client = LmStudioClientBuilder::new().build();
        let model = client.completion_model("test-model");

        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            }],
            tools: vec![ToolDefinition {
                name: "no_args_tool".to_string(),
                description: "A tool with no parameters".to_string(),
                parameters: serde_json::json!({"type": "object"}), // No properties!
            }],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };

        let lmstudio_request = model.build_request(&request, &model.client.inner);
        let json = serde_json::to_value(&lmstudio_request).unwrap();

        // Verify properties was injected
        let tool_params = &json["tools"][0]["function"]["parameters"];
        assert!(tool_params.get("properties").is_some());
        assert_eq!(tool_params["properties"], serde_json::json!({}));
    }

    #[test]
    fn test_extract_thinking_blocks_single() {
        let content = r#"<think>
Let me analyze this.
</think>

Here is my response."#;

        let (reasoning, text) = extract_thinking_blocks(content);

        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("analyze this"));
        assert!(text.contains("Here is my response"));
        assert!(!text.contains("<think>"));
    }

    #[test]
    fn test_extract_thinking_blocks_multiple() {
        let content = r#"<think>
First thought.
</think>

Middle text.

<think>
Second thought.
</think>

Final text."#;

        let (reasoning, text) = extract_thinking_blocks(content);

        let reasoning = reasoning.unwrap();
        assert!(reasoning.contains("First thought"));
        assert!(reasoning.contains("Second thought"));
        assert!(text.contains("Middle text"));
        assert!(text.contains("Final text"));
    }

    #[test]
    fn test_extract_thinking_blocks_none() {
        let content = "A response without thinking blocks.";
        let (reasoning, text) = extract_thinking_blocks(content);

        assert!(reasoning.is_none());
        assert_eq!(text, content);
    }

    #[test]
    fn test_parse_json_tool_calls_xml_wrapped() {
        let content = r#"I'll check that for you.

<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>"#;

        let (text, tool_calls) = parse_json_tool_calls(content);
        assert_eq!(text, "I'll check that for you.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_parse_json_tool_calls_none() {
        let content = "Just a regular response.";
        let (text, tool_calls) = parse_json_tool_calls(content);

        assert_eq!(text, content);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_parse_response_with_native_tool_calls() {
        let client = LmStudioClientBuilder::new().build();
        let model = client.completion_model("test-model");

        let response = LmStudioResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![LmStudioChoice {
                index: 0,
                message: LmStudioMessage {
                    role: "assistant".to_string(),
                    content: Some("Let me check.".to_string()),
                    tool_calls: Some(vec![LmStudioToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: LmStudioFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"Paris"}"#.to_string(),
                        },
                    }]),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: LmStudioUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };

        let result = model.parse_response(response).unwrap();
        assert_eq!(result.message.role, "assistant");
        assert!(result.message.tool_calls.is_some());
        let tcs = result.message.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "get_weather");
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 20);
    }

    #[test]
    fn test_parse_response_with_thinking() {
        let client = LmStudioClientBuilder::new().build();
        let model = client.completion_model("test-model");

        let response = LmStudioResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![LmStudioChoice {
                index: 0,
                message: LmStudioMessage {
                    role: "assistant".to_string(),
                    content: Some(
                        "<think>Let me reason...</think>\n\nThe answer is 42.".to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: LmStudioUsage::default(),
        };

        let result = model.parse_response(response).unwrap();
        assert!(result.reasoning_content.is_some());
        assert!(result.reasoning_content.unwrap().contains("Let me reason"));
        assert!(result.message.content.contains("The answer is 42"));
        assert!(!result.message.content.contains("<think>"));
    }

    #[test]
    fn test_parse_response_plain_text() {
        let client = LmStudioClientBuilder::new().build();
        let model = client.completion_model("test-model");

        let response = LmStudioResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![LmStudioChoice {
                index: 0,
                message: LmStudioMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello! How can I help?".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: LmStudioUsage {
                prompt_tokens: 5,
                completion_tokens: 10,
                total_tokens: 15,
            },
        };

        let result = model.parse_response(response).unwrap();
        assert_eq!(result.message.content, "Hello! How can I help?");
        assert!(result.message.tool_calls.is_none());
        assert!(result.reasoning_content.is_none());
    }
}
