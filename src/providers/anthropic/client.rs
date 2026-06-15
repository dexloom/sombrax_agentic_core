//! Anthropic (Claude) client and completion model

use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::Client;
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::provider::CacheHints;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default Anthropic API base URL
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Default API version
const API_VERSION: &str = "2023-06-01";

/// Default max tokens for Anthropic
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Anthropic client configuration
#[derive(Clone)]
pub struct AnthropicClient {
    inner: Arc<AnthropicClientInner>,
}

struct AnthropicClientInner {
    http_client: Client,
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    enable_thinking: bool,
    thinking_budget_tokens: Option<u64>,
    prompt_caching: bool,
}

impl AnthropicClient {
    /// Create a new Anthropic client from environment variable
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("ANTHROPIC_API_KEY".to_string()))?;

        Ok(AnthropicClientBuilder::new(&api_key).build())
    }

    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> AnthropicCompletionModel {
        AnthropicCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for Anthropic client configuration
pub struct AnthropicClientBuilder {
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    enable_thinking: bool,
    thinking_budget_tokens: Option<u64>,
    prompt_caching: bool,
}

impl AnthropicClientBuilder {
    /// Create a new builder with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            enable_thinking: false,
            thinking_budget_tokens: None,
            prompt_caching: true,
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

    /// Enable/disable thinking mode (default: false)
    pub fn enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = enabled;
        self
    }

    /// Set thinking budget in tokens (overrides default budget derived from max_tokens)
    pub fn thinking_budget_tokens(mut self, tokens: u64) -> Self {
        self.thinking_budget_tokens = Some(tokens);
        self
    }

    /// Enable/disable explicit prompt-cache breakpoints (default: true).
    ///
    /// When enabled, the client translates the request's `CacheHints` into
    /// `cache_control` markers on the system, tools, and tail messages. When
    /// disabled — or when the request carries no hints — the wire body is
    /// byte-identical to the pre-caching representation.
    pub fn prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Build the client
    pub fn build(self) -> AnthropicClient {
        AnthropicClient {
            inner: Arc::new(AnthropicClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                base_url: self.base_url,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
                enable_thinking: self.enable_thinking,
                thinking_budget_tokens: self.thinking_budget_tokens,
                prompt_caching: self.prompt_caching,
            }),
        }
    }
}

/// Translate provider-independent [`CacheHints`] into Anthropic `cache_control`
/// markers on the already-merged request structure.
///
/// Placement (≤3 of Anthropic's 4 allowed breakpoints):
/// 1. `cache_system` → the static prefix: the system block if present, else the
///    last tool (tools precede the system prompt in the cacheable prefix, so one
///    marker there caches the whole tools+system head).
/// 2. The last content block of the final message (the moving tail).
/// 3. The last content block of the most recent *earlier* user message — the
///    resilient breakpoint that matches the previous request's tail even when
///    this turn appended several blocks.
///
/// Operates on the merged `Vec<AnthropicMessage>`, so it is robust to the
/// consecutive-tool-result merge. A no-op when `hints` is empty.
fn apply_cache_breakpoints(
    system: &mut Option<AnthropicSystem>,
    messages: &mut [AnthropicMessage],
    tools: Option<&mut Vec<AnthropicTool>>,
    hints: &CacheHints,
) {
    if hints.is_empty() {
        return;
    }

    // 1. Static prefix: prefer the system block, fall back to the last tool.
    if hints.cache_system {
        match system {
            Some(sys) => mark_system(sys),
            None => {
                if let Some(last_tool) = tools.and_then(|t| t.last_mut()) {
                    last_tool.cache_control = Some(AnthropicCacheControl::ephemeral());
                }
            }
        }
    }

    // 2 & 3. Moving tail breakpoints.
    if !hints.breakpoints.is_empty() && !messages.is_empty() {
        let last = messages.len() - 1;
        mark_last_block(&mut messages[last]);
        if let Some(prev_user) = messages[..last].iter().rposition(|m| m.role == "user") {
            mark_last_block(&mut messages[prev_user]);
        }
    }
}

/// Attach a cache breakpoint to the system prompt, promoting a plain string to a
/// single cache-marked text block if needed.
fn mark_system(system: &mut AnthropicSystem) {
    match system {
        AnthropicSystem::Blocks(blocks) => {
            if let Some(last) = blocks.last_mut() {
                last.cache_control = Some(AnthropicCacheControl::ephemeral());
            }
        }
        AnthropicSystem::Text(text) => {
            *system = AnthropicSystem::Blocks(vec![AnthropicSystemBlock {
                block_type: "text".to_string(),
                text: std::mem::take(text),
                cache_control: Some(AnthropicCacheControl::ephemeral()),
            }]);
        }
    }
}

/// Attach a cache breakpoint to the last cache-eligible content block of a
/// message (skipping `thinking` blocks, which the API forbids marking).
/// Promotes a plain `Text` content to a single cache-marked block if needed.
fn mark_last_block(message: &mut AnthropicMessage) {
    match &mut message.content {
        AnthropicContent::Blocks(blocks) => {
            for block in blocks.iter_mut().rev() {
                match block {
                    AnthropicContentBlock::Text { cache_control, .. }
                    | AnthropicContentBlock::ToolUse { cache_control, .. }
                    | AnthropicContentBlock::ToolResult { cache_control, .. } => {
                        *cache_control = Some(AnthropicCacheControl::ephemeral());
                        return;
                    }
                    AnthropicContentBlock::Thinking { .. } => continue,
                }
            }
        }
        AnthropicContent::Text(text) => {
            message.content = AnthropicContent::Blocks(vec![AnthropicContentBlock::Text {
                text: std::mem::take(text),
                cache_control: Some(AnthropicCacheControl::ephemeral()),
            }]);
        }
    }
}

/// Anthropic completion model
#[derive(Clone)]
pub struct AnthropicCompletionModel {
    client: AnthropicClient,
    model_id: String,
}

impl AnthropicCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "anthropic"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "anthropic"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<AnthropicResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Extract system message from preamble. Plain `Text` by default — only
        // promoted to cache-marked blocks below when caching is active.
        let mut system = request.preamble.clone().map(AnthropicSystem::Text);

        // Convert messages, merging consecutive tool results into a single user message.
        // The Anthropic API requires all tool_result blocks for a multi-tool-call assistant
        // message to appear in one user message immediately after.
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for msg in request.messages.iter().filter(|m| m.role != "system") {
            if let Some(tool_call_id) = &msg.tool_call_id {
                // Tool result — try to merge into previous user message if it also has tool results
                let tool_result_block = AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_call_id.clone(),
                    content: msg.content.clone(),
                    is_error: None,
                    cache_control: None,
                };

                if let Some(last) = messages.last_mut() {
                    if last.role == "user" {
                        if let AnthropicContent::Blocks(ref mut blocks) = last.content {
                            let all_tool_results = blocks
                                .iter()
                                .all(|b| matches!(b, AnthropicContentBlock::ToolResult { .. }));
                            if all_tool_results {
                                blocks.push(tool_result_block);
                                continue;
                            }
                        }
                    }
                }

                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![tool_result_block]),
                });
                continue;
            }

            let mut blocks = Vec::new();

            // Include thinking block for assistant messages with reasoning
            if msg.role == "assistant" {
                if let Some(reasoning) = &msg.reasoning {
                    let include_thinking =
                        !reasoning.is_empty() && !msg.content.contains(reasoning);
                    if include_thinking {
                        blocks.push(AnthropicContentBlock::Thinking {
                            thinking: reasoning.clone(),
                        });
                    }
                }
            }

            let content = if let Some(tool_calls) = &msg.tool_calls {
                // Convert tool calls to content blocks
                if !msg.content.is_empty() {
                    blocks.push(AnthropicContentBlock::Text {
                        text: msg.content.clone(),
                        cache_control: None,
                    });
                }

                for tc in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    blocks.push(AnthropicContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input,
                        cache_control: None,
                    });
                }

                AnthropicContent::Blocks(blocks)
            } else if blocks.is_empty() {
                AnthropicContent::Text(msg.content.clone())
            } else {
                if !msg.content.is_empty() {
                    blocks.push(AnthropicContentBlock::Text {
                        text: msg.content.clone(),
                        cache_control: None,
                    });
                }
                AnthropicContent::Blocks(blocks)
            };

            messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content,
            });
        }

        // Build tools (always send array, even if empty)
        // Ensure each tool's input_schema has a "properties" field (required by some APIs)
        let mut tools = Some(
            request
                .tools
                .iter()
                .map(|t| {
                    let mut schema = t.parameters.clone();
                    if let Some(obj) = schema.as_object_mut() {
                        obj.entry("properties")
                            .or_insert_with(|| serde_json::json!({}));
                    }
                    AnthropicTool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: schema,
                        cache_control: None,
                    }
                })
                .collect(),
        );

        // Translate provider-independent cache hints into Anthropic
        // `cache_control` markers on the merged structure. No-op (byte-identical
        // wire body) when caching is disabled or the request carries no hints.
        if inner.prompt_caching {
            apply_cache_breakpoints(&mut system, &mut messages, tools.as_mut(), &request.cache);
        }

        let max_tokens = request
            .max_tokens
            .or(inner.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        // Build thinking config
        let thinking = if inner.enable_thinking {
            let budget = inner.thinking_budget_tokens.unwrap_or(max_tokens);
            Some(AnthropicThinkingConfig::enabled(budget))
        } else {
            None
        };

        let anthropic_request = AnthropicRequest {
            model: self.model_id.clone(),
            messages,
            max_tokens,
            system,
            temperature: request.temperature.or(inner.temperature),
            top_p: inner.top_p,
            top_k: inner.top_k,
            tools,
            tool_choice: None,
            metadata: None,
            thinking,
            stream: Some(true),
        };

        let url = format!("{}/v1/messages", inner.base_url);

        let response = inner
            .http_client
            .post(&url)
            .header("x-api-key", &inner.api_key)
            .header("anthropic-version", API_VERSION)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&anthropic_request)
            .send()
            .instrument(info_span!("anthropic_http_request"))
            .await
            .map_err(ProviderError::Request)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();

            // Check for rate limiting
            if status.as_u16() == 429 {
                return Err(CompletionError::Provider(ProviderError::RateLimited {
                    retry_after_ms: None,
                }));
            }

            // Check for authentication error
            if status.as_u16() == 401 {
                return Err(CompletionError::Provider(ProviderError::Authentication(
                    error_text,
                )));
            }

            return Err(CompletionError::Provider(ProviderError::Http {
                status: status.as_u16(),
                message: error_text,
            }));
        }

        let anthropic_response = collect_sse_stream(response, &self.model_id).await?;

        // Extract content, reasoning, and tool calls
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();

        for block in &anthropic_response.content {
            match block {
                AnthropicResponseContent::Text { text } => {
                    content.push_str(text);
                }
                AnthropicResponseContent::Thinking { thinking } => {
                    reasoning.push_str(thinking);
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    });
                }
            }
        }

        let message = Message {
            role: anthropic_response.role.clone(),
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            // Carry reasoning so the adapter can re-emit it as a `thinking`
            // block on the next request. Moonshot/Kimi's anthropic-compat
            // gateway 400s on assistant tool-call turns missing reasoning_content
            // when thinking is enabled.
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning.clone())
            },
        };

        let finish_reason = anthropic_response.stop_reason.clone();

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens
                    + anthropic_response.usage.output_tokens,
                cache_read_tokens: anthropic_response
                    .usage
                    .cache_read_input_tokens
                    .unwrap_or(0),
                cache_creation_tokens: anthropic_response
                    .usage
                    .cache_creation_input_tokens
                    .unwrap_or(0),
            },
            raw: anthropic_response,
            reasoning_content: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
            finish_reason,
        })
    }
}

/// Consume an Anthropic Messages SSE stream and assemble a full `AnthropicResponse`.
///
/// Non-streaming requests to some Anthropic-compatible gateways (notably z.ai) get
/// their TCP connection RST when server-side generation exceeds ~60s. Streaming keeps
/// bytes flowing so the gateway leaves the connection open.
#[derive(Default)]
struct PartialBlock {
    kind: BlockKind,
    text: String,
    thinking: String,
    tool_id: String,
    tool_name: String,
    tool_input_json: String,
}

#[derive(Default, PartialEq)]
enum BlockKind {
    #[default]
    Unknown,
    Text,
    Thinking,
    ToolUse,
}

async fn collect_sse_stream(
    response: reqwest::Response,
    model_id: &str,
) -> Result<AnthropicResponse, CompletionError> {
    let mut resp = AnthropicResponse {
        id: String::new(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: Vec::new(),
        model: model_id.to_string(),
        stop_reason: None,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        },
    };
    let mut blocks: BTreeMap<usize, PartialBlock> = BTreeMap::new();

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(ProviderError::Request)?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // SSE events are separated by \n\n. Drain complete events.
        while let Some(idx) = buffer.find("\n\n") {
            let event_raw = buffer[..idx].to_string();
            buffer.drain(..idx + 2);
            process_sse_event(&event_raw, &mut resp, &mut blocks)?;
        }
    }

    // Flush any trailing event (rare, but handle gracefully)
    if !buffer.trim().is_empty() {
        let leftover = std::mem::take(&mut buffer);
        process_sse_event(&leftover, &mut resp, &mut blocks)?;
    }

    // Assemble final content blocks in index order
    resp.content = blocks
        .into_values()
        .filter_map(|b| match b.kind {
            BlockKind::Text => Some(AnthropicResponseContent::Text { text: b.text }),
            BlockKind::Thinking => Some(AnthropicResponseContent::Thinking {
                thinking: b.thinking,
            }),
            BlockKind::ToolUse => {
                let input: serde_json::Value = if b.tool_input_json.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&b.tool_input_json).unwrap_or(serde_json::json!({}))
                };
                Some(AnthropicResponseContent::ToolUse {
                    id: b.tool_id,
                    name: b.tool_name,
                    input,
                })
            }
            BlockKind::Unknown => None,
        })
        .collect();

    Ok(resp)
}

fn process_sse_event(
    raw: &str,
    resp: &mut AnthropicResponse,
    blocks: &mut BTreeMap<usize, PartialBlock>,
) -> Result<(), CompletionError> {
    // Concatenate `data:` lines (Anthropic uses single-line JSON, but be defensive)
    let mut data = String::new();
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    if data.is_empty() {
        return Ok(());
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return Ok(());
    };
    let Some(event_type) = json.get("type").and_then(|v| v.as_str()) else {
        return Ok(());
    };

    match event_type {
        "message_start" => {
            if let Some(msg) = json.get("message") {
                if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                    resp.id = id.to_string();
                }
                if let Some(role) = msg.get("role").and_then(|v| v.as_str()) {
                    resp.role = role.to_string();
                }
                if let Some(model) = msg.get("model").and_then(|v| v.as_str()) {
                    resp.model = model.to_string();
                }
                if let Some(usage) = msg.get("usage") {
                    if let Some(v) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        resp.usage.input_tokens = v;
                    }
                    if let Some(v) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        resp.usage.output_tokens = v;
                    }
                    if let Some(v) = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        resp.usage.cache_read_input_tokens = Some(v);
                    }
                    if let Some(v) = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        resp.usage.cache_creation_input_tokens = Some(v);
                    }
                }
            }
        }
        "content_block_start" => {
            let index = json.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let block = blocks.entry(index).or_default();
            if let Some(cb) = json.get("content_block") {
                match cb.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        block.kind = BlockKind::Text;
                        if let Some(t) = cb.get("text").and_then(|v| v.as_str()) {
                            block.text.push_str(t);
                        }
                    }
                    Some("thinking") => {
                        block.kind = BlockKind::Thinking;
                        if let Some(t) = cb.get("thinking").and_then(|v| v.as_str()) {
                            block.thinking.push_str(t);
                        }
                    }
                    Some("tool_use") => {
                        block.kind = BlockKind::ToolUse;
                        if let Some(id) = cb.get("id").and_then(|v| v.as_str()) {
                            block.tool_id = id.to_string();
                        }
                        if let Some(name) = cb.get("name").and_then(|v| v.as_str()) {
                            block.tool_name = name.to_string();
                        }
                        // Do NOT seed tool_input_json from content_block_start.input —
                        // Anthropic streams {} there as a placeholder; the real JSON
                        // arrives as input_json_delta fragments that concatenate here.
                    }
                    _ => {}
                }
            }
        }
        "content_block_delta" => {
            let index = json.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let block = blocks.entry(index).or_default();
            if let Some(delta) = json.get("delta") {
                match delta.get("type").and_then(|v| v.as_str()) {
                    Some("text_delta") => {
                        if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                            block.text.push_str(t);
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                            block.thinking.push_str(t);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(t) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            block.tool_input_json.push_str(t);
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            // nothing to do; block data already accumulated
        }
        "message_delta" => {
            if let Some(delta) = json.get("delta") {
                if let Some(stop) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                    resp.stop_reason = Some(stop.to_string());
                }
                if let Some(seq) = delta.get("stop_sequence").and_then(|v| v.as_str()) {
                    resp.stop_sequence = Some(seq.to_string());
                }
            }
            if let Some(usage) = json.get("usage") {
                if let Some(v) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                    resp.usage.output_tokens = v;
                }
                if let Some(v) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                    if v > 0 {
                        resp.usage.input_tokens = v;
                    }
                }
            }
        }
        "message_stop" | "ping" => {}
        "error" => {
            let message = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown streaming error")
                .to_string();
            return Err(CompletionError::Provider(ProviderError::Http {
                status: 0,
                message,
            }));
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn text_msg(role: &str, text: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_string(),
            content: AnthropicContent::Text(text.to_string()),
        }
    }

    fn has_cache_control_json<T: serde::Serialize>(v: &T) -> bool {
        serde_json::to_string(v).unwrap().contains("cache_control")
    }

    /// With no hints requested, the function is a no-op and nothing serializes a
    /// `cache_control` key — i.e. byte-identical to the pre-caching wire body.
    #[test]
    fn empty_hints_is_noop() {
        let mut system = Some(AnthropicSystem::Text("sys".into()));
        let mut messages = vec![text_msg("user", "hi"), text_msg("assistant", "yo")];
        let mut tools = vec![AnthropicTool {
            name: "t".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            cache_control: None,
        }];
        apply_cache_breakpoints(
            &mut system,
            &mut messages,
            Some(&mut tools),
            &CacheHints::default(),
        );

        assert!(
            matches!(system, Some(AnthropicSystem::Text(_))),
            "system must stay a plain string"
        );
        assert!(!has_cache_control_json(&system));
        assert!(!has_cache_control_json(&messages));
        assert!(!has_cache_control_json(&tools));
    }

    /// A `Text` content block with `cache_control: None` serializes without the
    /// key (the safety property that keeps caching-off output unchanged).
    #[test]
    fn unmarked_block_omits_cache_control_key() {
        let block = AnthropicContentBlock::Text {
            text: "x".into(),
            cache_control: None,
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(
            !json.contains("cache_control"),
            "unmarked block must omit cache_control: {json}"
        );
    }

    /// `cache_system` marks the system prefix and the two tail breakpoints land
    /// on the final message and the most recent earlier user message.
    #[test]
    fn marks_system_and_tail() {
        let mut system = Some(AnthropicSystem::Text("sys".into()));
        let mut messages = vec![
            text_msg("user", "first"),    // 0: earlier user -> should be marked
            text_msg("assistant", "mid"), // 1
            text_msg("user", "last"),     // 2: final -> should be marked
        ];
        let hints = CacheHints {
            cache_system: true,
            breakpoints: vec![2],
        };
        apply_cache_breakpoints(&mut system, &mut messages, None, &hints);

        // System promoted to a marked block.
        assert!(matches!(system, Some(AnthropicSystem::Blocks(_))));
        assert!(has_cache_control_json(&system));
        // Final message + earlier user marked; the assistant in between is not.
        assert!(
            has_cache_control_json(&messages[2]),
            "final message must be marked"
        );
        assert!(
            has_cache_control_json(&messages[0]),
            "penultimate user must be marked"
        );
        assert!(
            !has_cache_control_json(&messages[1]),
            "the assistant between them must not be marked"
        );
    }

    /// `mark_last_block` never marks a `thinking` block; it marks the last
    /// cache-eligible (text/tool) block instead.
    #[test]
    fn never_marks_thinking_block() {
        let mut msg = AnthropicMessage {
            role: "assistant".into(),
            content: AnthropicContent::Blocks(vec![
                AnthropicContentBlock::Text {
                    text: "reasoned answer".into(),
                    cache_control: None,
                },
                AnthropicContentBlock::Thinking {
                    thinking: "secret".into(),
                },
            ]),
        };
        mark_last_block(&mut msg);
        let json = serde_json::to_string(&msg).unwrap();
        // The text block carries the marker; the thinking block must not.
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            let text_marked = matches!(
                &blocks[0],
                AnthropicContentBlock::Text {
                    cache_control: Some(_),
                    ..
                }
            );
            let thinking_marked = matches!(&blocks[1], AnthropicContentBlock::Thinking { .. })
                && json.matches("cache_control").count() == 1;
            assert!(text_marked, "text block should be marked");
            assert!(
                thinking_marked,
                "exactly one marker, on the text block, not the thinking block"
            );
        } else {
            panic!("expected blocks");
        }
    }

    /// Empty hints leave a plain-string system untouched (no promotion to blocks).
    #[test]
    fn no_system_falls_back_to_tools() {
        let mut system: Option<AnthropicSystem> = None;
        let mut messages = vec![text_msg("user", "hi")];
        let mut tools = vec![AnthropicTool {
            name: "t".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            cache_control: None,
        }];
        let hints = CacheHints {
            cache_system: true,
            breakpoints: vec![],
        };
        apply_cache_breakpoints(&mut system, &mut messages, Some(&mut tools), &hints);
        // With no system prompt, the static-prefix marker lands on the last tool.
        assert!(
            has_cache_control_json(&tools),
            "tool must carry the static-prefix marker when there is no system"
        );
    }
}
