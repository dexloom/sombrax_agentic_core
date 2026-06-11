//! OpenAI client and completion model

use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::Client;
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default OpenAI API base URL
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI client configuration
#[derive(Clone)]
pub struct OpenAIClient {
    inner: Arc<OpenAIClientInner>,
}

struct OpenAIClientInner {
    http_client: Client,
    api_key: String,
    base_url: String,
    organization: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    repetition_penalty: Option<f64>,
    max_tokens: Option<u64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    parallel_tool_calls: Option<bool>,
    /// Explicit thinking-mode toggle for OpenAI-compatible servers
    /// fronting thinking-by-default models. `Some(false)` injects
    /// `chat_template_kwargs.enable_thinking=false` so the server's
    /// Jinja template suppresses the reasoning channel and the model
    /// emits regular `content`. `None` is a passthrough.
    enable_thinking: Option<bool>,
}

impl OpenAIClient {
    /// Create a new OpenAI client from environment variable
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| ProviderError::EnvVarNotSet("OPENAI_API_KEY".to_string()))?;

        let mut builder = OpenAIClientBuilder::new(&api_key);

        // Optionally load organization from env
        if let Ok(org) = env::var("OPENAI_ORGANIZATION") {
            builder = builder.organization(&org);
        }

        Ok(builder.build())
    }

    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> OpenAICompletionModel {
        OpenAICompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for OpenAI client configuration
pub struct OpenAIClientBuilder {
    api_key: String,
    base_url: String,
    organization: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    repetition_penalty: Option<f64>,
    max_tokens: Option<u64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    parallel_tool_calls: Option<bool>,
    enable_thinking: Option<bool>,
}

impl OpenAIClientBuilder {
    /// Create a new builder with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            organization: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            max_tokens: None,
            frequency_penalty: None,
            presence_penalty: None,
            parallel_tool_calls: Some(true),
            enable_thinking: None,
        }
    }

    /// Set custom base URL (useful for Azure OpenAI or proxies)
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set organization ID
    pub fn organization(mut self, org: &str) -> Self {
        self.organization = Some(org.to_string());
        self
    }

    /// Set temperature (clamped to 0.0-2.0)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp.clamp(0.0, 2.0));
        self
    }

    /// Set top_p sampling parameter
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set top_k sampling cap. Not part of canonical OpenAI but
    /// recognised by OpenAI-compatible local servers.
    pub fn top_k(mut self, k: u64) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set repetition penalty (multiplicative; 1.0 = no penalty).
    /// Recognised by OpenAI-compatible local servers (mlx-lm forks,
    /// vllm, sglang). Clamped to a safe range.
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty.clamp(0.5, 2.0));
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set frequency penalty (-2.0 to 2.0)
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Set presence penalty (-2.0 to 2.0)
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Enable/disable parallel tool calls (default: true)
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Set explicit `enable_thinking` for OpenAI-compatible servers
    /// fronting thinking-by-default models (GLM-5.1, Qwen-thinking, …).
    /// `Some(false)` adds `chat_template_kwargs.enable_thinking=false`
    /// to the request body so the server's Jinja chat template
    /// suppresses the reasoning channel and the model emits regular
    /// `content`. `None` is a passthrough — server-side default applies.
    pub fn enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = Some(enabled);
        self
    }

    /// Build the client
    pub fn build(self) -> OpenAIClient {
        OpenAIClient {
            inner: Arc::new(OpenAIClientInner {
                http_client: build_http_client(),
                api_key: self.api_key,
                base_url: self.base_url,
                organization: self.organization,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                repetition_penalty: self.repetition_penalty,
                max_tokens: self.max_tokens,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                parallel_tool_calls: self.parallel_tool_calls,
                enable_thinking: self.enable_thinking,
            }),
        }
    }
}

/// OpenAI completion model
#[derive(Clone)]
pub struct OpenAICompletionModel {
    client: OpenAIClient,
    model_id: String,
}

impl OpenAICompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "openai"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "openai"))]
    pub async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<OpenAIResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Build messages
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.push(OpenAIMessage {
                role: msg.role.clone(),
                content: if msg.content.is_empty() {
                    None
                } else {
                    Some(msg.content.clone())
                },
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| OpenAIToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: msg.tool_call_id.clone(),
                name: None,
                // Replay reasoning on follow-up turns. OpenAI's canonical
                // o1 spec drops chain-of-thought between turns, but the
                // MiniMax/DeepSeek/GLM/Qwen3 dialect templates were
                // designed assuming prior `<think>` blocks ARE preserved
                // — without them the model retraces from scratch every
                // turn and the tool-use chain stalls after 3–4 calls
                // (verified empirically: dropping reasoning replay made
                // the audit freeze on turn 6 instead of turn 10).
                // Single canonical field name; servers that read
                // `reasoning_content` accept it via the `serde(alias)`
                // on the inbound side and bridge in their own dialect
                // shape_request if needed.
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
                    .map(|t| OpenAITool {
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(normalize_tool_parameters(t.parameters.clone())),
                        },
                    })
                    .collect(),
            )
        };

        // Inject Jinja chat-template kwargs for thinking-by-default
        // models served behind an OpenAI-compatible endpoint (mlx_fun,
        // vllm, sglang, llama.cpp `--jinja`). When the user has opted
        // out of thinking, send `{"enable_thinking": false}` so the
        // server's chat template suppresses the reasoning channel and
        // the model emits `content` rather than `reasoning`-only output.
        let chat_template_kwargs = inner
            .enable_thinking
            .map(|enabled| serde_json::json!({ "enable_thinking": enabled }));

        let openai_request = OpenAIRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request.max_tokens.or(inner.max_tokens),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            tools,
            tool_choice: None,
            parallel_tool_calls: inner.parallel_tool_calls,
            user: None,
            chat_template_kwargs,
            // Always request SSE streaming. The reassembled response is
            // identical to the non-streaming JSON shape, but streaming lets
            // local servers (mlx_fun, vllm) flush deltas instead of buffering
            // the entire response — which on thinking-by-default models can
            // span many minutes per turn and look like a stall on non-streaming.
            stream: Some(true),
            stream_options: Some(OpenAIStreamOptions {
                include_usage: true,
            }),
        };

        let url = format!("{}/chat/completions", inner.base_url);

        // DEBUG: dump the exact JSON body we're about to POST.
        // Set SAC_DUMP_REQUESTS=1 to capture every request into
        // /tmp/lms/sac_request_<ts>.json (or SAC_DUMP_REQUESTS_DIR
        // for a custom directory). Use when the upstream server has
        // no server-side request log (e.g. LM Studio) and you need
        // to compare wire bodies across providers.
        if std::env::var("SAC_DUMP_REQUESTS").is_ok() {
            if let Ok(body_json) = serde_json::to_string_pretty(&openai_request) {
                let dir = std::env::var("SAC_DUMP_REQUESTS_DIR")
                    .unwrap_or_else(|_| "/tmp/lms".to_string());
                let _ = std::fs::create_dir_all(&dir);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let path = format!("{}/sac_request_{}.json", dir, ts);
                let _ = std::fs::write(&path, body_json.as_bytes());
                tracing::info!("SAC REQUEST DUMP → {} ({} bytes)", path, body_json.len());
            }
        }

        let mut req = inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", inner.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        // Add organization header if set
        if let Some(org) = &inner.organization {
            req = req.header("OpenAI-Organization", org);
        }

        let response = req
            .json(&openai_request)
            .send()
            .instrument(info_span!("openai_http_request"))
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

        let openai_response = collect_openai_sse_stream(response, &self.model_id).await?;

        // Extract first choice
        let choice = openai_response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        // Convert tool calls.
        //
        // Defensive Kimi-dialect splitter: some OpenAI-compatible servers
        // fronting Kimi-K2.6 quantizations (Q3, REAP-pruned variants) emit
        // multiple tool calls packed into a single tool_call's `arguments`
        // string, separated by inner `functions.NAME:N<|tool_call_argument_begin|>`
        // markers. The mlx_fun-side parser (`kimi_k26_tool_parser.py`) is
        // the primary fix; this is the secondary fix for any other server
        // that exhibits the same dialect bug. If a tool call's arguments
        // contain the inner separator, split into multiple ToolCall entries
        // and infer each subsequent call's name from its prefix.
        let tool_calls = choice.message.tool_calls.as_ref().map(|calls| {
            let mut out: Vec<ToolCall> = Vec::with_capacity(calls.len());
            for tc in calls {
                let id = tc.id.clone();
                let name = tc.function.name.clone();
                let args = tc.function.arguments.clone();
                let split = split_packed_kimi_args(&id, &name, &args);
                out.extend(split);
            }
            out
        });

        // Keep reasoning and content as separate channels. Reasoning is
        // surfaced on `Message.reasoning` and `CompletionResponse.reasoning_content`
        // so downstream consumers can display/analyze the chain-of-thought
        // independently of the model's user-facing answer.
        //
        // Earlier versions of this code copied `reasoning_text` into `content`
        // when `content` was empty, on the theory that an "empty turn" would
        // stall the agent. That behavior leaked chain-of-thought into the
        // structured output stream: when a thinking model emits a tool-call
        // turn (content="", tool_calls=[...], reasoning="..."), the reasoning
        // got captured into the assistant text aggregation downstream and
        // displaced the model's final structured answer.
        //
        // If a turn has neither real content nor tool_calls but does have
        // reasoning, the `response_validation` min_length check in `retry.rs`
        // will trigger a retry — the correct behavior, since the model
        // thought-only and didn't produce a useful response.
        let reasoning_text = choice.message.reasoning.clone();

        let content = choice.message.content.clone().unwrap_or_default();

        let message = Message {
            role: choice.message.role.clone(),
            content,
            tool_calls,
            tool_call_id: None,
            reasoning: reasoning_text.clone(),
        };

        // Extract cache tokens if available
        let cache_read_tokens = openai_response
            .usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);

        let finish_reason = choice.finish_reason.clone();

        Ok(CompletionResponse {
            message,
            usage: Usage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
                cache_read_tokens,
                cache_creation_tokens: 0,
            },
            raw: openai_response,
            reasoning_content: reasoning_text,
            finish_reason,
        })
    }
}

/// Split a Kimi-dialect packed tool call into multiple `ToolCall` entries.
///
/// Normalize a tool's parameters JSON Schema so strict OpenAI-spec
/// validators (LM Studio, some self-hosted gateways) accept it.
///
/// schemars produces `{"type":"object"}` (no `properties` key) for tools
/// whose `Args` struct has zero fields — e.g. `TodoReadArgs`. Hosted
/// OpenAI and OpenRouter accept this lenient form, but LM Studio rejects
/// it with `HTTP 400 invalid_type at function.parameters.properties`.
///
/// Coerce empty/missing `properties` to `{}` for any object schema. Pass
/// through anything that isn't an object schema unchanged.
fn normalize_tool_parameters(mut value: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = value.as_object_mut() {
        let is_object_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| s == "object")
            .unwrap_or(false);
        if is_object_type && !obj.contains_key("properties") {
            obj.insert("properties".to_string(), serde_json::json!({}));
        }
    }
    value
}

/// Some OpenAI-compatible servers fronting Kimi-K2.6 (especially Q3 and
/// REAP-pruned Solidity-tuned variants) emit multiple tool calls inside a
/// single OpenAI `tool_calls[i].function.arguments` string, separated by
/// inner `functions.NAME:N<|tool_call_argument_begin|>` markers. This
/// helper detects that pattern and splits one packed call into N ToolCalls
/// (the first keeps the outer tool's `id`/`name`; subsequent ones derive
/// `name` from the inner marker's `NAME`). Returns a single-element vec
/// with the original call when no packing is detected.
fn split_packed_kimi_args(id: &str, name: &str, args: &str) -> Vec<ToolCall> {
    const SEP_CORE: &str = "<|tool_call_argument_begin|>";
    if !args.contains(SEP_CORE) {
        return vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }];
    }
    // Look for ``functions.NAME:N<|tool_call_argument_begin|>`` boundaries.
    // We don't pull in the regex crate just for this — a hand parse is fine.
    let mut boundaries: Vec<(usize, usize, String)> = Vec::new(); // (start, end, name)
    let bytes = args.as_bytes();
    let sep_bytes = SEP_CORE.as_bytes();
    let mut i = 0;
    while i + sep_bytes.len() <= bytes.len() {
        if bytes[i..i + sep_bytes.len()] == *sep_bytes {
            // Walk back to find the start of `[functions.]NAME:N`
            let mut start = i;
            // skip back over digits (the :N part)
            while start > 0 && bytes[start - 1].is_ascii_digit() {
                start -= 1;
            }
            // expect a ':'
            if start == 0 || bytes[start - 1] != b':' {
                i += 1;
                continue;
            }
            start -= 1; // consume ':'
                        // walk back over the name
            let name_end = start;
            while start > 0
                && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
            {
                start -= 1;
            }
            // optional `functions.` prefix
            let prefix_start = if start >= "functions.".len()
                && &args[start - "functions.".len()..start] == "functions."
            {
                start - "functions.".len()
            } else {
                start
            };
            let inner_name = args[start..name_end].to_string();
            if inner_name.is_empty() {
                i += 1;
                continue;
            }
            boundaries.push((prefix_start, i + sep_bytes.len(), inner_name));
            i += sep_bytes.len();
        } else {
            i += 1;
        }
    }
    if boundaries.is_empty() {
        return vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args.to_string(),
        }];
    }
    // Build the calls. The outer (passed-in) call's args run from 0 to the
    // first boundary's start (or to end if no boundary appears in args).
    let mut out: Vec<ToolCall> = Vec::with_capacity(boundaries.len() + 1);
    let first_args_end = boundaries[0].0;
    out.push(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: args[..first_args_end].trim().to_string(),
    });
    for idx in 0..boundaries.len() {
        let (_, args_start, inner_name) = &boundaries[idx];
        let args_end = if idx + 1 < boundaries.len() {
            boundaries[idx + 1].0
        } else {
            args.len()
        };
        out.push(ToolCall {
            id: format!("{id}_split_{}", idx + 1),
            name: inner_name.clone(),
            arguments: args[*args_start..args_end].trim().to_string(),
        });
    }
    out
}

/// Consume an OpenAI Chat-Completions SSE stream and assemble a full
/// `OpenAIResponse` matching the non-streaming JSON shape.
///
/// The wire format is `data: {…json…}\n\n` repeating, terminated by
/// `data: [DONE]`. Each chunk has `choices[0].delta` carrying incremental
/// `content`, `reasoning`/`reasoning_content`, and `tool_calls[i].function`
/// (name + partial JSON arguments) fragments. The final chunk before
/// `[DONE]` typically carries `finish_reason`; usage arrives only when
/// `stream_options.include_usage` was set.
///
/// Streaming exists primarily because local mlx_fun / vllm servers buffer
/// the entire non-streaming response until end-of-turn, which on
/// thinking-by-default models can take many minutes and looks like a stall.
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

async fn collect_openai_sse_stream(
    response: reqwest::Response,
    model_id: &str,
) -> Result<OpenAIResponse, CompletionError> {
    let mut resp_id = String::new();
    let mut resp_object = "chat.completion".to_string();
    let mut resp_created: u64 = 0;
    let mut resp_model = model_id.to_string();
    let mut system_fingerprint: Option<String> = None;

    let mut role = String::from("assistant");
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: BTreeMap<u64, PartialToolCall> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage = OpenAIUsage::default();

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    'outer: while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(ProviderError::Request)?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // SSE events are separated by \n\n. Drain complete events.
        while let Some(idx) = buffer.find("\n\n") {
            let event_raw = buffer[..idx].to_string();
            buffer.drain(..idx + 2);
            if process_openai_sse_event(
                &event_raw,
                &mut resp_id,
                &mut resp_object,
                &mut resp_created,
                &mut resp_model,
                &mut system_fingerprint,
                &mut role,
                &mut content,
                &mut reasoning,
                &mut tool_calls,
                &mut finish_reason,
                &mut usage,
            )? {
                // `[DONE]` sentinel — stop draining.
                break 'outer;
            }
        }
    }

    // Flush any trailing partial event (defensive).
    if !buffer.trim().is_empty() {
        let leftover = std::mem::take(&mut buffer);
        let _ = process_openai_sse_event(
            &leftover,
            &mut resp_id,
            &mut resp_object,
            &mut resp_created,
            &mut resp_model,
            &mut system_fingerprint,
            &mut role,
            &mut content,
            &mut reasoning,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
        );
    }

    // Assemble tool calls in index order.
    let assembled_tool_calls: Vec<OpenAIToolCall> = tool_calls
        .into_values()
        .filter(|tc| !tc.id.is_empty() || !tc.name.is_empty() || !tc.arguments.is_empty())
        .map(|tc| OpenAIToolCall {
            id: tc.id,
            call_type: "function".to_string(),
            function: OpenAIFunctionCall {
                name: tc.name,
                arguments: tc.arguments,
            },
        })
        .collect();

    let message = OpenAIMessage {
        role,
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls: if assembled_tool_calls.is_empty() {
            None
        } else {
            Some(assembled_tool_calls)
        },
        tool_call_id: None,
        name: None,
        reasoning: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
    };

    Ok(OpenAIResponse {
        id: resp_id,
        object: resp_object,
        created: resp_created,
        model: resp_model,
        choices: vec![OpenAIChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
        system_fingerprint,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_openai_sse_event(
    raw: &str,
    resp_id: &mut String,
    resp_object: &mut String,
    resp_created: &mut u64,
    resp_model: &mut String,
    system_fingerprint: &mut Option<String>,
    role: &mut String,
    content: &mut String,
    reasoning: &mut String,
    tool_calls: &mut BTreeMap<u64, PartialToolCall>,
    finish_reason: &mut Option<String>,
    usage: &mut OpenAIUsage,
) -> Result<bool, CompletionError> {
    // Concatenate `data:` lines.
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
        return Ok(false);
    }
    if data == "[DONE]" {
        return Ok(true);
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return Ok(false);
    };

    if let Some(v) = json.get("id").and_then(|v| v.as_str()) {
        *resp_id = v.to_string();
    }
    if let Some(v) = json.get("object").and_then(|v| v.as_str()) {
        *resp_object = v.to_string();
    }
    if let Some(v) = json.get("created").and_then(|v| v.as_u64()) {
        *resp_created = v;
    }
    if let Some(v) = json.get("model").and_then(|v| v.as_str()) {
        *resp_model = v.to_string();
    }
    if let Some(v) = json.get("system_fingerprint").and_then(|v| v.as_str()) {
        *system_fingerprint = Some(v.to_string());
    }

    // Usage may arrive on the same chunk as a delta, on a final usage-only
    // chunk (when stream_options.include_usage=true), or never (servers
    // that don't honor stream_options). Read whenever present.
    if let Some(u) = json.get("usage") {
        if let Some(v) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
            usage.prompt_tokens = v;
        }
        if let Some(v) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
            usage.completion_tokens = v;
        }
        if let Some(v) = u.get("total_tokens").and_then(|v| v.as_u64()) {
            usage.total_tokens = v;
        }
    }

    let Some(choices) = json.get("choices").and_then(|v| v.as_array()) else {
        return Ok(false);
    };
    let Some(choice) = choices.first() else {
        return Ok(false);
    };

    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
        *finish_reason = Some(fr.to_string());
    }

    let Some(delta) = choice.get("delta") else {
        return Ok(false);
    };
    if let Some(r) = delta.get("role").and_then(|v| v.as_str()) {
        *role = r.to_string();
    }
    if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
        content.push_str(c);
    }
    // Accept either delta field name from the server — both are merged
    // into our single `reasoning` accumulator.
    if let Some(r) = delta
        .get("reasoning")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()))
    {
        reasoning.push_str(r);
    }
    if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
            let entry = tool_calls.entry(idx).or_insert_with(|| PartialToolCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                if !id.is_empty() {
                    entry.id = id.to_string();
                }
            }
            if let Some(func) = tc.get("function") {
                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        entry.name = name.to_string();
                    }
                }
                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                    entry.arguments.push_str(args);
                }
            }
        }
    }

    Ok(false)
}
