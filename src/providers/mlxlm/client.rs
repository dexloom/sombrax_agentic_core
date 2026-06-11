//! MLX-LM client and completion model
//!
//! Provides access to local MLX-LM server with support for custom chat templates.

use std::sync::Arc;

use reqwest::Client;
use tracing::{info_span, instrument, Instrument};

use super::types::*;
use crate::providers::error::{CompletionError, ProviderError};
use crate::providers::http::build_http_client;
use crate::providers::zai::client::{
    CompletionRequest, CompletionResponse, Message, ToolCall, Usage,
};

/// Default MLX-LM server base URL (includes /v1 for OpenAI-compatible endpoint)
const DEFAULT_BASE_URL: &str = "http://localhost:8080/v1";

/// Default max tokens for MLX-LM
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Maximum cap for max_tokens on local MLX models.
/// The agent framework may request 32K+ tokens, but local models can't efficiently
/// generate that many tokens. This cap prevents KV cache allocation issues.
const MAX_TOKENS_CAP: u64 = 8192;

/// MLX-LM client configuration
#[derive(Clone)]
pub struct MlxLmClient {
    inner: Arc<MlxLmClientInner>,
}

struct MlxLmClientInner {
    http_client: Client,
    base_url: String,
    chat_template: ChatTemplate,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    repetition_penalty: Option<f64>,
    repetition_context_size: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    min_p: Option<f64>,
    stop_sequences: Option<Vec<String>>,
}

impl MlxLmClient {
    /// Create a completion model for a specific model ID
    pub fn completion_model(&self, model_id: &str) -> MlxLmCompletionModel {
        MlxLmCompletionModel {
            client: self.clone(),
            model_id: model_id.to_string(),
        }
    }
}

/// Builder for MLX-LM client configuration
pub struct MlxLmClientBuilder {
    base_url: String,
    chat_template: ChatTemplate,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    max_tokens: Option<u64>,
    repetition_penalty: Option<f64>,
    repetition_context_size: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    min_p: Option<f64>,
    stop_sequences: Option<Vec<String>>,
}

impl Default for MlxLmClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MlxLmClientBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            chat_template: ChatTemplate::default(),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            repetition_penalty: None,
            repetition_context_size: None,
            frequency_penalty: None,
            presence_penalty: None,
            min_p: None,
            stop_sequences: None,
        }
    }

    /// Set custom base URL for the MLX-LM server
    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set the chat template format to use
    ///
    /// Different models may require different chat template formats.
    /// The template affects how messages are serialized and how responses are parsed.
    pub fn chat_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    /// Auto-detect and set the chat template based on model name
    ///
    /// This is a convenience method that calls `ChatTemplate::from_model_name()`
    /// to automatically determine the appropriate chat template format.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let client = MlxLmClientBuilder::new()
    ///     .auto_chat_template("zai-org/GLM-4.7")  // Detects GLM template
    ///     .build();
    /// ```
    pub fn auto_chat_template(mut self, model: &str) -> Self {
        self.chat_template = ChatTemplate::from_model_name(model);
        self
    }

    /// Auto-detect and set the chat template based on Jinja template content
    ///
    /// Use this when you have access to the raw chat template file content
    /// (e.g., from a HuggingFace model's chat_template.jinja file).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let template_content = "[gMASK]<sop>{% for m in messages %}...";
    /// let client = MlxLmClientBuilder::new()
    ///     .auto_chat_template_from_content(template_content)  // Detects GLM template
    ///     .build();
    /// ```
    pub fn auto_chat_template_from_content(mut self, content: &str) -> Self {
        self.chat_template = ChatTemplate::from_template_content(content);
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

    /// Set repetition penalty (clamped to 1.0-2.0)
    ///
    /// Controls how much the model penalizes repeating the same tokens.
    /// - `1.0` = no penalty (default behavior)
    /// - `1.1-1.2` = mild penalty, good for most use cases
    /// - `1.3-1.5` = moderate penalty, reduces repetition significantly
    /// - `1.5+` = strong penalty, may affect output quality
    ///
    /// **Recommended for code generation:** `1.15`
    /// **Recommended for avoiding loops:** `1.2-1.3`
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let client = MlxLmClientBuilder::new()
    ///     .repetition_penalty(1.15)  // Mild anti-repetition
    ///     .build();
    /// ```
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        // Clamp to reasonable range: 1.0 (no penalty) to 2.0 (maximum penalty)
        // Values below 1.0 would encourage repetition, which is rarely useful
        self.repetition_penalty = Some(penalty.clamp(1.0, 2.0));
        self
    }

    /// Set repetition context size (how many recent tokens to check for repeats)
    ///
    /// Controls the window of tokens considered for `repetition_penalty`.
    /// - `-1` = full context (check all tokens)
    /// - `0` = disabled
    /// - positive = last N tokens (e.g., 64, 256)
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

    /// Add recommended stop sequences for ChatML template
    ///
    /// Adds standard ChatML delimiters and common repetition patterns.
    /// These help prevent the model from:
    /// - Generating infinite loops
    /// - Continuing beyond the assistant's turn
    /// - Producing excessive whitespace
    ///
    /// Recommended for IQuest, Qwen, and other ChatML-based models.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let client = MlxLmClientBuilder::new()
    ///     .chat_template(sombrax_agentic_core::providers::mlxlm::ChatTemplate::ChatML)
    ///     .with_chatml_stop_sequences()  // Adds recommended stops
    ///     .build();
    /// ```
    pub fn with_chatml_stop_sequences(mut self) -> Self {
        let sequences = vec![
            // ChatML template markers
            "<|im_end|>".to_string(),
            "<|im_start|>".to_string(),
            // Common repetition patterns
            "\n\n\n\n".to_string(), // Excessive newlines
            "####".to_string(),     // Repetitive tokens
        ];

        match &mut self.stop_sequences {
            Some(existing) => existing.extend(sequences),
            None => self.stop_sequences = Some(sequences),
        }
        self
    }

    /// Configure recommended anti-loop settings for code generation
    ///
    /// This is a convenience method that sets multiple parameters to prevent
    /// the model from entering repetition loops. It configures:
    /// - `temperature`: 0.7 (increased from typical 0.4 to add randomness)
    /// - `top_p`: 0.95 (nucleus sampling)
    /// - `top_k`: 40 (limits token pool)
    /// - `repetition_penalty`: 1.15 (mild penalty for code generation)
    ///
    /// These values are optimized for code generation with local models like
    /// IQuest, Qwen, and other instruction-tuned models.
    ///
    /// **Note:** This does NOT add stop sequences. Use in combination with
    /// `.with_chatml_stop_sequences()` and `.with_anti_repetition_stops()`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let client = MlxLmClientBuilder::new()
    ///     .with_anti_loop_config()           // Sets sampling params
    ///     .with_chatml_stop_sequences()      // Adds ChatML stops
    ///     .with_anti_repetition_stops()      // Adds pattern detection
    ///     .build();
    /// ```
    pub fn with_anti_loop_config(mut self) -> Self {
        self.temperature = Some(0.7);
        self.top_p = Some(0.95);
        self.top_k = Some(40);
        self.repetition_penalty = Some(1.15);
        self.repetition_context_size = Some(256);
        self
    }

    /// Add anti-repetition stop sequences
    ///
    /// Adds patterns that commonly appear when models enter repetition loops.
    /// Includes detection for:
    /// - Repeated characters (EEEE, ====, ----, etc.)
    /// - Excessive whitespace (spaces, newlines, tabs)
    /// - Other repetitive patterns
    ///
    /// Use this in addition to other stop sequences.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sombrax_agentic_core::providers::MlxLmClientBuilder;
    ///
    /// let client = MlxLmClientBuilder::new()
    ///     .with_anti_repetition_stops()
    ///     .build();
    /// ```
    pub fn with_anti_repetition_stops(mut self) -> Self {
        let sequences = vec![
            // Excessive whitespace
            "\n\n\n\n".to_string(),     // Excessive newlines (4+)
            "        ".to_string(),     // Excessive spaces (8+)
            "            ".to_string(), // Even more spaces (12+)
            "\t\t\t\t".to_string(),     // Excessive tabs
            // Repetitive characters
            "EEEE".to_string(), // Known failure token
            "====".to_string(), // Repetitive equals
            "----".to_string(), // Repetitive dashes
            "####".to_string(), // Repetitive hashes
            "****".to_string(), // Repetitive asterisks
            "....".to_string(), // Repetitive dots
            ",,,,".to_string(), // Repetitive commas
        ];

        match &mut self.stop_sequences {
            Some(existing) => existing.extend(sequences),
            None => self.stop_sequences = Some(sequences),
        }
        self
    }

    /// Build the client
    pub fn build(self) -> MlxLmClient {
        MlxLmClient {
            inner: Arc::new(MlxLmClientInner {
                http_client: build_http_client(),
                base_url: self.base_url,
                chat_template: self.chat_template,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                max_tokens: self.max_tokens,
                repetition_penalty: self.repetition_penalty,
                repetition_context_size: self.repetition_context_size,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                min_p: self.min_p,
                stop_sequences: self.stop_sequences,
            }),
        }
    }
}

/// MLX-LM completion model
#[derive(Clone)]
pub struct MlxLmCompletionModel {
    client: MlxLmClient,
    model_id: String,
}

impl MlxLmCompletionModel {
    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        "mlxlm"
    }

    /// Send a completion request
    #[instrument(skip(self, request), fields(model = %self.model_id, provider = "mlxlm"))]
    pub async fn completion(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let inner = &self.client.inner;

        // Cap max_tokens for local models to prevent KV cache allocation issues
        if let Some(max_tokens) = request.max_tokens {
            if max_tokens > MAX_TOKENS_CAP {
                tracing::debug!(
                    requested = max_tokens,
                    capped = MAX_TOKENS_CAP,
                    "Capping max_tokens for local MLX model"
                );
                request.max_tokens = Some(MAX_TOKENS_CAP);
            }
        }

        // Build the request based on chat template
        let mut mlxlm_request = match &inner.chat_template {
            ChatTemplate::OpenAI => self.build_openai_request(&request, inner),
            ChatTemplate::Minimax => self.build_minimax_request(&request, inner),
            ChatTemplate::Minimax25 => self.build_openai_request(&request, inner),
            ChatTemplate::ChatML => self.build_chatml_request(&request, inner),
            ChatTemplate::Qwen35 => self.build_qwen35_request(&request, inner),
            ChatTemplate::GLM => self.build_glm_request(&request, inner),
            ChatTemplate::Gemma => self.build_gemma_request(&request, inner),
        };

        // Cap max_tokens for all local MLX models to prevent server hangs
        if let Some(max_tokens) = mlxlm_request.max_tokens {
            if max_tokens > MAX_TOKENS_CAP {
                tracing::debug!(
                    requested = max_tokens,
                    capped = MAX_TOKENS_CAP,
                    "Capping max_tokens for local MLX model"
                );
                mlxlm_request.max_tokens = Some(MAX_TOKENS_CAP);
            }
        }

        let url = format!("{}/chat/completions", inner.base_url);

        // Log request size for debugging slow local inference
        if let Ok(request_json) = serde_json::to_string(&mlxlm_request) {
            tracing::debug!(
                request_bytes = request_json.len(),
                messages = mlxlm_request.messages.len(),
                tools = mlxlm_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                max_tokens = ?mlxlm_request.max_tokens,
                "MLX-LM request"
            );
            // Per-turn dump: SAC_DUMP_REQUESTS=1 → /tmp/sac_mlxlm_request_<ts>.json
            // captures the exact JSON body about to be POSTed to mlx_lm. Use
            // when you're pointing SAC at a bare mlx_lm.server (no mlx_fun
            // wrapper, so no server-side hook) and need to inspect what SAC
            // is sending across turns.
            if std::env::var("SAC_DUMP_REQUESTS").is_ok() {
                if let Ok(pretty) = serde_json::to_string_pretty(&mlxlm_request) {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    let path = format!("/tmp/sac_mlxlm_request_{}.json", ts);
                    let _ = std::fs::write(&path, pretty.as_bytes());
                    tracing::info!("SAC MLXLM REQUEST DUMP → {} ({} bytes)", path, pretty.len());
                }
            }
            // Legacy single-file dump (overwrites each turn)
            if let Err(e) = std::fs::write("/tmp/mlx_request_dump.json", &request_json) {
                tracing::warn!("Failed to dump MLX request: {}", e);
            }
        }

        let response = inner
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&mlxlm_request)
            .send()
            .instrument(info_span!("mlxlm_http_request"))
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

        let mlxlm_response: MlxLmResponse =
            response.json().await.map_err(ProviderError::Request)?;

        // Parse response based on chat template
        match &inner.chat_template {
            ChatTemplate::OpenAI => self.parse_openai_response(mlxlm_response),
            ChatTemplate::Minimax | ChatTemplate::Minimax25 => {
                self.parse_minimax_response(mlxlm_response)
            }
            ChatTemplate::ChatML => self.parse_chatml_response(mlxlm_response),
            ChatTemplate::Qwen35 => self.parse_qwen35_response(mlxlm_response),
            ChatTemplate::GLM => self.parse_glm_response(mlxlm_response),
            ChatTemplate::Gemma => self.parse_gemma_response(mlxlm_response),
        }
    }

    /// Build request in standard OpenAI-compatible format
    fn build_openai_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(MlxLmMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                tool_calls: None,
                tool_call_id: None,
                ..Default::default()
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.push(MlxLmMessage {
                role: msg.role.clone(),
                content: if msg.content.is_empty() {
                    None
                } else {
                    Some(msg.content.clone())
                },
                tool_calls: msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| MlxLmToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: MlxLmFunctionCall {
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

        // Build tools
        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .tools
                    .iter()
                    .map(|t| MlxLmTool {
                        tool_type: "function".to_string(),
                        function: MlxLmFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools,
            tool_choice: None,
        }
    }

    /// Build request for Minimax chat template format
    ///
    /// Minimax M2.1 uses special delimiters and role names:
    /// - Roles: system -> system, user -> user, assistant -> ai, tool -> user
    /// - Tool calls use <minimax:tool_call> XML format embedded in content
    /// - Tool responses are wrapped in <response> tags with "user" role
    fn build_minimax_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(MlxLmMessage {
                role: "system".to_string(),
                content: Some(preamble.clone()),
                tool_calls: None,
                tool_call_id: None,
                ..Default::default()
            });
        }

        // Convert messages - Minimax M2.1 uses "ai" instead of "assistant"
        // Tool responses use "user" role since tool_calls are embedded in content, not in tool_calls field
        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "assistant" => "ai".to_string(),
                "tool" => "user".to_string(),
                other => other.to_string(),
            };

            // For Minimax, if there are tool calls, format them in the content
            // using the <minimax:tool_call> XML format
            let content = if let Some(tool_calls) = &msg.tool_calls {
                let mut content_parts = Vec::new();

                if !msg.content.is_empty() {
                    content_parts.push(msg.content.clone());
                }

                // Format tool calls in Minimax XML format
                for tc in tool_calls {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let mut params = String::new();
                    if let serde_json::Value::Object(map) = args {
                        for (key, value) in map {
                            let value_str = match value {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            };
                            params.push_str(&format!(
                                "<parameter name=\"{}\">{}</parameter>\n",
                                key, value_str
                            ));
                        }
                    }

                    content_parts.push(format!(
                        "<minimax:tool_call>\n<invoke name=\"{}\">\n{}</invoke>\n</minimax:tool_call>",
                        tc.name, params.trim_end()
                    ));
                }

                if content_parts.is_empty() {
                    None
                } else {
                    Some(content_parts.join("\n\n"))
                }
            } else if msg.content.is_empty() {
                None
            } else {
                // For tool results, wrap in <response> tags
                if msg.role == "tool" {
                    Some(format!("<response>{}</response>", msg.content))
                } else {
                    Some(msg.content.clone())
                }
            };

            messages.push(MlxLmMessage {
                role,
                content,
                // Don't send tool_calls in JSON format for Minimax - it's in the content
                tool_calls: None,
                tool_call_id: msg.tool_call_id.clone(),
                reasoning: msg.reasoning.clone(),
                ..Default::default()
            });
        }

        // Build tools in Minimax format (still JSON but will be rendered by template)
        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .tools
                    .iter()
                    .map(|t| MlxLmTool {
                        tool_type: "function".to_string(),
                        function: MlxLmFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools,
            tool_choice: None,
        }
    }

    /// Build request for ChatML template format (IQuest, Qwen, etc.)
    ///
    /// ChatML uses <|im_start|> and <|im_end|> delimiters with XML tool formatting.
    /// We pre-render the messages client-side to avoid server-side Jinja template issues
    /// with the "unhashable type 'list'" error that occurs with some chat templates.
    ///
    /// Tool format:
    /// - Tools are included in the system message as <tools>JSON</tools>
    /// - Tool calls use <tool_call>{"name": ..., "arguments": ...}</tool_call>
    /// - Tool results use <tool_response>content</tool_response>
    fn build_chatml_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut messages = Vec::new();

        // Build system message with tools embedded
        let mut system_content = String::new();
        if let Some(preamble) = &request.preamble {
            system_content.push_str(preamble);
        } else {
            system_content.push_str("You are a helpful assistant.");
        }

        // Add tools to system message in ChatML XML format
        if !request.tools.is_empty() {
            system_content.push_str("\n\n# Tools\n\n");
            system_content
                .push_str("You may call one or more functions to assist with the user query.\n\n");
            system_content.push_str(
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
            );

            for tool in &request.tools {
                let tool_json = serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                });
                system_content.push('\n');
                system_content.push_str(&serde_json::to_string(&tool_json).unwrap_or_default());
            }

            system_content.push_str("\n</tools>\n\n");
            system_content.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
            system_content.push_str("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        }

        messages.push(MlxLmMessage {
            role: "system".to_string(),
            content: Some(system_content),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });

        // Convert messages
        for msg in &request.messages {
            let content = if let Some(tool_calls) = &msg.tool_calls {
                // Format tool calls in ChatML XML format
                let mut content_parts = Vec::new();

                if !msg.content.is_empty() {
                    content_parts.push(msg.content.clone());
                }

                for tc in tool_calls {
                    let tool_call_json = serde_json::json!({
                        "name": tc.name,
                        "arguments": serde_json::from_str::<serde_json::Value>(&tc.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                    });
                    content_parts.push(format!(
                        "<tool_call>\n{}\n</tool_call>",
                        serde_json::to_string(&tool_call_json).unwrap_or_default()
                    ));
                }

                if content_parts.is_empty() {
                    None
                } else {
                    Some(content_parts.join("\n"))
                }
            } else if msg.role == "tool" {
                // Wrap tool results in <tool_response> tags
                Some(format!(
                    "<tool_response>\n{}\n</tool_response>",
                    msg.content
                ))
            } else if msg.content.is_empty() {
                None
            } else {
                Some(msg.content.clone())
            };

            // ChatML templates only support system/user/assistant roles
            // Remap "tool" to "user" (content is already wrapped in <tool_response> tags)
            let role = if msg.role == "tool" {
                "user".to_string()
            } else {
                msg.role.clone()
            };

            messages.push(MlxLmMessage {
                role,
                content,
                // Don't send tool_calls in JSON format - they're embedded in content
                tool_calls: None,
                tool_call_id: None,
                reasoning: msg.reasoning.clone(),
                ..Default::default()
            });
        }

        // Don't pass tools to server - they're embedded in system message
        // This avoids the Jinja template "unhashable type 'list'" error
        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
        }
    }

    /// Build request for Qwen 3.5+ template format
    ///
    /// Qwen 3.5 uses ChatML delimiters (<|im_start|>/<|im_end|>) with:
    /// - Tools in system message using Qwen3.5 format (JSON array in <tools> tags)
    /// - Tool call format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    /// - Tool results wrapped in `<tool_response>` tags within user messages
    /// - `<think>...</think>` reasoning blocks
    ///
    /// We pre-render messages client-side to match the Jinja template behavior.
    fn build_qwen35_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut messages = Vec::new();

        // Build system message with tools embedded (matching Qwen3.5 Jinja format)
        let mut system_content = String::new();
        if let Some(preamble) = &request.preamble {
            system_content.push_str(preamble);
        } else {
            system_content.push_str("You are a helpful assistant.");
        }

        // Add tools to system message in Qwen3.5 format
        if !request.tools.is_empty() {
            let mut tools_section = String::new();
            tools_section
                .push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");

            let tools_array: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                })
                .collect();
            tools_section.push('\n');
            tools_section.push_str(&serde_json::to_string_pretty(&tools_array).unwrap_or_default());
            tools_section.push_str("\n</tools>");
            tools_section.push_str(concat!(
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n",
                "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n",
                "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n",
                "</function>\n</tool_call>\n\n",
                "<IMPORTANT>\nReminder:\n",
                "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
                "- Required parameters MUST be specified\n",
                "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
                "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
                "</IMPORTANT>",
            ));

            // If there's already a system preamble, append tools section with separator
            if !system_content.is_empty() {
                system_content.push_str("\n\n");
            }
            system_content.push_str(&tools_section);
        }

        messages.push(MlxLmMessage {
            role: "system".to_string(),
            content: Some(system_content),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });

        // Convert messages
        for msg in &request.messages {
            let content = if let Some(tool_calls) = &msg.tool_calls {
                // Format tool calls in Qwen3.5 parameter-based format
                let mut content_parts = Vec::new();

                if !msg.content.is_empty() {
                    content_parts.push(msg.content.clone());
                }

                for tc in tool_calls {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let mut tool_call_str = format!("<tool_call>\n<function={}>\n", tc.name);
                    if let Some(obj) = args.as_object() {
                        for (key, value) in obj {
                            let value_str = if value.is_string() {
                                value.as_str().unwrap_or_default().to_string()
                            } else {
                                serde_json::to_string(value).unwrap_or_default()
                            };
                            tool_call_str.push_str(&format!(
                                "<parameter={}>\n{}\n</parameter>\n",
                                key, value_str
                            ));
                        }
                    }
                    tool_call_str.push_str("</function>\n</tool_call>");
                    content_parts.push(tool_call_str);
                }

                if content_parts.is_empty() {
                    None
                } else {
                    Some(content_parts.join("\n"))
                }
            } else if msg.role == "tool" {
                // Wrap tool results in <tool_response> tags (rendered as user messages)
                Some(format!(
                    "<tool_response>\n{}\n</tool_response>",
                    msg.content
                ))
            } else if msg.content.is_empty() {
                None
            } else {
                Some(msg.content.clone())
            };

            // Qwen3.5 renders tool responses as user messages
            let role = if msg.role == "tool" {
                "user".to_string()
            } else {
                msg.role.clone()
            };

            messages.push(MlxLmMessage {
                role,
                content,
                tool_calls: None,
                tool_call_id: None,
                // Qwen3.5 Jinja template expects `reasoning_content`, not `reasoning`
                reasoning_content: msg.reasoning.clone(),
                ..Default::default()
            });
        }

        // Don't pass tools to server - they're embedded in system message
        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
        }
    }

    /// Build request for GLM template format (GLM-4.7, etc.)
    ///
    /// GLM uses special delimiters and role markers:
    /// - Roles: <|user|>, <|assistant|>, <|system|>, <|tool|>
    /// - Tool calls use <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
    /// - Tool responses use <tool_response>content</tool_response> with "tool" role
    /// - Supports <think>...</think> reasoning blocks
    ///
    /// We pre-render messages client-side to avoid server-side Jinja template issues.
    fn build_glm_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut messages = Vec::new();

        // Build system message with tools embedded (GLM expects tools in system prompt)
        let mut system_content = String::new();
        if let Some(preamble) = &request.preamble {
            system_content.push_str(preamble);
        } else {
            system_content.push_str("You are a helpful assistant.");
        }

        // Add tools to system message in GLM format
        if !request.tools.is_empty() {
            system_content.push_str("\n\n# Tools\n\n");
            system_content
                .push_str("You may call one or more functions to assist with the user query.\n\n");
            system_content.push_str(
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
            );

            for tool in &request.tools {
                let tool_json = serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                });
                system_content.push('\n');
                system_content.push_str(&serde_json::to_string(&tool_json).unwrap_or_default());
            }

            system_content.push_str("\n</tools>\n\n");
            system_content.push_str("For each function call, output the function name and arguments within the following XML format:\n");
            system_content.push_str("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
        }

        messages.push(MlxLmMessage {
            role: "system".to_string(),
            content: Some(system_content),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });

        // Convert messages
        for msg in &request.messages {
            let content = if let Some(tool_calls) = &msg.tool_calls {
                // Format tool calls in GLM XML format
                let mut content_parts = Vec::new();

                if !msg.content.is_empty() {
                    content_parts.push(msg.content.clone());
                }

                for tc in tool_calls {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let mut formatted_call = format!("<tool_call>{}", tc.name);

                    if let serde_json::Value::Object(map) = args {
                        for (key, value) in map {
                            let value_str = match value {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            };
                            formatted_call.push_str(&format!(
                                "<arg_key>{}</arg_key><arg_value>{}</arg_value>",
                                key, value_str
                            ));
                        }
                    }

                    formatted_call.push_str("</tool_call>");
                    content_parts.push(formatted_call);
                }

                if content_parts.is_empty() {
                    None
                } else {
                    Some(content_parts.join("\n"))
                }
            } else if msg.role == "tool" {
                // Wrap tool results in <tool_response> tags
                Some(format!("<tool_response>{}</tool_response>", msg.content))
            } else if msg.content.is_empty() {
                None
            } else {
                Some(msg.content.clone())
            };

            let role = msg.role.clone();

            messages.push(MlxLmMessage {
                role,
                content,
                // Don't send tool_calls in JSON format - they're embedded in content
                tool_calls: None,
                tool_call_id: None,
                reasoning: msg.reasoning.clone(),
                ..Default::default()
            });
        }

        // Don't pass tools to server - they're embedded in system message
        // This avoids Jinja template errors
        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools: None,
            tool_choice: None,
        }
    }

    /// Build request for Gemma chat template format
    ///
    /// Gemma uses `<|turn>role\n...<turn|>` delimiters with:
    /// - `assistant` mapped to `model`
    /// - Tool definitions: `<|tool>declaration:name{...}<tool|>`
    /// - Tool calls: `<|tool_call>call:name{key:value,...}<tool_call|>`
    /// - Tool responses: `<|tool_response>response:name{...}<tool_response|>`
    /// - Strings quoted with `<|"|>` not `"`
    fn build_gemma_request(
        &self,
        request: &CompletionRequest,
        inner: &MlxLmClientInner,
    ) -> MlxLmRequest {
        let mut content_parts: Vec<String> = Vec::new();

        // Build system turn with tools
        let mut system_content = String::new();

        // Add preamble (system message)
        if let Some(preamble) = &request.preamble {
            system_content.push_str(preamble.trim());
        }

        // Add tool definitions in Gemma format
        if !request.tools.is_empty() {
            for tool in &request.tools {
                system_content.push_str(&format!(
                    "<|tool>declaration:{name}{{description:<|\"|>{desc}<|\"|>,parameters:{{properties:{{{props}}}}}}}<tool|>",
                    name = tool.name,
                    desc = tool.description,
                    props = Self::format_gemma_params(&tool.parameters),
                ));
            }
        }

        // Build system turn
        if !system_content.is_empty() {
            content_parts.push(format!("<|turn>system\n{}<turn|>\n", system_content));
        }

        // Convert messages
        for msg in &request.messages {
            let role = if msg.role == "assistant" {
                "model"
            } else {
                &msg.role
            };

            let mut turn_content = String::new();

            // Handle tool calls (assistant/model messages)
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    // Parse arguments JSON string into key-value pairs
                    let args_formatted = if let Ok(args_obj) =
                        serde_json::from_str::<serde_json::Value>(&tc.arguments)
                    {
                        Self::format_gemma_value(&args_obj, false)
                    } else {
                        tc.arguments.clone()
                    };
                    turn_content.push_str(&format!(
                        "<|tool_call>call:{}{}<tool_call|>",
                        tc.name, args_formatted
                    ));
                }
            }

            // Handle tool results (role=tool messages become tool_response)
            if msg.role == "tool" {
                let tool_name = msg.tool_call_id.as_deref().unwrap_or("unknown");
                let response_formatted = format!(
                    "<|tool_response>response:{}{{value:<|\"|>{}<|\"|>}}<tool_response|>",
                    tool_name,
                    msg.content.replace('"', "\\\"")
                );
                turn_content.push_str(&response_formatted);
                // Tool response turns use "user" role in Gemma
                content_parts.push(format!("<|turn>user\n{}<turn|>\n", turn_content));
                continue;
            }

            // Add text content
            if !msg.content.is_empty() {
                turn_content.push_str(msg.content.trim());
            }

            if !turn_content.is_empty() {
                content_parts.push(format!("<|turn>{}\n{}<turn|>\n", role, turn_content));
            }
        }

        // Add generation prompt
        content_parts.push("<|turn>model\n".to_string());

        // Combine into a single user message (pre-rendered)
        let rendered = content_parts.join("");

        let messages = vec![MlxLmMessage {
            role: "user".to_string(),
            content: Some(rendered),
            ..Default::default()
        }];

        MlxLmRequest {
            model: self.model_id.clone(),
            messages,
            temperature: request.temperature.or(inner.temperature),
            max_tokens: request
                .max_tokens
                .or(inner.max_tokens)
                .or(Some(DEFAULT_MAX_TOKENS))
                .map(|t| t.min(MAX_TOKENS_CAP)),
            top_p: inner.top_p,
            top_k: inner.top_k,
            repetition_penalty: inner.repetition_penalty,
            repetition_context_size: inner.repetition_context_size,
            frequency_penalty: inner.frequency_penalty,
            presence_penalty: inner.presence_penalty,
            min_p: inner.min_p,
            stop: inner.stop_sequences.clone(),
            tools: None, // Pre-rendered in content
            tool_choice: None,
        }
    }

    /// Format a JSON value in Gemma's key-value format (no JSON quotes, uses <|"|> for strings)
    fn format_gemma_value(value: &serde_json::Value, escape_keys: bool) -> String {
        match value {
            serde_json::Value::Object(map) => {
                let mut parts = Vec::new();
                for (k, v) in map {
                    let key = if escape_keys {
                        format!("<|\"|>{}<|\"|>", k)
                    } else {
                        k.clone()
                    };
                    parts.push(format!(
                        "{}:{}",
                        key,
                        Self::format_gemma_value(v, escape_keys)
                    ));
                }
                format!("{{{}}}", parts.join(","))
            }
            serde_json::Value::Array(arr) => {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| Self::format_gemma_value(v, escape_keys))
                    .collect();
                format!("[{}]", items.join(","))
            }
            serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
            serde_json::Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Null => "null".to_string(),
        }
    }

    /// Format tool parameters in Gemma's compact format
    fn format_gemma_params(params: &serde_json::Value) -> String {
        if let Some(props) = params.get("properties").and_then(|v| v.as_object()) {
            let mut parts = Vec::new();
            for (name, schema) in props {
                let type_str = schema
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("STRING")
                    .to_uppercase();
                let desc = schema
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let mut field = format!(
                    "{}:{{description:<|\"|>{}<|\"|>,type:<|\"|>{}<|\"|>}}",
                    name, desc, type_str
                );
                if type_str == "OBJECT" {
                    if let Some(inner_props) = schema.get("properties") {
                        field = format!("{}:{{description:<|\"|>{}<|\"|>,properties:{{{}}},type:<|\"|>{}<|\"|>}}",
                            name, desc, Self::format_gemma_params(inner_props), type_str);
                    }
                }
                parts.push(field);
            }
            parts.join(",")
        } else {
            String::new()
        }
    }

    /// Parse OpenAI-compatible response
    fn parse_openai_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
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

        let finish_reason = choice.finish_reason.clone();

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
            reasoning_content: None,
            finish_reason,
        })
    }

    /// Parse Minimax-formatted response
    ///
    /// Minimax responses may contain:
    /// - Thinking content in `<think>...</think>` tags (MiniMax M2.1 interleaved thinking)
    /// - Tool calls in XML format within the content:
    ///   <minimax:tool_call>
    ///   <invoke name="tool-name">
    ///   <parameter name="param-key">param-value</parameter>
    ///   </invoke>
    ///   </minimax:tool_call>
    fn parse_minimax_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        let finish_reason = choice.finish_reason.clone();
        let content = choice.message.content.clone().unwrap_or_default();

        // Check for reasoning content in dedicated JSON fields (MiniMax M2.1 may return this)
        let json_reasoning = choice
            .message
            .reasoning
            .clone()
            .or_else(|| choice.message.reasoning_content.clone());

        // Extract thinking blocks from content (MiniMax M2.1 interleaved thinking in <think> tags)
        let (tag_reasoning, content_without_thinking) = self.extract_thinking_blocks(&content);

        // Combine reasoning from JSON field and <think> tags
        let reasoning_content = match (&json_reasoning, &tag_reasoning) {
            (Some(json), Some(tag)) => Some(format!("{}\n\n{}", json, tag)),
            (Some(json), None) => Some(json.clone()),
            (None, Some(tag)) => Some(tag.clone()),
            (None, None) => None,
        };

        // Check for native tool calls first (standard OpenAI format via choice.message.tool_calls)
        // MiniMax may return tool calls in this format instead of XML
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                tracing::debug!(
                    native_tool_calls_count = native_tool_calls.len(),
                    "Using native tool calls from response (Minimax)"
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
                    finish_reason: finish_reason.clone(),
                });
            }
        }

        // Parse tool calls from Minimax XML format in content outside thinking blocks
        let (text_content, mut tool_calls) = self.parse_minimax_content(&content_without_thinking);

        // Also try JSON format in content (some models return JSON tool calls in text)
        if tool_calls.is_empty() {
            let json_tool_calls = self.parse_json_tool_calls(&content_without_thinking);
            if !json_tool_calls.is_empty() {
                tracing::debug!(
                    count = json_tool_calls.len(),
                    "Found JSON tool calls in content"
                );
                tool_calls.extend(json_tool_calls);
            }
        }

        // ALSO parse tool calls from inside <think> tags (models may emit tool calls there)
        if let Some(ref reasoning) = tag_reasoning {
            // Try XML format first
            let (_, thinking_tool_calls) = self.parse_minimax_content(reasoning);
            if !thinking_tool_calls.is_empty() {
                tracing::debug!(
                    count = thinking_tool_calls.len(),
                    "Found XML tool calls inside <think> blocks"
                );
                tool_calls.extend(thinking_tool_calls);
            } else {
                // Try JSON format
                let json_thinking_tool_calls = self.parse_json_tool_calls(reasoning);
                if !json_thinking_tool_calls.is_empty() {
                    tracing::debug!(
                        count = json_thinking_tool_calls.len(),
                        "Found JSON tool calls inside <think> blocks"
                    );
                    tool_calls.extend(json_thinking_tool_calls);
                }
            }
        }

        // ALSO parse tool calls from JSON reasoning field (models may emit tool calls there)
        if let Some(ref json_reason) = json_reasoning {
            // Try XML format first
            let (_, json_reasoning_tool_calls) = self.parse_minimax_content(json_reason);
            if !json_reasoning_tool_calls.is_empty() {
                tracing::debug!(
                    count = json_reasoning_tool_calls.len(),
                    "Found XML tool calls inside JSON reasoning field"
                );
                tool_calls.extend(json_reasoning_tool_calls);
            } else {
                // Try JSON format
                let json_format_tool_calls = self.parse_json_tool_calls(json_reason);
                if !json_format_tool_calls.is_empty() {
                    tracing::debug!(
                        count = json_format_tool_calls.len(),
                        "Found JSON tool calls inside JSON reasoning field"
                    );
                    tool_calls.extend(json_format_tool_calls);
                }
            }
        }

        // Map role back from "ai" to "assistant"
        let role = match choice.message.role.as_str() {
            "ai" => "assistant".to_string(),
            other => other.to_string(),
        };

        let message = Message {
            role,
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

    /// Parse ChatML-formatted response
    ///
    /// ChatML responses may contain tool calls in XML format within the content:
    /// <tool_call>
    /// {"name": "tool-name", "arguments": {...}}
    /// </tool_call>
    fn parse_chatml_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        let finish_reason = choice.finish_reason.clone();
        let content = choice.message.content.clone().unwrap_or_default();

        // If the model returned native tool calls (OpenAI format), use those
        // Some models configured for ChatML template still return tool calls in native format
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                tracing::debug!(
                    native_tool_calls_count = native_tool_calls.len(),
                    "ChatML response has native tool calls, using OpenAI parsing path"
                );
                return self.parse_openai_response(response);
            }
        }

        // Parse tool calls from ChatML XML format
        let (text_content, tool_calls) = self.parse_chatml_content(&content);

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
            reasoning_content: None,
            finish_reason,
        })
    }

    /// Parse ChatML content to extract text and tool calls
    fn parse_chatml_content(&self, content: &str) -> (String, Vec<ToolCall>) {
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = content;
        let mut tool_call_counter = 0;

        while let Some(start) = remaining.find("<tool_call>") {
            // Add text before the tool call
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            // Find the end of the tool call
            if let Some(end) = remaining.find("</tool_call>") {
                let tool_call_content = &remaining[start + "<tool_call>".len()..end];

                // Parse the JSON tool call
                if let Some(tc) =
                    self.parse_chatml_tool_call(tool_call_content.trim(), tool_call_counter)
                {
                    tool_calls.push(tc);
                    tool_call_counter += 1;
                }

                remaining = &remaining[end + "</tool_call>".len()..];
            } else {
                break;
            }
        }

        // Add any remaining text
        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), tool_calls)
    }

    /// Parse a single ChatML tool call JSON
    fn parse_chatml_tool_call(&self, json_str: &str, index: usize) -> Option<ToolCall> {
        // Try to parse as JSON
        let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

        let name = parsed.get("name")?.as_str()?.to_string();
        let arguments = match parsed.get("arguments") {
            Some(args) => {
                if args.is_string() {
                    args.as_str()?.to_string()
                } else {
                    serde_json::to_string(args).ok()?
                }
            }
            None => "{}".to_string(),
        };

        Some(ToolCall {
            id: format!("call_{}", index),
            name,
            arguments,
        })
    }

    /// Parse Qwen3.5-formatted response
    ///
    /// Qwen3.5 responses may contain:
    /// - Thinking content in `<think>...</think>` tags
    /// - Tool calls in parameter-based format:
    ///   `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    fn parse_qwen35_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
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
        let (tag_reasoning, content_without_thinking) = self.extract_thinking_blocks(&content);

        // Combine reasoning from JSON field and <think> tags
        let reasoning_content = match (&json_reasoning, &tag_reasoning) {
            (Some(json), Some(tag)) => Some(format!("{}\n\n{}", json, tag)),
            (Some(json), None) => Some(json.clone()),
            (None, Some(tag)) => Some(tag.clone()),
            (None, None) => None,
        };

        // If the model returned native tool calls (OpenAI format), use those
        // but still preserve reasoning extracted above
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                tracing::debug!(
                    native_tool_calls_count = native_tool_calls.len(),
                    "Qwen3.5 response has native tool calls, preserving reasoning"
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

        // Parse tool calls from Qwen3.5 format in content outside thinking blocks
        let (text_content, mut tool_calls) = self.parse_qwen35_content(&content_without_thinking);

        // Also parse tool calls from reasoning sources (models may emit tool calls there)
        // Check both <think> tag content and JSON reasoning fields, skipping duplicates
        // (some backends mirror the same reasoning into both fields)
        let mut parsed_reasoning: Option<&str> = None;
        for reasoning in [&tag_reasoning, &json_reasoning].into_iter().flatten() {
            // Skip if this is the same content we already parsed
            if parsed_reasoning == Some(reasoning.as_str()) {
                continue;
            }
            parsed_reasoning = Some(reasoning.as_str());

            let (_, reasoning_tool_calls) = self.parse_qwen35_content(reasoning);
            if !reasoning_tool_calls.is_empty() {
                tracing::debug!(
                    count = reasoning_tool_calls.len(),
                    "Found tool calls inside reasoning blocks (Qwen3.5)"
                );
                // Renumber to avoid duplicate IDs with tool calls from main content
                let base = tool_calls.len();
                for (i, mut tc) in reasoning_tool_calls.into_iter().enumerate() {
                    tc.id = format!("call_{}", base + i);
                    tool_calls.push(tc);
                }
            }
        }

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

    /// Parse Qwen3.5 content to extract text and tool calls
    ///
    /// Qwen3.5 tool call format:
    /// `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    fn parse_qwen35_content(&self, content: &str) -> (String, Vec<ToolCall>) {
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = content;
        let mut tool_call_counter = 0;

        while let Some(start) = remaining.find("<tool_call>") {
            // Add text before the tool call
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            // Find the end of the tool call
            if let Some(end) = remaining[start..].find("</tool_call>") {
                let tool_call_content = &remaining[start + "<tool_call>".len()..start + end];

                // Try Qwen3.5 parameter-based format first
                if let Some(tc) =
                    self.parse_qwen35_tool_call(tool_call_content.trim(), tool_call_counter)
                {
                    tool_calls.push(tc);
                    tool_call_counter += 1;
                } else if let Some(tc) =
                    self.parse_chatml_tool_call(tool_call_content.trim(), tool_call_counter)
                {
                    // Fall back to JSON format (ChatML style) for compatibility
                    tool_calls.push(tc);
                    tool_call_counter += 1;
                }

                remaining = &remaining[start + end + "</tool_call>".len()..];
            } else {
                break;
            }
        }

        // Add any remaining text
        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), tool_calls)
    }

    /// Parse a single Qwen3.5 parameter-based tool call
    ///
    /// Format: `<function=name><parameter=key>value</parameter>...</function>`
    fn parse_qwen35_tool_call(&self, content: &str, index: usize) -> Option<ToolCall> {
        // Extract function name from <function=name>
        let func_start = content.find("<function=")?;
        let name_start = func_start + "<function=".len();
        let name_end = content[name_start..].find('>')? + name_start;
        let name = content[name_start..name_end].to_string();

        // Extract parameters
        let mut params = serde_json::Map::new();
        let mut search_start = name_end;

        while let Some(param_offset) = content[search_start..].find("<parameter=") {
            let abs_param_start = search_start + param_offset;
            let key_start = abs_param_start + "<parameter=".len();

            if let Some(key_end_offset) = content[key_start..].find('>') {
                let key_end = key_start + key_end_offset;
                let key = content[key_start..key_end].to_string();

                // Find the value between > and </parameter>
                let value_start = key_end + 1;
                if let Some(value_end_offset) = content[value_start..].find("</parameter>") {
                    let value_end = value_start + value_end_offset;
                    let raw_value = &content[value_start..value_end];
                    // Strip at most one leading/trailing newline (template artifact)
                    // but preserve internal whitespace that may be significant
                    let value = raw_value.strip_prefix('\n').unwrap_or(raw_value);
                    let value = value.strip_suffix('\n').unwrap_or(value);

                    // Try to parse as JSON value (number, bool, object, array)
                    // Use trimmed version for type detection so " 42 " still parses as number
                    let json_value = if let Ok(parsed) =
                        serde_json::from_str::<serde_json::Value>(value.trim())
                    {
                        parsed
                    } else {
                        serde_json::Value::String(value.to_string())
                    };
                    params.insert(key, json_value);
                    search_start = value_end + "</parameter>".len();
                    continue;
                }
            }
            break;
        }

        if name.is_empty() {
            return None;
        }

        let arguments = serde_json::to_string(&serde_json::Value::Object(params)).ok()?;

        Some(ToolCall {
            id: format!("call_{}", index),
            name,
            arguments,
        })
    }

    /// Parse Minimax content to extract text and tool calls
    fn parse_minimax_content(&self, content: &str) -> (String, Vec<ToolCall>) {
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = content;
        let mut tool_call_counter = 0;

        while let Some(start) = remaining.find("<minimax:tool_call>") {
            // Add text before the tool call
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            // Find the end of the tool call
            if let Some(end) = remaining.find("</minimax:tool_call>") {
                let tool_call_xml = &remaining[start..end + "</minimax:tool_call>".len()];

                // Parse the tool call
                if let Some(tc) = self.parse_minimax_tool_call(tool_call_xml, tool_call_counter) {
                    tool_calls.push(tc);
                    tool_call_counter += 1;
                }

                remaining = &remaining[end + "</minimax:tool_call>".len()..];
            } else {
                break;
            }
        }

        // Add any remaining text
        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), tool_calls)
    }

    /// Parse a single Minimax tool call XML block
    fn parse_minimax_tool_call(&self, xml: &str, index: usize) -> Option<ToolCall> {
        // Extract function name from <invoke name="...">
        let name_start = xml.find("<invoke name=\"")? + "<invoke name=\"".len();
        let name_end = xml[name_start..].find('"')? + name_start;
        let name = xml[name_start..name_end].to_string();

        // Extract parameters
        let mut params = serde_json::Map::new();

        let mut search_start = 0;
        while let Some(param_start) = xml[search_start..].find("<parameter name=\"") {
            let abs_param_start = search_start + param_start;
            let key_start = abs_param_start + "<parameter name=\"".len();

            if let Some(key_end_offset) = xml[key_start..].find('"') {
                let key_end = key_start + key_end_offset;
                let key = xml[key_start..key_end].to_string();

                // Find the value between > and </parameter>
                if let Some(value_start_offset) = xml[key_end..].find('>') {
                    let value_start = key_end + value_start_offset + 1;
                    if let Some(value_end_offset) = xml[value_start..].find("</parameter>") {
                        let value_end = value_start + value_end_offset;
                        let value = xml[value_start..value_end].to_string();
                        params.insert(key, serde_json::Value::String(value));
                        search_start = value_end;
                        continue;
                    }
                }
            }
            break;
        }

        let arguments = serde_json::to_string(&serde_json::Value::Object(params)).ok()?;

        Some(ToolCall {
            id: format!("call_{}", index),
            name,
            arguments,
        })
    }

    /// Parse JSON-formatted tool calls from text content
    ///
    /// Looks for tool calls in various JSON formats:
    /// - OpenAI-style: `{"type": "function", "function": {"name": "...", "arguments": "..."}}`
    /// - Simple: `{"name": "...", "arguments": {...}}`
    /// - Wrapped: `<tool_call>{"name": "...", ...}</tool_call>`
    fn parse_json_tool_calls(&self, content: &str) -> Vec<ToolCall> {
        let mut tool_calls = Vec::new();
        let mut call_index = 0;

        // Try to find <tool_call>...</tool_call> wrapped JSON
        let mut remaining = content;
        while let Some(start) = remaining.find("<tool_call>") {
            if let Some(end) = remaining.find("</tool_call>") {
                let json_str = &remaining[start + "<tool_call>".len()..end];
                if let Some(tc) = self.parse_json_tool_call(json_str.trim(), call_index) {
                    tool_calls.push(tc);
                    call_index += 1;
                }
                remaining = &remaining[end + "</tool_call>".len()..];
            } else {
                break;
            }
        }

        // Also try to find standalone JSON objects that look like tool calls
        // Look for {"name": "...", "arguments": ...} patterns
        let mut search_pos = 0;
        while search_pos < content.len() {
            // Find potential JSON object start
            if let Some(obj_start) = content[search_pos..].find("{\"name\"") {
                let abs_start = search_pos + obj_start;
                // Try to find matching closing brace
                if let Some(tc) =
                    self.try_parse_json_object_as_tool_call(&content[abs_start..], call_index)
                {
                    tool_calls.push(tc);
                    call_index += 1;
                }
                search_pos = abs_start + 1;
            } else {
                break;
            }
        }

        tool_calls
    }

    /// Try to parse a JSON object as a tool call
    fn parse_json_tool_call(&self, json_str: &str, index: usize) -> Option<ToolCall> {
        // Try OpenAI-style format: {"type": "function", "function": {"name": "...", "arguments": "..."}}
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(json_str) {
            // Check for OpenAI format
            if let Some(func) = obj.get("function") {
                let name = func.get("name")?.as_str()?.to_string();
                let arguments = if let Some(args) = func.get("arguments") {
                    if args.is_string() {
                        args.as_str()?.to_string()
                    } else {
                        serde_json::to_string(args).ok()?
                    }
                } else {
                    "{}".to_string()
                };
                let id = obj
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
            if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
                let arguments = if let Some(args) = obj.get("arguments") {
                    if args.is_string() {
                        args.as_str()?.to_string()
                    } else {
                        serde_json::to_string(args).ok()?
                    }
                } else {
                    "{}".to_string()
                };
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("call_{}", index));
                return Some(ToolCall {
                    id,
                    name: name.to_string(),
                    arguments,
                });
            }
        }
        None
    }

    /// Try to parse a JSON object starting at the given position
    fn try_parse_json_object_as_tool_call(&self, content: &str, index: usize) -> Option<ToolCall> {
        // Find the matching closing brace
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
        self.parse_json_tool_call(json_str, index)
    }

    /// Extract thinking blocks from content (MiniMax M2.1 interleaved thinking)
    ///
    /// MiniMax M2.1 outputs reasoning content in `<think>...</think>` tags.
    /// This method extracts all thinking blocks and returns them separately
    /// from the main content.
    ///
    /// Returns (reasoning_content, content_without_thinking)
    fn extract_thinking_blocks(&self, content: &str) -> (Option<String>, String) {
        let mut thinking_parts = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = content;

        while let Some(start) = remaining.find("<think>") {
            // Add text before the thinking block
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            // Find the end of the thinking block
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

        // Add any remaining text after all thinking blocks
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

    /// Parse GLM-formatted response
    ///
    /// GLM responses may contain:
    /// - Thinking content in `<think>...</think>` tags
    /// - Tool calls in XML format within the content:
    ///   <tool_call>{function-name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value>...</tool_call>
    fn parse_glm_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        let finish_reason = choice.finish_reason.clone();
        let content = choice.message.content.clone().unwrap_or_default();

        // If the model returned native tool calls (OpenAI format), use those
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                tracing::debug!(
                    native_tool_calls_count = native_tool_calls.len(),
                    "GLM response has native tool calls, using OpenAI parsing path"
                );
                return self.parse_openai_response(response);
            }
        }

        // Check for reasoning content in dedicated JSON fields
        let json_reasoning = choice
            .message
            .reasoning
            .clone()
            .or_else(|| choice.message.reasoning_content.clone());

        // Extract thinking blocks from content (<think> tags)
        let (tag_reasoning, content_without_thinking) = self.extract_thinking_blocks(&content);

        // Combine reasoning from JSON field and <think> tags
        let reasoning_content = match (&json_reasoning, &tag_reasoning) {
            (Some(json), Some(tag)) => Some(format!("{}\n\n{}", json, tag)),
            (Some(json), None) => Some(json.clone()),
            (None, Some(tag)) => Some(tag.clone()),
            (None, None) => None,
        };

        // Parse tool calls from GLM XML format in content outside thinking blocks
        let (text_content, mut tool_calls) = self.parse_glm_content(&content_without_thinking);

        // ALSO parse tool calls from inside <think> tags (models may emit tool calls there)
        if let Some(ref reasoning) = tag_reasoning {
            let (_, thinking_tool_calls) = self.parse_glm_content(reasoning);
            if !thinking_tool_calls.is_empty() {
                tracing::debug!(
                    count = thinking_tool_calls.len(),
                    "Found tool calls inside <think> blocks (GLM)"
                );
                tool_calls.extend(thinking_tool_calls);
            }
        }

        // ALSO parse tool calls from JSON reasoning field (models may emit tool calls there)
        if let Some(ref json_reason) = json_reasoning {
            let (_, json_reasoning_tool_calls) = self.parse_glm_content(json_reason);
            if !json_reasoning_tool_calls.is_empty() {
                tracing::debug!(
                    count = json_reasoning_tool_calls.len(),
                    "Found tool calls inside JSON reasoning field (GLM)"
                );
                tool_calls.extend(json_reasoning_tool_calls);
            }
        }

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

    /// Parse GLM content to extract text and tool calls
    ///
    /// GLM tool call format:
    /// <tool_call>{function-name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value>...</tool_call>
    fn parse_glm_content(&self, content: &str) -> (String, Vec<ToolCall>) {
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = content;
        let mut tool_call_counter = 0;

        while let Some(start) = remaining.find("<tool_call>") {
            // Add text before the tool call
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            // Find the end of the tool call
            if let Some(end) = remaining.find("</tool_call>") {
                let tool_call_content = &remaining[start + "<tool_call>".len()..end];

                // Parse the GLM tool call
                if let Some(tc) = self.parse_glm_tool_call(tool_call_content, tool_call_counter) {
                    tool_calls.push(tc);
                    tool_call_counter += 1;
                }

                remaining = &remaining[end + "</tool_call>".len()..];
            } else {
                break;
            }
        }

        // Add any remaining text
        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), tool_calls)
    }

    /// Parse a single GLM tool call
    ///
    /// GLM format: {function-name}<arg_key>{key1}</arg_key><arg_value>{value1}</arg_value>...
    fn parse_glm_tool_call(&self, content: &str, index: usize) -> Option<ToolCall> {
        // The function name is at the beginning, before any <arg_key> tags
        let name_end = content.find("<arg_key>").unwrap_or(content.len());
        let name = content[..name_end].trim().to_string();

        if name.is_empty() {
            return None;
        }

        // Extract key-value pairs
        let mut params = serde_json::Map::new();
        let mut search_start = 0;

        while let Some(key_start) = content[search_start..].find("<arg_key>") {
            let abs_key_start = search_start + key_start + "<arg_key>".len();

            if let Some(key_end_offset) = content[abs_key_start..].find("</arg_key>") {
                let key_end = abs_key_start + key_end_offset;
                let key = content[abs_key_start..key_end].trim().to_string();

                // Find the corresponding value
                if let Some(value_start_offset) = content[key_end..].find("<arg_value>") {
                    let value_start = key_end + value_start_offset + "<arg_value>".len();

                    if let Some(value_end_offset) = content[value_start..].find("</arg_value>") {
                        let value_end = value_start + value_end_offset;
                        let value = content[value_start..value_end].to_string();

                        // Try to parse the value as JSON, otherwise keep as string
                        let json_value = serde_json::from_str::<serde_json::Value>(&value)
                            .unwrap_or(serde_json::Value::String(value));
                        params.insert(key, json_value);

                        search_start = value_end;
                        continue;
                    }
                }
            }
            break;
        }

        let arguments = serde_json::to_string(&serde_json::Value::Object(params)).ok()?;

        Some(ToolCall {
            id: format!("call_{}", index),
            name,
            arguments,
        })
    }

    /// Parse Gemma response format
    ///
    /// Gemma tool calls: `<|tool_call>call:name{key:value,...}<tool_call|>`
    /// Gemma thinking: `<|channel>thought\n...<channel|>`
    fn parse_gemma_response(
        &self,
        response: MlxLmResponse,
    ) -> Result<CompletionResponse<MlxLmResponse>, CompletionError> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::Provider(ProviderError::InvalidResponse(
                "No choices in response".to_string(),
            ))
        })?;

        // If native tool calls present, use OpenAI parsing path
        if let Some(ref native_tool_calls) = choice.message.tool_calls {
            if !native_tool_calls.is_empty() {
                return self.parse_openai_response(response);
            }
        }

        let finish_reason = choice.finish_reason.clone();
        let content = choice.message.content.clone().unwrap_or_default();

        // Extract thinking blocks (<|channel>thought\n...<channel|>)
        let (thinking, main_content) = self.extract_gemma_thinking(&content);

        // Parse tool calls from content
        let (text, tool_calls) = self.parse_gemma_content(&main_content);

        let message = Message {
            role: "assistant".to_string(),
            content: text.trim().to_string(),
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
            reasoning_content: thinking,
            finish_reason,
        })
    }

    /// Extract Gemma thinking blocks: `<|channel>thought\n...<channel|>`
    fn extract_gemma_thinking(&self, content: &str) -> (Option<String>, String) {
        let mut thinking_parts = Vec::new();
        let mut main_content = String::new();
        let mut remaining = content;

        while let Some(start) = remaining.find("<|channel>") {
            main_content.push_str(&remaining[..start]);
            let after_tag = &remaining[start + "<|channel>".len()..];
            if let Some(end) = after_tag.find("<channel|>") {
                let block = &after_tag[..end];
                // Strip "thought\n" prefix if present
                let thinking = block.strip_prefix("thought\n").unwrap_or(block);
                if !thinking.trim().is_empty() {
                    thinking_parts.push(thinking.trim().to_string());
                }
                remaining = &after_tag[end + "<channel|>".len()..];
            } else {
                // Unclosed tag — treat rest as thinking
                let thinking = after_tag.strip_prefix("thought\n").unwrap_or(after_tag);
                if !thinking.trim().is_empty() {
                    thinking_parts.push(thinking.trim().to_string());
                }
                remaining = "";
                break;
            }
        }
        main_content.push_str(remaining);

        // Also try standard <think> blocks
        let (std_thinking, main_after_std) = self.extract_thinking_blocks(&main_content);
        if let Some(std) = std_thinking {
            thinking_parts.push(std);
        }

        let thinking = if thinking_parts.is_empty() {
            None
        } else {
            Some(thinking_parts.join("\n\n"))
        };
        (thinking, main_after_std)
    }

    /// Parse Gemma content to extract text and tool calls
    ///
    /// Tool calls: `<|tool_call>call:name{args}<tool_call|>`
    fn parse_gemma_content(&self, content: &str) -> (String, Vec<ToolCall>) {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut remaining = content;
        let mut call_index = 0;

        while let Some(start) = remaining.find("<|tool_call>") {
            // Collect text before the tool call
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            let after_tag = &remaining[start + "<|tool_call>".len()..];
            if let Some(end) = after_tag.find("<tool_call|>") {
                let call_content = &after_tag[..end];
                if let Some(tc) = self.parse_gemma_tool_call(call_content, call_index) {
                    tool_calls.push(tc);
                    call_index += 1;
                }
                remaining = &after_tag[end + "<tool_call|>".len()..];
            } else {
                remaining = after_tag;
                break;
            }
        }

        // Collect remaining text
        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        // Strip turn markers from text
        let text = text_parts
            .join("\n")
            .replace("<turn|>", "")
            .replace("<|turn>model\n", "")
            .replace("<|turn>user\n", "");

        (text.trim().to_string(), tool_calls)
    }

    /// Parse a single Gemma tool call: `call:name{key:value,...}`
    fn parse_gemma_tool_call(&self, content: &str, index: usize) -> Option<ToolCall> {
        let content = content.trim();
        let content = content.strip_prefix("call:").unwrap_or(content);

        // Find the function name (everything before the first `{`)
        let brace_pos = content.find('{')?;
        let name = content[..brace_pos].trim().to_string();
        let args_str = &content[brace_pos..];

        // Convert Gemma format to JSON: replace <|"|> with "
        let json_args = args_str.replace("<|\"|>", "\"");

        // Try parsing as JSON directly
        let arguments = if serde_json::from_str::<serde_json::Value>(&json_args).is_ok() {
            json_args
        } else {
            // Fallback: wrap in JSON object
            format!(
                "{{{}}}",
                json_args.trim_start_matches('{').trim_end_matches('}')
            )
        };

        Some(ToolCall {
            id: format!("call_{}", index),
            name,
            arguments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let client = MlxLmClientBuilder::new().build();
        assert_eq!(client.inner.base_url, "http://localhost:8080/v1");
        assert_eq!(client.inner.chat_template, ChatTemplate::OpenAI);
    }

    #[test]
    fn test_builder_custom_url() {
        let client = MlxLmClientBuilder::new()
            .base_url("http://custom:9000/v1")
            .build();
        assert_eq!(client.inner.base_url, "http://custom:9000/v1");
    }

    #[test]
    fn test_builder_minimax_template() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Minimax);
    }

    #[test]
    fn test_parse_minimax_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let xml = r#"<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"#;

        let tool_call = model.parse_minimax_tool_call(xml, 0).unwrap();
        assert_eq!(tool_call.name, "get_weather");
        assert_eq!(tool_call.id, "call_0");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_parse_minimax_content_with_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"I'll check the weather for you.

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Paris</parameter>
</invoke>
</minimax:tool_call>"#;

        let (text, tool_calls) = model.parse_minimax_content(content);
        assert_eq!(text, "I'll check the weather for you.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_parse_minimax_content_no_tool_calls() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = "This is just a regular response without any tool calls.";
        let (text, tool_calls) = model.parse_minimax_content(content);

        assert_eq!(text, content);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_temperature_clamping() {
        let client = MlxLmClientBuilder::new().temperature(3.0).build();
        assert_eq!(client.inner.temperature, Some(2.0));

        let client = MlxLmClientBuilder::new().temperature(-1.0).build();
        assert_eq!(client.inner.temperature, Some(0.0));
    }

    #[test]
    fn test_top_p_clamping() {
        let client = MlxLmClientBuilder::new().top_p(1.5).build();
        assert_eq!(client.inner.top_p, Some(1.0));

        let client = MlxLmClientBuilder::new().top_p(-0.5).build();
        assert_eq!(client.inner.top_p, Some(0.0));
    }

    #[test]
    fn test_stop_sequences() {
        let client = MlxLmClientBuilder::new()
            .stop_sequences(vec!["STOP".to_string(), "END".to_string()])
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        assert_eq!(stops.len(), 2);
        assert!(stops.contains(&"STOP".to_string()));
        assert!(stops.contains(&"END".to_string()));
    }

    #[test]
    fn test_add_stop_sequences() {
        let client = MlxLmClientBuilder::new()
            .stop_sequences(vec!["STOP".to_string()])
            .add_stop_sequences(vec!["END".to_string(), "QUIT".to_string()])
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        assert_eq!(stops.len(), 3);
        assert!(stops.contains(&"STOP".to_string()));
        assert!(stops.contains(&"END".to_string()));
        assert!(stops.contains(&"QUIT".to_string()));
    }

    #[test]
    fn test_chatml_stop_sequences() {
        let client = MlxLmClientBuilder::new()
            .with_chatml_stop_sequences()
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        assert!(stops.contains(&"<|im_end|>".to_string()));
        assert!(stops.contains(&"<|im_start|>".to_string()));
        assert!(stops.contains(&"\n\n\n\n".to_string()));
        assert!(stops.contains(&"####".to_string()));
    }

    #[test]
    fn test_anti_repetition_stops() {
        let client = MlxLmClientBuilder::new()
            .with_anti_repetition_stops()
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();

        // Repetitive characters
        assert!(stops.contains(&"EEEE".to_string()));
        assert!(stops.contains(&"====".to_string()));
        assert!(stops.contains(&"----".to_string()));
        assert!(stops.contains(&"####".to_string()));
        assert!(stops.contains(&"****".to_string()));
        assert!(stops.contains(&"....".to_string()));
        assert!(stops.contains(&",,,,".to_string()));

        // Excessive whitespace
        assert!(stops.contains(&"\n\n\n\n".to_string()));
        assert!(stops.contains(&"        ".to_string())); // 8 spaces
        assert!(stops.contains(&"            ".to_string())); // 12 spaces
        assert!(stops.contains(&"\t\t\t\t".to_string()));
    }

    #[test]
    fn test_combined_stop_sequences() {
        let client = MlxLmClientBuilder::new()
            .with_chatml_stop_sequences()
            .with_anti_repetition_stops()
            .add_stop_sequences(vec!["CUSTOM".to_string()])
            .build();

        let stops = client.inner.stop_sequences.as_ref().unwrap();
        // Should have ChatML stops
        assert!(stops.contains(&"<|im_end|>".to_string()));
        // Should have anti-repetition stops
        assert!(stops.contains(&"EEEE".to_string()));
        // Should have custom stop
        assert!(stops.contains(&"CUSTOM".to_string()));
    }

    #[test]
    fn test_repetition_penalty_clamping() {
        // Test lower bound
        let client = MlxLmClientBuilder::new().repetition_penalty(0.5).build();
        assert_eq!(client.inner.repetition_penalty, Some(1.0));

        // Test upper bound
        let client = MlxLmClientBuilder::new().repetition_penalty(3.0).build();
        assert_eq!(client.inner.repetition_penalty, Some(2.0));

        // Test valid value
        let client = MlxLmClientBuilder::new().repetition_penalty(1.15).build();
        assert_eq!(client.inner.repetition_penalty, Some(1.15));
    }

    #[test]
    fn test_anti_loop_config() {
        let client = MlxLmClientBuilder::new().with_anti_loop_config().build();

        assert_eq!(client.inner.temperature, Some(0.7));
        assert_eq!(client.inner.top_p, Some(0.95));
        assert_eq!(client.inner.top_k, Some(40));
        assert_eq!(client.inner.repetition_penalty, Some(1.15));
        assert_eq!(client.inner.repetition_context_size, Some(256));
    }

    #[test]
    fn test_anti_loop_config_can_override() {
        // Test that manual settings can override the preset
        let client = MlxLmClientBuilder::new()
            .with_anti_loop_config()
            .temperature(0.8) // Override after preset
            .build();

        assert_eq!(client.inner.temperature, Some(0.8));
        assert_eq!(client.inner.top_p, Some(0.95)); // Still from preset
    }

    #[test]
    fn test_extract_thinking_blocks_single() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<think>
Let me analyze this code for vulnerabilities.
The function doesn't check for reentrancy.
</think>

Based on my analysis, I found a reentrancy vulnerability in the withdraw function."#;

        let (reasoning, text) = model.extract_thinking_blocks(content);

        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("analyze this code"));
        assert!(text.contains("Based on my analysis"));
        assert!(!text.contains("<think>"));
    }

    #[test]
    fn test_extract_thinking_blocks_multiple() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<think>
First thought process.
</think>

Some intermediate text.

<think>
Second thought process.
</think>

Final response text."#;

        let (reasoning, text) = model.extract_thinking_blocks(content);

        assert!(reasoning.is_some());
        let reasoning = reasoning.unwrap();
        assert!(reasoning.contains("First thought"));
        assert!(reasoning.contains("Second thought"));
        assert!(text.contains("intermediate text"));
        assert!(text.contains("Final response"));
        assert!(!text.contains("<think>"));
    }

    #[test]
    fn test_extract_thinking_blocks_none() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = "This is a response without any thinking blocks.";
        let (reasoning, text) = model.extract_thinking_blocks(content);

        assert!(reasoning.is_none());
        assert_eq!(text, content);
    }

    #[test]
    fn test_extract_thinking_blocks_with_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<think>
I need to read the file to analyze it.
</think>

<minimax:tool_call>
<invoke name="read">
<parameter name="path">/src/contract.sol</parameter>
</invoke>
</minimax:tool_call>"#;

        let (reasoning, text) = model.extract_thinking_blocks(content);

        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("need to read"));
        assert!(text.contains("<minimax:tool_call>"));
    }

    #[test]
    fn test_builder_glm_template() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::GLM);
    }

    #[test]
    fn test_builder_minimax25_template() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Minimax25)
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Minimax25);
    }

    #[test]
    fn test_builder_auto_detect_minimax25() {
        let client = MlxLmClientBuilder::new()
            .auto_chat_template("MiniMaxAI/MiniMax-M2.5")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Minimax25);

        // M2.1 should still get old Minimax
        let client = MlxLmClientBuilder::new()
            .auto_chat_template("MiniMax/MiniMax-M2.1")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Minimax);
    }

    #[test]
    fn test_parse_glm_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value>"#;

        let tool_call = model.parse_glm_tool_call(content, 0).unwrap();
        assert_eq!(tool_call.name, "get_weather");
        assert_eq!(tool_call.id, "call_0");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_parse_glm_content_with_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"I'll check the weather for you.

<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>"#;

        let (text, tool_calls) = model.parse_glm_content(content);
        assert_eq!(text, "I'll check the weather for you.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_parse_glm_content_no_tool_calls() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        let content = "This is just a regular response without any tool calls.";
        let (text, tool_calls) = model.parse_glm_content(content);

        assert_eq!(text, content);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_parse_glm_content_multiple_tool_calls() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<tool_call>read_file<arg_key>path</arg_key><arg_value>/src/main.rs</arg_value></tool_call>
<tool_call>list_dir<arg_key>path</arg_key><arg_value>/src</arg_value></tool_call>"#;

        let (text, tool_calls) = model.parse_glm_content(content);
        assert!(text.is_empty());
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].id, "call_0");
        assert_eq!(tool_calls[1].name, "list_dir");
        assert_eq!(tool_calls[1].id, "call_1");
    }

    #[test]
    fn test_parse_glm_content_with_thinking_and_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        // First extract thinking, then parse the remaining content
        let content = r#"<think>
I need to read the file to analyze it.
</think>

<tool_call>read<arg_key>path</arg_key><arg_value>/src/contract.sol</arg_value></tool_call>"#;

        let (reasoning, text_without_thinking) = model.extract_thinking_blocks(content);
        let (text, tool_calls) = model.parse_glm_content(&text_without_thinking);

        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("need to read"));
        assert!(text.is_empty());
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read");
    }

    #[test]
    fn test_parse_glm_tool_call_with_json_value() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .build();
        let model = client.completion_model("test-model");

        // Test with a value that looks like JSON (number)
        let content = r#"search<arg_key>query</arg_key><arg_value>rust async</arg_value><arg_key>limit</arg_key><arg_value>10</arg_value>"#;

        let tool_call = model.parse_glm_tool_call(content, 0).unwrap();
        assert_eq!(tool_call.name, "search");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["query"], "rust async");
        assert_eq!(args["limit"], 10); // Should be parsed as number
    }

    #[test]
    fn test_chat_template_from_model_name_glm() {
        assert_eq!(
            ChatTemplate::from_model_name("zai-org/GLM-4.7"),
            ChatTemplate::GLM
        );
        assert_eq!(
            ChatTemplate::from_model_name("THUDM/chatglm3-6b"),
            ChatTemplate::GLM
        );
        assert_eq!(
            ChatTemplate::from_model_name("mlx-community/glm-4-9b-chat"),
            ChatTemplate::GLM
        );
    }

    #[test]
    fn test_chat_template_from_model_name_minimax() {
        assert_eq!(
            ChatTemplate::from_model_name("MiniMax/MiniMax-M1-40k"),
            ChatTemplate::Minimax
        );
        assert_eq!(
            ChatTemplate::from_model_name("minimax-text-01"),
            ChatTemplate::Minimax
        );
        assert_eq!(
            ChatTemplate::from_model_name("abab6.5s-chat"),
            ChatTemplate::Minimax
        );
        assert_eq!(
            ChatTemplate::from_model_name("MiniMax/MiniMax-M2.1"),
            ChatTemplate::Minimax
        );
    }

    #[test]
    fn test_chat_template_from_model_name_minimax25() {
        assert_eq!(
            ChatTemplate::from_model_name("MiniMaxAI/MiniMax-M2.5"),
            ChatTemplate::Minimax25
        );
        assert_eq!(
            ChatTemplate::from_model_name("sombra/MiniMax-M2.5-Q8"),
            ChatTemplate::Minimax25
        );
        assert_eq!(
            ChatTemplate::from_model_name("MiniMax-M3-something"),
            ChatTemplate::Minimax25
        );
        // Model path without "M" prefix (e.g., MiniMax-2.5-MXFP8)
        assert_eq!(
            ChatTemplate::from_model_name("/models/sombra/MiniMax-2.5-MXFP8"),
            ChatTemplate::Minimax25
        );
    }

    #[test]
    fn test_chat_template_from_model_name_chatml() {
        assert_eq!(
            ChatTemplate::from_model_name("Qwen/Qwen2.5-7B-Instruct"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_model_name("mlx-community/Qwen2-7B"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_model_name("IQuest-v1-chatml"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_chat_template_from_model_name_openai_fallback() {
        assert_eq!(
            ChatTemplate::from_model_name("gpt-4-turbo"),
            ChatTemplate::OpenAI
        );
        assert_eq!(
            ChatTemplate::from_model_name("llama-3.1-70b"),
            ChatTemplate::OpenAI
        );
        assert_eq!(
            ChatTemplate::from_model_name("unknown-model"),
            ChatTemplate::OpenAI
        );
    }

    #[test]
    fn test_chat_template_from_template_content_glm() {
        let glm_template = r#"[gMASK]<sop>
{%- if tools -%}
<|system|>
# Tools
{% endif %}"#;
        assert_eq!(
            ChatTemplate::from_template_content(glm_template),
            ChatTemplate::GLM
        );

        let glm_template2 = r#"<|observation|><tool_response>..."#;
        assert_eq!(
            ChatTemplate::from_template_content(glm_template2),
            ChatTemplate::GLM
        );
    }

    #[test]
    fn test_chat_template_from_template_content_minimax() {
        // M2.1 template: has minimax markers but no message.tool_calls access
        let minimax_template = r#"<minimax:tool_call>
<invoke name="test">
</invoke>
</minimax:tool_call>"#;
        assert_eq!(
            ChatTemplate::from_template_content(minimax_template),
            ChatTemplate::Minimax
        );
    }

    #[test]
    fn test_chat_template_from_template_content_minimax25() {
        // M2.5 template: has minimax markers AND message.tool_calls access
        let minimax25_template = r#"]~b]ai
{%- if message.tool_calls -%}
<minimax:tool_call>
{%- for tool_call in message.tool_calls -%}
<invoke name="{{ tool_call.name }}">
</invoke>
{%- endfor -%}
</minimax:tool_call>
{%- endif -%}
[e~["#;
        assert_eq!(
            ChatTemplate::from_template_content(minimax25_template),
            ChatTemplate::Minimax25
        );
    }

    #[test]
    fn test_chat_template_from_template_content_chatml() {
        let chatml_template = r#"<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user"#;
        assert_eq!(
            ChatTemplate::from_template_content(chatml_template),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_chat_template_from_template_content_openai_fallback() {
        let generic_template = r#"{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}"#;
        assert_eq!(
            ChatTemplate::from_template_content(generic_template),
            ChatTemplate::OpenAI
        );
    }

    #[test]
    fn test_builder_auto_chat_template() {
        let client = MlxLmClientBuilder::new()
            .auto_chat_template("zai-org/GLM-4.7")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::GLM);

        let client = MlxLmClientBuilder::new()
            .auto_chat_template("Qwen/Qwen2.5-7B")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::ChatML);
    }

    #[test]
    fn test_builder_auto_chat_template_from_content() {
        let client = MlxLmClientBuilder::new()
            .auto_chat_template_from_content("[gMASK]<sop>{% for m in messages %}")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::GLM);

        let client = MlxLmClientBuilder::new()
            .auto_chat_template_from_content("<|im_start|>system")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::ChatML);
    }

    #[test]
    fn test_frequency_penalty_clamping() {
        // Test lower bound
        let client = MlxLmClientBuilder::new().frequency_penalty(-3.0).build();
        assert_eq!(client.inner.frequency_penalty, Some(-2.0));

        // Test upper bound
        let client = MlxLmClientBuilder::new().frequency_penalty(3.0).build();
        assert_eq!(client.inner.frequency_penalty, Some(2.0));

        // Test valid value
        let client = MlxLmClientBuilder::new().frequency_penalty(0.5).build();
        assert_eq!(client.inner.frequency_penalty, Some(0.5));

        // Test negative valid value
        let client = MlxLmClientBuilder::new().frequency_penalty(-1.0).build();
        assert_eq!(client.inner.frequency_penalty, Some(-1.0));
    }

    #[test]
    fn test_presence_penalty_clamping() {
        // Test lower bound
        let client = MlxLmClientBuilder::new().presence_penalty(-3.0).build();
        assert_eq!(client.inner.presence_penalty, Some(-2.0));

        // Test upper bound
        let client = MlxLmClientBuilder::new().presence_penalty(3.0).build();
        assert_eq!(client.inner.presence_penalty, Some(2.0));

        // Test valid value
        let client = MlxLmClientBuilder::new().presence_penalty(0.5).build();
        assert_eq!(client.inner.presence_penalty, Some(0.5));

        // Test negative valid value
        let client = MlxLmClientBuilder::new().presence_penalty(-1.5).build();
        assert_eq!(client.inner.presence_penalty, Some(-1.5));
    }

    #[test]
    fn test_min_p_clamping() {
        // Test lower bound
        let client = MlxLmClientBuilder::new().min_p(-0.5).build();
        assert_eq!(client.inner.min_p, Some(0.0));

        // Test upper bound
        let client = MlxLmClientBuilder::new().min_p(1.5).build();
        assert_eq!(client.inner.min_p, Some(1.0));

        // Test valid value
        let client = MlxLmClientBuilder::new().min_p(0.05).build();
        assert_eq!(client.inner.min_p, Some(0.05));
    }

    #[test]
    fn test_repetition_context_size() {
        // Test positive value
        let client = MlxLmClientBuilder::new()
            .repetition_context_size(256)
            .build();
        assert_eq!(client.inner.repetition_context_size, Some(256));

        // Test -1 (full context)
        let client = MlxLmClientBuilder::new()
            .repetition_context_size(-1)
            .build();
        assert_eq!(client.inner.repetition_context_size, Some(-1));

        // Test 0 (disabled)
        let client = MlxLmClientBuilder::new().repetition_context_size(0).build();
        assert_eq!(client.inner.repetition_context_size, Some(0));
    }

    #[test]
    fn test_request_serialization_includes_anti_repetition_params() {
        // Verify that anti-repetition parameters actually appear in the serialized JSON
        // sent to the mlx-lm server (only repetition_penalty and repetition_context_size
        // are honored by mlx-lm; frequency_penalty/presence_penalty are ignored by the server)
        let request = MlxLmRequest {
            model: "test-model".to_string(),
            messages: vec![MlxLmMessage {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                ..Default::default()
            }],
            temperature: Some(0.1),
            max_tokens: Some(500),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.15),
            repetition_context_size: Some(64),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            min_p: Some(0.05),
            stop: None,
            tools: None,
            tool_choice: None,
        };

        let json = serde_json::to_value(&request).unwrap();

        // These are honored by mlx-lm server
        assert_eq!(json["temperature"], 0.1);
        assert_eq!(json["top_p"], 0.9);
        assert_eq!(json["top_k"], 40);
        assert_eq!(json["repetition_penalty"], 1.15);
        assert_eq!(json["repetition_context_size"], 64);
        assert_eq!(json["min_p"], 0.05);

        // These are sent but silently ignored by mlx-lm server
        assert_eq!(json["frequency_penalty"], 0.5);
        assert_eq!(json["presence_penalty"], 0.3);
    }

    #[test]
    fn test_request_serialization_omits_none_fields() {
        // Verify that None fields are omitted from JSON (skip_serializing_if = "Option::is_none")
        let request = MlxLmRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: Some(0.1),
            max_tokens: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            repetition_context_size: None,
            frequency_penalty: None,
            presence_penalty: None,
            min_p: None,
            stop: None,
            tools: None,
            tool_choice: None,
        };

        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(json["temperature"], 0.1);
        assert!(json.get("repetition_penalty").is_none());
        assert!(json.get("repetition_context_size").is_none());
        assert!(json.get("frequency_penalty").is_none());
        assert!(json.get("presence_penalty").is_none());
        assert!(json.get("min_p").is_none());
    }

    #[test]
    fn test_build_glm_request_passes_anti_repetition_params() {
        // Verify the GLM request builder wires anti-repetition params from inner config
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::GLM)
            .repetition_penalty(1.15)
            .repetition_context_size(64)
            .min_p(0.05)
            .build();
        let model = client.completion_model("glm-4.7");

        let request = CompletionRequest {
            preamble: Some("You are helpful.".to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };

        let mlxlm_request = model.build_glm_request(&request, &client.inner);
        let json = serde_json::to_value(&mlxlm_request).unwrap();

        assert_eq!(json["repetition_penalty"], 1.15);
        assert_eq!(json["repetition_context_size"], 64);
        assert_eq!(json["min_p"], 0.05);
    }

    #[test]
    fn test_builder_qwen35_template() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Qwen35);
    }

    #[test]
    fn test_chat_template_from_model_name_qwen35() {
        // Qwen 3.5 models
        assert_eq!(
            ChatTemplate::from_model_name("Qwen/Qwen3.5-397B-A17B"),
            ChatTemplate::Qwen35
        );
        assert_eq!(
            ChatTemplate::from_model_name("Qwen3.5-7B"),
            ChatTemplate::Qwen35
        );
        // Qwen 3 models (also Qwen35 template)
        assert_eq!(
            ChatTemplate::from_model_name("Qwen3-30B-A3B"),
            ChatTemplate::Qwen35
        );
        // Qwen 4+ future models
        assert_eq!(
            ChatTemplate::from_model_name("Qwen4-100B"),
            ChatTemplate::Qwen35
        );
        // Hyphenated model names (qwen-3.5-*)
        assert_eq!(
            ChatTemplate::from_model_name("sombra/qwen-3.5-large"),
            ChatTemplate::Qwen35
        );
        assert_eq!(
            ChatTemplate::from_model_name("qwen_3-instruct"),
            ChatTemplate::Qwen35
        );
        // Qwen 2.5 should still use ChatML
        assert_eq!(
            ChatTemplate::from_model_name("Qwen/Qwen2.5-7B-Instruct"),
            ChatTemplate::ChatML
        );
        // Qwen 2 should still use ChatML
        assert_eq!(
            ChatTemplate::from_model_name("mlx-community/Qwen2-7B"),
            ChatTemplate::ChatML
        );
        // Directory name should not affect model detection (basename-only matching)
        assert_eq!(
            ChatTemplate::from_model_name("/models/qwen3-cache/Qwen2.5-7B"),
            ChatTemplate::ChatML
        );
        // Size suffixes should NOT be mistaken for version numbers
        assert_eq!(
            ChatTemplate::from_model_name("Qwen-7B-Chat"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_model_name("Qwen-72B-Instruct"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_model_name("mlx-community/qwen-30b"),
            ChatTemplate::ChatML
        );
        // Trailing slash should not break detection
        assert_eq!(
            ChatTemplate::from_model_name("/models/Qwen3.5-7B/"),
            ChatTemplate::Qwen35
        );
        assert_eq!(
            ChatTemplate::from_model_name("Qwen/Qwen2.5-7B-Instruct/"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_chat_template_from_template_content_qwen35() {
        let qwen35_template = r#"<|im_start|>system
You have access to the following functions:
<tools>
</tools>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
<|im_end|>"#;
        assert_eq!(
            ChatTemplate::from_template_content(qwen35_template),
            ChatTemplate::Qwen35
        );
    }

    #[test]
    fn test_parse_qwen35_tool_call_single() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
celsius
</parameter>
</function>"#;

        let tool_call = model.parse_qwen35_tool_call(content, 0).unwrap();
        assert_eq!(tool_call.name, "get_weather");
        assert_eq!(tool_call.id, "call_0");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_parse_qwen35_tool_call_multiline_value() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<function=write_file>
<parameter=path>
/src/main.rs
</parameter>
<parameter=content>
fn main() {
    println!("hello");
}
</parameter>
</function>"#;

        let tool_call = model.parse_qwen35_tool_call(content, 0).unwrap();
        assert_eq!(tool_call.name, "write_file");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["path"], "/src/main.rs");
        // Verify internal whitespace is preserved (indentation matters for code)
        let content_val = args["content"].as_str().unwrap();
        assert!(
            content_val.contains("    println!"),
            "indentation should be preserved"
        );
        assert!(
            content_val.starts_with("fn main()"),
            "leading newline stripped but content preserved"
        );
    }

    #[test]
    fn test_parse_qwen35_content_with_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"I'll check the weather for you.

<tool_call>
<function=get_weather>
<parameter=location>Paris</parameter>
</function>
</tool_call>"#;

        let (text, tool_calls) = model.parse_qwen35_content(content);
        assert_eq!(text, "I'll check the weather for you.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_parse_qwen35_content_no_tool_calls() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = "This is just a regular response without any tool calls.";
        let (text, tool_calls) = model.parse_qwen35_content(content);

        assert_eq!(text, content);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_parse_qwen35_content_multiple_tool_calls() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<tool_call>
<function=read_file>
<parameter=path>/src/main.rs</parameter>
</function>
</tool_call>
<tool_call>
<function=list_dir>
<parameter=path>/src</parameter>
</function>
</tool_call>"#;

        let (text, tool_calls) = model.parse_qwen35_content(content);
        assert!(text.is_empty());
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].id, "call_0");
        assert_eq!(tool_calls[1].name, "list_dir");
        assert_eq!(tool_calls[1].id, "call_1");
    }

    #[test]
    fn test_parse_qwen35_content_with_thinking_and_tool_call() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        // First extract thinking, then parse the remaining content
        let content = r#"<think>
I need to read the file to analyze it.
</think>

<tool_call>
<function=read>
<parameter=path>/src/contract.sol</parameter>
</function>
</tool_call>"#;

        let (reasoning, text_without_thinking) = model.extract_thinking_blocks(content);
        let (text, tool_calls) = model.parse_qwen35_content(&text_without_thinking);

        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("need to read"));
        assert!(text.is_empty());
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read");
    }

    #[test]
    fn test_parse_qwen35_content_falls_back_to_json() {
        // Qwen3.5 parser should fall back to ChatML JSON format if <function= is not found
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        let content = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>"#;

        let (text, tool_calls) = model.parse_qwen35_content(content);
        assert!(text.is_empty());
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_parse_qwen35_tool_call_with_json_value() {
        let client = MlxLmClientBuilder::new()
            .chat_template(ChatTemplate::Qwen35)
            .build();
        let model = client.completion_model("test-model");

        // Test with a numeric value that should be parsed as a number
        let content = r#"<function=search>
<parameter=query>rust async</parameter>
<parameter=limit>10</parameter>
</function>"#;

        let tool_call = model.parse_qwen35_tool_call(content, 0).unwrap();
        assert_eq!(tool_call.name, "search");

        let args: serde_json::Value = serde_json::from_str(&tool_call.arguments).unwrap();
        assert_eq!(args["query"], "rust async");
        assert_eq!(args["limit"], 10); // Should be parsed as number
    }

    #[test]
    fn test_builder_auto_detect_qwen35() {
        let client = MlxLmClientBuilder::new()
            .auto_chat_template("Qwen/Qwen3.5-397B-A17B")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Qwen35);

        let client = MlxLmClientBuilder::new()
            .auto_chat_template("Qwen3-30B-A3B")
            .build();
        assert_eq!(client.inner.chat_template, ChatTemplate::Qwen35);
    }
}
