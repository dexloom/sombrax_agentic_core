//! OpenAI request/response types

use serde::{Deserialize, Serialize};

/// OpenAI Chat Completions API request
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling temperature.
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum tokens in the response.
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Nucleus sampling probability.
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Top-k sampling cap. Not part of the canonical OpenAI Chat
    /// Completions schema but accepted as an extension by OpenAI-
    /// compatible local servers (mlx-lm forks, vllm, sglang,
    /// llama.cpp `--jinja`). OpenAI itself ignores unknown fields, so
    /// passing this is safe upstream.
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Repetition penalty (multiplicative). Same status as `top_k` —
    /// not in canonical OpenAI schema, honored by local OpenAI-shaped
    /// servers, ignored by upstream OpenAI.
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions.
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Allow parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// End-user identifier.
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Chat-template kwargs forwarded to OpenAI-compatible servers
    /// (mlx-lm forks, vllm, sglang, llama.cpp `--jinja`) whose Jinja
    /// chat template reads `enable_thinking`. Used to suppress the
    /// reasoning channel on thinking-by-default models like GLM-5.1 or
    /// Qwen-thinking, which would otherwise emit reasoning-only output
    /// and leave `content` empty.
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Server-sent event streaming flag. When `Some(true)`, the response
    /// is delivered as SSE chunks instead of a buffered JSON body.
    /// Required for long-running local servers (mlx_fun, vllm) whose
    /// non-streaming handler buffers everything until end-of-turn.
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Streaming usage-reporting opt-in. OpenAI sends a final usage frame
    /// only when `stream_options.include_usage` is true. Many
    /// OpenAI-compatible local servers also honor it.
    pub stream_options: Option<OpenAIStreamOptions>,
}

/// Options for streaming requests.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIStreamOptions {
    /// Whether to include a final usage frame in the stream.
    pub include_usage: bool,
}

/// OpenAI message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    /// Message role.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Text content.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool calls from the assistant.
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results.
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional name for the message author.
    pub name: Option<String>,
    /// Reasoning channel returned by some OpenAI-compatible servers
    /// fronting thinking models. Accepts either field name on inbound:
    ///   - `reasoning` — OpenRouter, llama.cpp `--jinja`, OpenAI o1
    ///   - `reasoning_content` — DeepSeek, MiniMax, GLM, Qwen3, mlx-lm
    ///
    /// Inbound deserialization merges both into this single field;
    /// outbound serialization skips it when None (we don't replay
    /// reasoning on follow-up turns, per OpenAI o1 contract).
    #[serde(
        default,
        alias = "reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning: Option<String>,
}

/// OpenAI tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    /// Tool type label.
    pub tool_type: String,
    /// Tool function definition.
    pub function: OpenAIFunction,
}

/// OpenAI function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    /// Function name.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description.
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments.
    pub parameters: Option<serde_json::Value>,
}

/// OpenAI tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    /// Choice encoded as a string.
    String(String),
    /// Choice encoded as an object payload.
    Object {
        #[serde(rename = "type")]
        /// Choice type label.
        choice_type: String,
        /// Selected function.
        function: OpenAIToolChoiceFunction,
    },
}

/// OpenAI tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolChoiceFunction {
    /// Function name to force.
    pub name: String,
}

/// OpenAI tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    /// Tool call identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label.
    pub call_type: String,
    /// Function invocation details.
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// OpenAI Chat Completions API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponse {
    /// Response identifier.
    pub id: String,
    /// Response object type.
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Model identifier.
    pub model: String,
    /// Response choices.
    pub choices: Vec<OpenAIChoice>,
    /// Token usage.
    pub usage: OpenAIUsage,
    #[serde(default)]
    /// Optional system fingerprint.
    pub system_fingerprint: Option<String>,
}

/// OpenAI response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: OpenAIMessage,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// OpenAI token usage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIUsage {
    /// Prompt token count.
    pub prompt_tokens: u64,
    /// Completion token count.
    pub completion_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Prompt token details (includes cache info).
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

/// OpenAI prompt token details
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIPromptTokensDetails {
    /// Cached tokens.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}
