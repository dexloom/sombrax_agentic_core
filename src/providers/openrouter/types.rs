//! OpenRouter request/response types
//!
//! OpenRouter requires lenient parsing due to varying response formats
//! from different underlying providers.

use serde::{Deserialize, Serialize};

/// OpenRouter-specific request
#[derive(Debug, Clone, Serialize)]
pub struct OpenRouterRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OpenRouterMessage>,
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
    /// Top-k sampling parameter.
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions.
    pub tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<OpenRouterToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Provider routing configuration.
    pub provider: Option<super::routing::OpenRouterProviderConfig>,
}

/// OpenRouter message
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterMessage {
    /// Message role.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Text content.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool calls from the assistant.
    pub tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results.
    pub tool_call_id: Option<String>,
    /// Reasoning field (some providers use this)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Alternative reasoning field (some providers use this)
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Alternative reasoning payload.
    pub reasoning_content: Option<String>,
}

/// OpenRouter tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterTool {
    #[serde(rename = "type")]
    /// Tool type label.
    pub tool_type: String,
    /// Tool function definition.
    pub function: OpenRouterFunction,
}

/// OpenRouter function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterFunction {
    /// Function name.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description.
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments.
    pub parameters: Option<serde_json::Value>,
}

/// OpenRouter tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterToolChoice {
    /// Choice encoded as a string.
    String(String),
    /// Choice encoded as an object payload.
    Object {
        #[serde(rename = "type")]
        /// Choice type label.
        choice_type: String,
        /// Selected function.
        function: OpenRouterToolChoiceFunction,
    },
}

/// OpenRouter tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterToolChoiceFunction {
    /// Function name to force.
    pub name: String,
}

/// OpenRouter tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterToolCall {
    /// Tool call identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label.
    pub call_type: String,
    /// Function invocation details.
    pub function: OpenRouterFunctionCall,
}

/// OpenRouter function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterFunctionCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// OpenRouter-specific response (lenient parsing with optional fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterResponse {
    /// Response ID (optional - some providers omit)
    pub id: Option<String>,
    /// Choices array
    pub choices: Vec<OpenRouterChoice>,
    /// Usage statistics (optional - some providers omit)
    pub usage: Option<OpenRouterUsage>,
    /// System fingerprint (optional)
    pub system_fingerprint: Option<String>,
    /// Error in body (some providers return 200 with error)
    pub error: Option<OpenRouterError>,
}

/// OpenRouter response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChoice {
    /// Choice index.
    pub index: Option<u32>,
    /// Assistant message.
    pub message: OpenRouterMessage,
    /// Finish reason.
    pub finish_reason: Option<String>,
    /// Choice-level error (optional)
    pub error: Option<OpenRouterError>,
}

/// OpenRouter token usage (lenient)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterUsage {
    #[serde(default)]
    /// Prompt token count.
    pub prompt_tokens: u64,
    #[serde(default)]
    /// Completion token count.
    pub completion_tokens: u64,
    #[serde(default)]
    /// Total token count.
    pub total_tokens: u64,
    /// Prompt token details (includes cache info for OpenAI-compatible providers).
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
    /// Anthropic-style cache read tokens.
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Anthropic-style cache creation tokens.
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

/// OpenRouter prompt token details (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterPromptTokensDetails {
    /// Cached tokens (OpenAI format).
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// OpenRouter error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterError {
    /// Error message, if provided.
    pub message: Option<String>,
    /// Optional error code.
    pub code: Option<String>,
    #[serde(rename = "type")]
    /// Optional error type label.
    pub error_type: Option<String>,
}
