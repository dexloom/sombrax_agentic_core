//! ZAI request/response types

use serde::{Deserialize, Serialize};

/// ZAI-specific request
#[derive(Debug, Clone, Serialize)]
pub struct ZaiRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ZaiMessage>,
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
    pub tools: Option<Vec<ZaiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<ZaiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Thinking mode configuration.
    pub thinking: Option<ZaiThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Allow parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
}

/// ZAI thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiThinkingConfig {
    #[serde(rename = "type")]
    /// Thinking mode identifier.
    pub thinking_type: String,
}

impl ZaiThinkingConfig {
    /// Create enabled thinking config
    pub fn enabled() -> Self {
        Self {
            thinking_type: "enabled".to_string(),
        }
    }

    /// Create disabled thinking config
    pub fn disabled() -> Self {
        Self {
            thinking_type: "disabled".to_string(),
        }
    }
}

/// ZAI message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiMessage {
    /// Message role.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Text content.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool calls from the assistant.
    pub tool_calls: Option<Vec<ZaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results.
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional reasoning payload.
    pub reasoning_content: Option<String>,
}

/// ZAI tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiTool {
    #[serde(rename = "type")]
    /// Tool type label.
    pub tool_type: String,
    /// Tool function definition.
    pub function: ZaiFunction,
}

/// ZAI function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiFunction {
    /// Function name.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description.
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments.
    pub parameters: Option<serde_json::Value>,
}

/// ZAI tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ZaiToolChoice {
    /// Choice encoded as a string.
    String(String),
    /// Choice encoded as an object payload.
    Object {
        #[serde(rename = "type")]
        /// Choice type label.
        choice_type: String,
        /// Selected function.
        function: ZaiToolChoiceFunction,
    },
}

/// ZAI tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiToolChoiceFunction {
    /// Function name to force.
    pub name: String,
}

/// ZAI tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiToolCall {
    /// Tool call identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label.
    pub call_type: String,
    /// Function invocation details.
    pub function: ZaiFunctionCall,
}

/// ZAI function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiFunctionCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// ZAI-specific response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiResponse {
    /// Response identifier.
    pub id: String,
    /// Response choices.
    pub choices: Vec<ZaiChoice>,
    /// Token usage.
    pub usage: ZaiUsage,
}

/// ZAI response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: ZaiMessage,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// ZAI token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiUsage {
    /// Prompt token count.
    pub prompt_tokens: u64,
    /// Completion token count.
    pub completion_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Cached prompt tokens (if caching is enabled).
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// Prompt token details (OpenAI-compatible format).
    #[serde(default)]
    pub prompt_tokens_details: Option<ZaiPromptTokensDetails>,
}

/// ZAI prompt token details
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ZaiPromptTokensDetails {
    /// Cached tokens.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}
