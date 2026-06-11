//! Cerebras request/response types

use serde::{Deserialize, Serialize};

/// Cerebras-specific request
#[derive(Debug, Clone, Serialize)]
pub struct CerebrasRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<CerebrasMessage>,
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
    pub tools: Option<Vec<CerebrasTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<CerebrasToolChoice>,
}

/// Cerebras message
/// CRITICAL: content must always be a simple string, never an array
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasMessage {
    /// Message role.
    pub role: String,
    /// Content MUST be a simple string, not an array (Cerebras quirk).
    /// Note: When the model uses tool calls, content may be absent in the response.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional tool calls from the assistant.
    pub tool_calls: Option<Vec<CerebrasToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results.
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional reasoning content (returned by thinking models like glm-4.7).
    pub reasoning: Option<String>,
}

/// Cerebras tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasTool {
    #[serde(rename = "type")]
    /// Tool type label.
    pub tool_type: String,
    /// Tool function definition.
    pub function: CerebrasFunction,
}

/// Cerebras function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasFunction {
    /// Function name.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description.
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments.
    pub parameters: Option<serde_json::Value>,
}

/// Cerebras tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CerebrasToolChoice {
    /// Choice encoded as a string.
    String(String),
    /// Choice encoded as an object payload.
    Object {
        #[serde(rename = "type")]
        /// Choice type label.
        choice_type: String,
        /// Selected function.
        function: CerebrasToolChoiceFunction,
    },
}

/// Cerebras tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasToolChoiceFunction {
    /// Function name to force.
    pub name: String,
}

/// Cerebras tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasToolCall {
    /// Tool call identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label.
    pub call_type: String,
    /// Function invocation details.
    pub function: CerebrasFunctionCall,
}

/// Cerebras function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasFunctionCall {
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// Cerebras-specific response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasResponse {
    /// Response identifier.
    pub id: String,
    /// Response choices.
    pub choices: Vec<CerebrasChoice>,
    /// Token usage.
    pub usage: CerebrasUsage,
}

/// Cerebras response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebrasChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: CerebrasMessage,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// Cerebras token usage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CerebrasUsage {
    /// Prompt token count.
    pub prompt_tokens: u64,
    /// Completion token count.
    pub completion_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Prompt token details (if available).
    #[serde(default)]
    pub prompt_tokens_details: Option<CerebrasPromptTokensDetails>,
}

/// Cerebras prompt token details
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CerebrasPromptTokensDetails {
    /// Cached tokens.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}
