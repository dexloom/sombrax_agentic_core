//! LMStudio request/response types
//!
//! LMStudio uses an OpenAI-compatible API format with additional parameters
//! for anti-repetition control (`repeat_penalty`, `repeat_last_n`,
//! `frequency_penalty`, `presence_penalty`, `min_p`).

use serde::{Deserialize, Serialize};

/// LMStudio request (OpenAI-compatible format with anti-repetition extensions)
///
/// Field order is intentional: tools come before messages for better prompt caching
/// (tools are more stable and can be cached, while messages change frequently)
#[derive(Debug, Clone, Serialize)]
pub struct LmStudioRequest {
    /// Model identifier
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions (placed before messages for prompt caching)
    pub tools: Option<Vec<LmStudioTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode
    pub tool_choice: Option<LmStudioToolChoice>,
    /// Conversation messages
    pub messages: Vec<LmStudioMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling temperature
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum tokens in the response
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Nucleus sampling probability
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Top-k sampling parameter
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Repeat penalty (LMStudio field name, applied to prompt+output tokens)
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// How many recent tokens to consider for repeat_penalty.
    /// -1 = full context, 0 = disabled, positive = last N tokens.
    /// Serialized as `repeat_last_n` for llama.cpp backend.
    #[serde(rename = "repeat_last_n")]
    pub repetition_context_size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Frequency penalty (-2.0 to 2.0, penalizes based on token frequency in output)
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Presence penalty (-2.0 to 2.0, penalizes based on token presence in output)
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Minimum probability floor for sampling (0.0-1.0)
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Stop sequences
    pub stop: Option<Vec<String>>,
}

/// LMStudio message
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LmStudioMessage {
    #[serde(default)]
    /// Message role (system, user, assistant, tool)
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Text content
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Reasoning/thinking content
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Alternative field name for reasoning content
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool calls from the assistant
    pub tool_calls: Option<Vec<LmStudioToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results
    pub tool_call_id: Option<String>,
}

/// LMStudio tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioTool {
    #[serde(rename = "type")]
    /// Tool type label
    pub tool_type: String,
    /// Tool function definition
    pub function: LmStudioFunction,
}

/// LMStudio function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioFunction {
    /// Function name
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments
    pub parameters: Option<serde_json::Value>,
}

/// LMStudio tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LmStudioToolChoice {
    /// Choice encoded as a string ("auto", "none", "required")
    String(String),
    /// Choice encoded as an object payload
    Object {
        #[serde(rename = "type")]
        /// Choice type label
        choice_type: String,
        /// Selected function
        function: LmStudioToolChoiceFunction,
    },
}

/// LMStudio tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioToolChoiceFunction {
    /// Function name to force
    pub name: String,
}

/// LMStudio tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioToolCall {
    /// Tool call identifier
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label
    pub call_type: String,
    /// Function invocation details
    pub function: LmStudioFunctionCall,
}

/// LMStudio function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioFunctionCall {
    /// Function name
    pub name: String,
    /// JSON-encoded arguments
    pub arguments: String,
}

/// LMStudio response (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioResponse {
    /// Response identifier
    #[serde(default)]
    pub id: String,
    /// Object type
    #[serde(default)]
    pub object: String,
    /// Creation timestamp
    #[serde(default)]
    pub created: u64,
    /// Model identifier
    #[serde(default)]
    pub model: String,
    /// Response choices
    pub choices: Vec<LmStudioChoice>,
    /// Token usage
    #[serde(default)]
    pub usage: LmStudioUsage,
}

/// LMStudio response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioChoice {
    /// Choice index
    #[serde(default)]
    pub index: u32,
    /// Assistant message
    pub message: LmStudioMessage,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// LMStudio token usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LmStudioUsage {
    /// Prompt token count
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Completion token count
    #[serde(default)]
    pub completion_tokens: u64,
    /// Total token count
    #[serde(default)]
    pub total_tokens: u64,
}
