//! MiniMax (Anthropic-compatible) request/response types

use serde::{Deserialize, Serialize};

/// MiniMax prompt-cache control marker (`{"type": "ephemeral"}`),
/// Anthropic-compatible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MinimaxCacheControl {
    /// Cache type — always `"ephemeral"`.
    #[serde(rename = "type")]
    pub cache_type: String,
}

impl MinimaxCacheControl {
    /// The 5-minute ephemeral cache breakpoint.
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// MiniMax system prompt — a plain string, or text blocks that can carry a
/// cache_control marker. Serializes untagged; `Text` is byte-identical to the
/// previous `Option<String>` representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MinimaxSystem {
    /// Plain system string (default, no caching).
    Text(String),
    /// System as text blocks (used to attach a cache_control marker).
    Blocks(Vec<MinimaxSystemBlock>),
}

/// A system text block (`{"type":"text","text":...,"cache_control"?:...}`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MinimaxSystemBlock {
    /// Block type — always `"text"`.
    #[serde(rename = "type")]
    pub block_type: String,
    /// System text payload.
    pub text: String,
    /// Optional cache breakpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<MinimaxCacheControl>,
}

/// MiniMax Messages API request (Anthropic-compatible)
#[derive(Debug, Clone, Serialize)]
pub struct MinimaxRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<MinimaxMessage>,
    /// Maximum tokens in the response.
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional system prompt.
    pub system: Option<MinimaxSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling temperature.
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Nucleus sampling probability.
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Top-k sampling parameter.
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions.
    pub tools: Option<Vec<MinimaxTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<MinimaxToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional request metadata.
    pub metadata: Option<MinimaxMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Thinking/reasoning mode configuration.
    pub thinking: Option<MinimaxThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Server-sent event streaming flag. When `Some(true)`, the response is
    /// delivered as Anthropic-shape SSE events instead of a buffered JSON body.
    /// Streaming is required for long-running local servers (e.g. mlx_fun)
    /// whose non-streaming handler buffers the entire response until end-of-turn.
    pub stream: Option<bool>,
}

/// MiniMax thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxThinkingConfig {
    #[serde(rename = "type")]
    /// Thinking mode type ("enabled" or "disabled").
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Budget tokens for thinking (required by some APIs when enabled).
    pub budget_tokens: Option<u64>,
}

impl MinimaxThinkingConfig {
    /// Create enabled thinking config with a budget derived from max_tokens
    pub fn enabled(budget_tokens: u64) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(budget_tokens),
        }
    }

    /// Create disabled thinking config
    pub fn disabled() -> Self {
        Self {
            thinking_type: "disabled".to_string(),
            budget_tokens: None,
        }
    }
}

/// MiniMax message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxMessage {
    /// Message role.
    pub role: String,
    /// Message content.
    pub content: MinimaxContent,
}

/// MiniMax content - can be string or array of content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MinimaxContent {
    /// Plain text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<MinimaxContentBlock>),
}

impl MinimaxContent {
    /// Extract text content
    pub fn text(&self) -> String {
        match self {
            MinimaxContent::Text(s) => s.clone(),
            MinimaxContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    MinimaxContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// MiniMax content block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MinimaxContentBlock {
    #[serde(rename = "text")]
    /// Text block.
    Text {
        /// Text payload.
        text: String,
        /// Optional cache breakpoint (skipped when `None`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<MinimaxCacheControl>,
    },
    #[serde(rename = "thinking")]
    /// Reasoning/thinking block. Never carries cache_control.
    Thinking {
        /// Thinking payload.
        thinking: String,
    },
    #[serde(rename = "tool_use")]
    /// Tool invocation block.
    ToolUse {
        /// Tool call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input payload.
        input: serde_json::Value,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<MinimaxCacheControl>,
    },
    #[serde(rename = "tool_result")]
    /// Tool result block.
    ToolResult {
        /// Tool call identifier.
        tool_use_id: String,
        /// Tool output content.
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Indicates the tool returned an error.
        is_error: Option<bool>,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<MinimaxCacheControl>,
    },
}

/// MiniMax tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Tool input schema.
    pub input_schema: serde_json::Value,
    /// Optional cache breakpoint (marks the tools+system prefix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<MinimaxCacheControl>,
}

/// MiniMax tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MinimaxToolChoice {
    #[serde(rename = "auto")]
    /// Let the model decide.
    Auto,
    #[serde(rename = "any")]
    /// Force tool usage.
    Any,
    #[serde(rename = "tool")]
    /// Require a specific tool.
    Tool {
        /// Tool name.
        name: String,
    },
}

/// MiniMax request metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional end-user identifier.
    pub user_id: Option<String>,
}

/// MiniMax Messages API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxResponse {
    /// Response identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Response type label.
    pub response_type: String,
    /// Message role.
    pub role: String,
    /// Response content blocks.
    pub content: Vec<MinimaxResponseContent>,
    /// Model identifier.
    pub model: String,
    /// Stop reason, if any.
    pub stop_reason: Option<String>,
    /// Stop sequence, if any.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: MinimaxUsage,
}

/// MiniMax response content block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MinimaxResponseContent {
    #[serde(rename = "text")]
    /// Text block.
    Text {
        /// Text payload.
        text: String,
    },
    #[serde(rename = "thinking")]
    /// Reasoning/thinking block.
    Thinking {
        /// Thinking payload.
        thinking: String,
    },
    #[serde(rename = "tool_use")]
    /// Tool invocation block.
    ToolUse {
        /// Tool call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input payload.
        input: serde_json::Value,
    },
}

/// MiniMax token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxUsage {
    /// Input token count.
    pub input_tokens: u64,
    /// Output token count.
    pub output_tokens: u64,
    /// Tokens read from cache (cache hits).
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Tokens written to cache (cache creation).
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}
