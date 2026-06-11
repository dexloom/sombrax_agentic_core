//! Anthropic (Claude) request/response types

use serde::{Deserialize, Serialize};

/// Anthropic Messages API request
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AnthropicMessage>,
    /// Maximum tokens in the response.
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional system prompt.
    pub system: Option<String>,
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
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode.
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional request metadata.
    pub metadata: Option<AnthropicMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Thinking/reasoning mode configuration.
    pub thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Enable server-sent-events streaming.
    pub stream: Option<bool>,
}

/// Anthropic thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    /// Thinking mode type ("enabled" or "disabled").
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Budget tokens for thinking (required by some APIs when enabled).
    pub budget_tokens: Option<u64>,
}

impl AnthropicThinkingConfig {
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

/// Anthropic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// Message role.
    pub role: String,
    /// Message content.
    pub content: AnthropicContent,
}

/// Anthropic content - can be string or array of content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    /// Plain text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicContent {
    /// Extract text content
    pub fn text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Anthropic content block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
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
    },
}

/// Anthropic tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Tool input schema.
    pub input_schema: serde_json::Value,
}

/// Anthropic tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
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

/// Anthropic request metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional end-user identifier.
    pub user_id: Option<String>,
}

/// Anthropic Messages API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    /// Response identifier.
    pub id: String,
    #[serde(rename = "type")]
    /// Response type label.
    pub response_type: String,
    /// Message role.
    pub role: String,
    /// Response content blocks.
    pub content: Vec<AnthropicResponseContent>,
    /// Model identifier.
    pub model: String,
    /// Stop reason, if any.
    pub stop_reason: Option<String>,
    /// Stop sequence, if any.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: AnthropicUsage,
}

/// Anthropic response content block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContent {
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

/// Anthropic token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
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
