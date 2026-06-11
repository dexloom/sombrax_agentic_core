//! MLX-LM request/response types
//!
//! MLX-LM uses an OpenAI-compatible API format, but with support for
//! custom chat templates that may require different message formatting.

use serde::{Deserialize, Serialize};

/// Chat template variants supported by MLX-LM
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Standard OpenAI-compatible format (default)
    #[default]
    OpenAI,
    /// Minimax M2.1 template format with special delimiters
    /// Uses ]~b]role and [e~[ markers, <minimax:tool_call> for tools
    /// Pre-renders messages client-side (tool calls in content, "ai" role, "user" for tool results)
    Minimax,
    /// Minimax M2.5+ template format
    /// Uses standard OpenAI request format (tool_calls in field, "assistant" role, "tool" role)
    /// Server-side Jinja template handles Minimax-specific rendering
    /// Response parsing handles XML tool calls and `<think>` reasoning blocks
    Minimax25,
    /// ChatML template format (IQuest, Qwen 1/2.x, etc.)
    /// Uses <|im_start|> and <|im_end|> delimiters with XML tool formatting
    /// Pre-renders messages client-side to avoid server-side Jinja template issues
    ChatML,
    /// Qwen 3.5+ template format
    /// Uses <|im_start|> and <|im_end|> delimiters (ChatML-based) with:
    /// - `<think>...</think>` reasoning blocks
    /// - Parameter-based tool calls: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    /// - Tool responses wrapped in `<tool_response>` tags within user messages
    ///
    /// Pre-renders messages client-side because the server-side Jinja template expects
    /// `tool_call.arguments` as a dict (iterable with `|items`), but the OpenAI format
    /// sends arguments as a JSON string, which causes server exceptions.
    Qwen35,
    /// GLM template format (GLM-4.7, etc.)
    /// Uses <|user|>, <|assistant|>, <|system|>, <|tool|> role markers
    /// Tool calls use <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
    /// Supports <think>...</think> reasoning blocks
    GLM,
    /// Gemma template format (Gemma 4, etc.)
    /// Uses `<|turn>role\n...<turn|>\n` delimiters with:
    /// - `assistant` mapped to `model` role
    /// - Tool definitions: `<|tool>declaration:name{...}<tool|>`
    /// - Tool calls: `<|tool_call>call:name{key:value,...}<tool_call|>`
    /// - Tool responses: `<|tool_response>response:name{...}<tool_response|>`
    /// - Thinking via `<|channel>thought\n...<channel|>` (not `<think>`)
    /// - Strings quoted with `<|"|>` instead of `"`
    ///
    /// Pre-renders messages client-side because Gemma uses a custom key-value format
    /// for tool definitions and calls that differs from standard OpenAI JSON.
    Gemma,
}

impl ChatTemplate {
    /// Auto-detect chat template from model name/path
    ///
    /// Matches common model naming patterns:
    /// - GLM: "glm-4", "GLM-4.7", "chatglm", etc.
    /// - Minimax25: "MiniMax-M2.5", "MiniMax-M3", etc. (M2.5+)
    /// - Minimax: "minimax", "MiniMax-M1", "MiniMax-M2.1", "abab", etc. (pre-M2.5)
    /// - Qwen35: "Qwen3.5", "Qwen3", "Qwen4", etc. (Qwen 3+)
    /// - ChatML: "Qwen2.5", "qwen", "iquest", models with "chatml" in the name
    /// - OpenAI: default fallback for unrecognized models
    ///
    /// # Examples
    ///
    /// ```
    /// use sombrax_agentic_core::providers::mlxlm::ChatTemplate;
    ///
    /// assert_eq!(ChatTemplate::from_model_name("zai-org/GLM-4.7"), ChatTemplate::GLM);
    /// assert_eq!(ChatTemplate::from_model_name("mlx-community/Qwen2.5-7B"), ChatTemplate::ChatML);
    /// assert_eq!(ChatTemplate::from_model_name("Qwen/Qwen3.5-397B-A17B"), ChatTemplate::Qwen35);
    /// assert_eq!(ChatTemplate::from_model_name("Qwen3-30B-A3B"), ChatTemplate::Qwen35);
    /// assert_eq!(ChatTemplate::from_model_name("MiniMax/MiniMax-M1-40k"), ChatTemplate::Minimax);
    /// assert_eq!(ChatTemplate::from_model_name("MiniMaxAI/MiniMax-M2.5"), ChatTemplate::Minimax25);
    /// assert_eq!(ChatTemplate::from_model_name("gpt-4"), ChatTemplate::OpenAI);
    /// ```
    pub fn from_model_name(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Gemma models (gemma-4, gemma-4-26b, etc.)
        if model_lower.contains("gemma") {
            return ChatTemplate::Gemma;
        }

        // GLM models (GLM-4, GLM-4.7, ChatGLM, etc.)
        if model_lower.contains("glm") {
            return ChatTemplate::GLM;
        }

        // Minimax M2.5+ models use standard OpenAI format with server-side Jinja rendering
        // Match M2.5/2.5, M3/3, M4/4, etc. but NOT M1, M2, M2.1
        if model_lower.contains("minimax") || model_lower.contains("abab") {
            if model_lower.contains("m2.5")
                || model_lower.contains("-2.5")
                || model_lower.contains("m3")
                || model_lower.contains("-3-")
                || model_lower.contains("m4")
                || model_lower.contains("-4-")
            {
                return ChatTemplate::Minimax25;
            }
            return ChatTemplate::Minimax;
        }

        // Qwen models: detect version to route Qwen 3+ to Qwen35, older to ChatML
        // Use basename only to avoid false positives from directory names
        // (e.g., "/models/qwen3-cache/Qwen2.5-7B" should detect Qwen2.5, not Qwen3)
        // Note: file-path model IDs like "/models/Qwen3.5/model.gguf" won't match
        // since basename is "model.gguf" — use explicit chat_template config for those.
        let model_trimmed = model_lower.trim_end_matches('/');
        let model_basename = model_trimmed.rsplit('/').next().unwrap_or(model_trimmed);
        if model_basename.contains("qwen") {
            // Match Qwen3, Qwen3.5, Qwen4, etc. (version >= 3)
            // Pattern: "qwen" followed by a digit 3-9 (e.g., qwen3, qwen3.5, qwen4)
            // Excludes Qwen2, Qwen2.5, Qwen1.5 which use legacy ChatML
            // Pre-rendered client-side because the Jinja template expects arguments as
            // a dict (|items filter), but OpenAI format sends them as JSON strings.
            if Self::is_qwen_v3_plus(model_basename) {
                return ChatTemplate::Qwen35;
            }
            return ChatTemplate::ChatML;
        }

        // ChatML models (IQuest, or explicit chatml)
        if model_lower.contains("iquest") || model_lower.contains("chatml") {
            return ChatTemplate::ChatML;
        }

        // Default to OpenAI-compatible format
        ChatTemplate::OpenAI
    }

    /// Check if a Qwen model name indicates version 3+
    ///
    /// Handles both direct suffixes (qwen3, qwen3.5) and separator-based
    /// patterns (qwen-3.5, qwen_3, qwen/3.5).
    fn is_qwen_v3_plus(model_lower: &str) -> bool {
        // Find "qwen" and check the character(s) after it for version number
        let mut search_from = 0;
        while let Some(pos) = model_lower[search_from..].find("qwen") {
            let abs_pos = search_from + pos;
            let after = &model_lower[abs_pos + 4..];
            // Skip optional separator (-, _, /)
            let version_part = after.strip_prefix(['-', '_', '/']).unwrap_or(after);
            // Check for a digit >= 3 that isn't a size suffix (e.g., 7B, 30B, 70B)
            if let Some(first_char) = version_part.chars().next() {
                if first_char.is_ascii_digit() && first_char >= '3' {
                    // Consume all digits, then check for 'b' (size marker like 7B, 30B)
                    let rest = &version_part[first_char.len_utf8()..];
                    let after_digits = rest.trim_start_matches(|c: char| c.is_ascii_digit());
                    if after_digits.starts_with('b') {
                        // This is a size suffix (e.g., 7b, 30b, 70b), not a version
                        search_from = abs_pos + 4;
                        continue;
                    }
                    return true;
                }
            }
            search_from = abs_pos + 4;
        }
        false
    }

    /// Auto-detect chat template from Jinja template content
    ///
    /// Analyzes the template content for characteristic patterns:
    /// - GLM: `[gMASK]<sop>`, `<|observation|>`, `<arg_key>`, `<arg_value>`
    /// - Minimax: `<minimax:tool_call>`, `]~b]`, `[e~[`
    /// - Qwen35: ChatML markers + `<function=` parameter-based tool calls
    /// - ChatML: `<|im_start|>`, `<|im_end|>`
    /// - OpenAI: default fallback
    ///
    /// # Examples
    ///
    /// ```
    /// use sombrax_agentic_core::providers::mlxlm::ChatTemplate;
    ///
    /// let glm_template = "[gMASK]<sop>{% for m in messages %}<|user|>...";
    /// assert_eq!(ChatTemplate::from_template_content(glm_template), ChatTemplate::GLM);
    ///
    /// let chatml_template = "<|im_start|>system...";
    /// assert_eq!(ChatTemplate::from_template_content(chatml_template), ChatTemplate::ChatML);
    ///
    /// let qwen35_template = "<|im_start|>system\n<function=example>\n<parameter=foo>";
    /// assert_eq!(ChatTemplate::from_template_content(qwen35_template), ChatTemplate::Qwen35);
    /// ```
    pub fn from_template_content(content: &str) -> Self {
        // Gemma template markers: <|turn>, <|tool_call>, <|tool_response>, <|"|>
        if content.contains("<|turn>") && content.contains("<tool_call|>") {
            return ChatTemplate::Gemma;
        }

        // GLM template markers
        if content.contains("[gMASK]")
            || content.contains("<|observation|>")
            || (content.contains("<arg_key>") && content.contains("<arg_value>"))
        {
            return ChatTemplate::GLM;
        }

        // Minimax template markers
        if content.contains("<minimax:tool_call>")
            || content.contains("]~b]")
            || content.contains("[e~[")
        {
            // M2.5+ templates process tool_calls from the message field natively
            if content.contains("message.tool_calls") {
                return ChatTemplate::Minimax25;
            }
            return ChatTemplate::Minimax;
        }

        // ChatML-based templates: differentiate Qwen3.5+ from legacy ChatML
        if content.contains("<|im_start|>") || content.contains("<|im_end|>") {
            // Qwen3.5 uses <function=...> and <parameter=...> tags for tool calls
            if content.contains("<function=") && content.contains("<parameter=") {
                return ChatTemplate::Qwen35;
            }
            return ChatTemplate::ChatML;
        }

        // Default to OpenAI-compatible format
        ChatTemplate::OpenAI
    }
}

/// MLX-LM request (OpenAI-compatible format)
///
/// Field order is intentional: tools come before messages for better prompt caching
/// (tools are more stable and can be cached, while messages change frequently)
#[derive(Debug, Clone, Serialize)]
pub struct MlxLmRequest {
    /// Model identifier (often ignored by local server, but included for compatibility)
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions (placed before messages for prompt caching)
    pub tools: Option<Vec<MlxLmTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool selection mode
    pub tool_choice: Option<MlxLmToolChoice>,
    /// Conversation messages
    pub messages: Vec<MlxLmMessage>,
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
    /// Repetition penalty
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Repetition context size (how many recent tokens to check for repeats)
    pub repetition_context_size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Minimum probability floor for sampling (0.0-1.0)
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Stop sequences
    pub stop: Option<Vec<String>>,
}

/// MLX-LM message
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MlxLmMessage {
    #[serde(default)]
    /// Message role (system, user, assistant, tool)
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Text content
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Reasoning/thinking content (some models like MiniMax, DeepSeek return this separately)
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Alternative field name for reasoning content
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool calls from the assistant
    pub tool_calls: Option<Vec<MlxLmToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool call id for tool results
    pub tool_call_id: Option<String>,
}

/// MLX-LM tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmTool {
    #[serde(rename = "type")]
    /// Tool type label
    pub tool_type: String,
    /// Tool function definition
    pub function: MlxLmFunction,
}

/// MLX-LM function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmFunction {
    /// Function name
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional function description
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for arguments
    pub parameters: Option<serde_json::Value>,
}

/// MLX-LM tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MlxLmToolChoice {
    /// Choice encoded as a string ("auto", "none", "required")
    String(String),
    /// Choice encoded as an object payload
    Object {
        #[serde(rename = "type")]
        /// Choice type label
        choice_type: String,
        /// Selected function
        function: MlxLmToolChoiceFunction,
    },
}

/// MLX-LM tool choice function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmToolChoiceFunction {
    /// Function name to force
    pub name: String,
}

/// MLX-LM tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmToolCall {
    /// Tool call identifier
    pub id: String,
    #[serde(rename = "type")]
    /// Tool call type label
    pub call_type: String,
    /// Function invocation details
    pub function: MlxLmFunctionCall,
}

/// MLX-LM function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmFunctionCall {
    /// Function name
    pub name: String,
    /// JSON-encoded arguments
    pub arguments: String,
}

/// MLX-LM response (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmResponse {
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
    pub choices: Vec<MlxLmChoice>,
    /// Token usage
    #[serde(default)]
    pub usage: MlxLmUsage,
}

/// MLX-LM response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxLmChoice {
    /// Choice index
    #[serde(default)]
    pub index: u32,
    /// Assistant message
    pub message: MlxLmMessage,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// MLX-LM token usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MlxLmUsage {
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
