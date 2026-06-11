//! Ollama native `/api/chat` request/response wire types.
//!
//! These mirror Ollama's own protocol (NOT the OpenAI-compatible `/v1`
//! shim). Notable differences from OpenAI-style providers:
//!
//! - Tool calls carry **no `id`**; `function.arguments` is a JSON **object**.
//! - Sampling knobs live in an `options` sub-object.
//! - Reasoning is a first-class `thinking` string on the message.
//! - Token usage is `prompt_eval_count` / `eval_count`.

use serde::{Deserialize, Serialize};

/// `think` request parameter: either a boolean or a level string
/// (`"high"`, `"medium"`, `"low"`). Modeled untagged so it serializes
/// to a bare `true` / `"high"` as Ollama expects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OllamaThink {
    /// Plain on/off thinking toggle.
    Enabled(bool),
    /// Thinking effort level (`"high"` / `"medium"` / `"low"`).
    Level(String),
}

/// Ollama sampling options block.
#[derive(Debug, Clone, Serialize, Default)]
pub struct OllamaOptions {
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
    /// Minimum probability floor.
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Max output tokens (Ollama's name for `max_tokens`).
    pub num_predict: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Context window size.
    pub num_ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// RNG seed for reproducible outputs.
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
}

impl OllamaOptions {
    /// True when no option is set (so the whole block can be omitted).
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
            && self.num_predict.is_none()
            && self.num_ctx.is_none()
            && self.seed.is_none()
            && self.stop.is_none()
    }
}

/// Ollama `/api/chat` request body.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Available tool definitions.
    pub tools: Option<Vec<OllamaTool>>,
    /// Always `false` — sac providers are non-streaming.
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Thinking control (bool or level string).
    pub think: Option<OllamaThink>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Model residency hint (e.g. `"5m"`, `"0"`).
    pub keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling options.
    pub options: Option<OllamaOptions>,
}

/// Ollama message (request and response share this shape).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaMessage {
    /// Message role (`system` / `user` / `assistant` / `tool`).
    pub role: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    /// Text content. Absent when the assistant only emits tool calls.
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Native reasoning trace.
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool calls (assistant turns). No `id` in Ollama's protocol.
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Name of the tool a `role: "tool"` message answers.
    pub tool_name: Option<String>,
}

/// Ollama tool-call wrapper (no `id` field — unlike OpenAI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    /// The invoked function.
    pub function: OllamaFunctionCall,
}

/// Ollama function invocation. `arguments` is a JSON **object**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON object (not a string).
    pub arguments: serde_json::Value,
}

/// Ollama tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    /// Always `"function"`.
    pub tool_type: String,
    /// The function schema.
    pub function: OllamaFunction,
}

/// Ollama function schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// JSON schema for parameters.
    pub parameters: serde_json::Value,
}

/// Ollama `/api/chat` response body (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaResponse {
    #[serde(default)]
    /// Model that produced the response.
    pub model: String,
    /// The assistant message.
    pub message: OllamaMessage,
    #[serde(default)]
    /// Whether generation finished.
    pub done: bool,
    #[serde(default)]
    /// Why generation stopped (`"stop"`, `"length"`, …).
    pub done_reason: Option<String>,
    #[serde(default)]
    /// Prompt token count.
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    /// Generated token count.
    pub eval_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_serializes_untagged() {
        assert_eq!(
            serde_json::to_value(OllamaThink::Enabled(true)).unwrap(),
            serde_json::json!(true)
        );
        assert_eq!(
            serde_json::to_value(OllamaThink::Level("high".into())).unwrap(),
            serde_json::json!("high")
        );
    }

    #[test]
    fn request_omits_unset_optionals() {
        let req = OllamaRequest {
            model: "llama3.2".into(),
            messages: vec![OllamaMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            tools: None,
            stream: false,
            think: None,
            keep_alive: None,
            options: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("tools").is_none());
        assert!(v.get("think").is_none());
        assert!(v.get("options").is_none());
        assert!(v.get("keep_alive").is_none());
        assert_eq!(v["stream"], serde_json::json!(false));
    }

    #[test]
    fn request_includes_set_optionals() {
        let req = OllamaRequest {
            model: "gpt-oss:120b".into(),
            messages: vec![],
            tools: Some(vec![OllamaTool {
                tool_type: "function".into(),
                function: OllamaFunction {
                    name: "f".into(),
                    description: "d".into(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            stream: false,
            think: Some(OllamaThink::Level("medium".into())),
            keep_alive: Some("5m".into()),
            options: Some(OllamaOptions {
                temperature: Some(0.4),
                ..Default::default()
            }),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["think"], serde_json::json!("medium"));
        assert_eq!(v["keep_alive"], serde_json::json!("5m"));
        assert_eq!(v["options"]["temperature"], serde_json::json!(0.4));
        assert!(v["tools"].is_array());
    }

    #[test]
    fn options_is_empty() {
        assert!(OllamaOptions::default().is_empty());
        assert!(!OllamaOptions {
            top_k: Some(20),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn response_deserializes_with_thinking_and_tool_calls() {
        let raw = r#"{
          "model": "gpt-oss:120b",
          "created_at": "2025-10-17T23:14:07.414671Z",
          "message": {
            "role": "assistant",
            "content": "",
            "thinking": "let me call the tool",
            "tool_calls": [
              {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ]
          },
          "done": true,
          "done_reason": "stop",
          "prompt_eval_count": 11,
          "eval_count": 18
        }"#;
        let resp: OllamaResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(
            resp.message.thinking.as_deref(),
            Some("let me call the tool")
        );
        assert_eq!(resp.done_reason.as_deref(), Some("stop"));
        assert_eq!(resp.prompt_eval_count, Some(11));
        assert_eq!(resp.eval_count, Some(18));
        let tc = &resp.message.tool_calls.unwrap()[0];
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn response_tolerates_missing_optionals() {
        let raw = r#"{"message":{"role":"assistant","content":"hello"},"done":true}"#;
        let resp: OllamaResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.message.content, "hello");
        assert!(resp.done_reason.is_none());
        assert!(resp.prompt_eval_count.is_none());
    }
}
