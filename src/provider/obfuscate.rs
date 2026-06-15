//! Text obfuscation for completion models
//!
//! Provides the [`Obfuscator`] trait and [`ObfuscatingCompletionModel`] wrapper that
//! obfuscates all text content in requests before sending to the LLM and de-obfuscates
//! responses on return. This ensures sensitive text never reaches the LLM provider.
//!
//! # Built-in Implementations
//!
//! - [`MapObfuscator`]: Simple string replacement via `HashMap<String, String>`
//! - [`FnObfuscator`]: Custom closure-based obfuscation
//!
//! # Example
//!
//! ```rust,ignore
//! use sombrax_agentic_core::provider::{CompletionModelExt, MapObfuscator};
//! use std::collections::HashMap;
//!
//! let mut map = HashMap::new();
//! map.insert("0xdead...beef".into(), "CONTRACT_A".into());
//! map.insert("secret_key".into(), "REDACTED".into());
//!
//! let model = client
//!     .completion_model_adapter("gpt-4o")
//!     .with_obfuscator(MapObfuscator::new(map))
//!     .with_metrics();
//! ```

use crate::error::CompletionError;
use crate::message::{AssistantContent, Message, ToolCall, ToolCallFunction, UserContent};
use crate::provider::{CompletionModel, CompletionRequest, CompletionResponse};
use std::collections::HashMap;

/// Trait for bidirectional text obfuscation.
///
/// Implementors define how to replace sensitive text before sending to LLM inference
/// and how to restore original text when the response returns.
pub trait Obfuscator: Clone + Send + Sync + 'static {
    /// Obfuscate a text string before sending to the LLM.
    fn obfuscate(&self, text: &str) -> String;

    /// De-obfuscate a text string returned by the LLM.
    fn deobfuscate(&self, text: &str) -> String;
}

/// Simple string replacement obfuscator using a HashMap.
///
/// Each key in the map is replaced with its value when obfuscating,
/// and each value is replaced with its key when de-obfuscating.
///
/// Replacements are applied longest-key-first to prevent partial matches
/// when one key is a substring of another.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use sombrax_agentic_core::provider::MapObfuscator;
///
/// let mut map = HashMap::new();
/// map.insert("0xdead...beef".to_string(), "CONTRACT_A".to_string());
/// map.insert("secret_key_123".to_string(), "REDACTED_KEY".to_string());
///
/// let obfuscator = MapObfuscator::new(map);
/// assert_eq!(obfuscator.obfuscate("Call 0xdead...beef"), "Call CONTRACT_A");
/// assert_eq!(obfuscator.deobfuscate("Call CONTRACT_A"), "Call 0xdead...beef");
/// ```
#[derive(Clone, Debug)]
pub struct MapObfuscator {
    /// Forward mapping: original -> obfuscated (sorted longest-first)
    forward: Vec<(String, String)>,
    /// Reverse mapping: obfuscated -> original (sorted longest-first)
    reverse: Vec<(String, String)>,
}

impl MapObfuscator {
    /// Create a new map-based obfuscator.
    ///
    /// Keys are the original text to be replaced, values are the replacement tokens.
    pub fn new(map: HashMap<String, String>) -> Self {
        let mut forward: Vec<(String, String)> = map
            .iter()
            .filter(|(k, _)| !k.is_empty())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // Sort longest-key-first to prevent partial matches
        forward.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut reverse: Vec<(String, String)> = map
            .into_iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (v, k))
            .collect();
        // Sort longest-key-first for reverse too
        reverse.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Self { forward, reverse }
    }
}

/// Perform single-pass left-to-right replacement.
///
/// At each position in `text`, try each pattern in order (longest-first).
/// If a pattern matches, append its replacement and skip past the matched
/// characters. If no pattern matches, append the current character and
/// advance by one. Already-replaced text is never re-scanned.
fn single_pass_replace(text: &str, patterns: &[(String, String)]) -> String {
    if patterns.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let mut matched = false;
        for (pattern, replacement) in patterns {
            if text[i..].starts_with(pattern.as_str()) {
                result.push_str(replacement);
                i += pattern.len();
                matched = true;
                break; // longest-first sort means first match IS the longest
            }
        }
        if !matched {
            let ch = text[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    result
}

impl Obfuscator for MapObfuscator {
    fn obfuscate(&self, text: &str) -> String {
        single_pass_replace(text, &self.forward)
    }

    fn deobfuscate(&self, text: &str) -> String {
        single_pass_replace(text, &self.reverse)
    }
}

/// Closure-based obfuscator for custom obfuscation logic.
///
/// Wraps two functions: one for obfuscation and one for de-obfuscation.
///
/// # Example
///
/// ```rust,ignore
/// use sombrax_agentic_core::provider::FnObfuscator;
///
/// let obfuscator = FnObfuscator::new(
///     |text| text.replace("secret", "***"),
///     |text| text.replace("***", "secret"),
/// );
/// ```
#[derive(Clone)]
pub struct FnObfuscator<F, G>
where
    F: Fn(&str) -> String + Clone + Send + Sync + 'static,
    G: Fn(&str) -> String + Clone + Send + Sync + 'static,
{
    obfuscate_fn: F,
    deobfuscate_fn: G,
}

impl<F, G> FnObfuscator<F, G>
where
    F: Fn(&str) -> String + Clone + Send + Sync + 'static,
    G: Fn(&str) -> String + Clone + Send + Sync + 'static,
{
    /// Create a new closure-based obfuscator.
    pub fn new(obfuscate_fn: F, deobfuscate_fn: G) -> Self {
        Self {
            obfuscate_fn,
            deobfuscate_fn,
        }
    }
}

impl<F, G> Obfuscator for FnObfuscator<F, G>
where
    F: Fn(&str) -> String + Clone + Send + Sync + 'static,
    G: Fn(&str) -> String + Clone + Send + Sync + 'static,
{
    fn obfuscate(&self, text: &str) -> String {
        (self.obfuscate_fn)(text)
    }

    fn deobfuscate(&self, text: &str) -> String {
        (self.deobfuscate_fn)(text)
    }
}

/// Wrapper that adds text obfuscation to any [`CompletionModel`].
///
/// Obfuscates all text content in completion requests before sending to the LLM,
/// and de-obfuscates all text content in responses before returning to the caller.
///
/// This operates at the model level, below the agent and hook systems, ensuring
/// that sensitive text never reaches the LLM provider.
///
/// # What Gets Obfuscated
///
/// **Request (obfuscate):**
/// - System preamble
/// - User message text
/// - Tool result content
/// - Assistant text and tool call arguments (in conversation history)
/// - Reasoning content
///
/// **Response (de-obfuscate):**
/// - Assistant text
/// - Tool call arguments
/// - Reasoning content
///
/// **NOT modified:**
/// - Tool names (must remain valid for tool lookup)
/// - Tool call IDs
/// - Tool definitions/schemas
/// - Image/document binary data
/// - Usage stats, finish_reason
///
/// # Usage
///
/// ```rust,ignore
/// use sombrax_agentic_core::provider::{CompletionModelExt, MapObfuscator};
/// use std::collections::HashMap;
///
/// let mut map = HashMap::new();
/// map.insert("0xdead...beef".into(), "CONTRACT_A".into());
///
/// let model = client
///     .completion_model_adapter("gpt-4o")
///     .with_obfuscator(MapObfuscator::new(map))
///     .with_metrics();
///
/// let agent = Agent::builder(model)
///     .preamble("Analyze CONTRACT_A...")
///     .build();
/// ```
#[derive(Clone)]
pub struct ObfuscatingCompletionModel<M: CompletionModel, O: Obfuscator> {
    inner: M,
    obfuscator: O,
}

impl<M: CompletionModel, O: Obfuscator> ObfuscatingCompletionModel<M, O> {
    /// Create a new obfuscating wrapper.
    pub fn new(inner: M, obfuscator: O) -> Self {
        Self { inner, obfuscator }
    }

    /// Get a reference to the inner model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Unwrap and return the inner model.
    pub fn into_inner(self) -> M {
        self.inner
    }

    /// Obfuscate all text content in a CompletionRequest.
    fn obfuscate_request(&self, mut request: CompletionRequest) -> CompletionRequest {
        // Obfuscate preamble
        if let Some(ref preamble) = request.preamble {
            request.preamble = Some(self.obfuscator.obfuscate(preamble));
        }

        // Obfuscate messages
        request.messages = request
            .messages
            .into_iter()
            .map(|msg| self.obfuscate_message(msg))
            .collect();

        request
    }

    /// Obfuscate text content in a single Message.
    fn obfuscate_message(&self, msg: Message) -> Message {
        match msg {
            Message::User { content, id } => {
                let content = content
                    .into_iter()
                    .map(|c| match c {
                        UserContent::Text { text } => UserContent::Text {
                            text: self.obfuscator.obfuscate(&text),
                        },
                        UserContent::ToolResult { id, content } => UserContent::ToolResult {
                            id,
                            content: self.obfuscator.obfuscate(&content),
                        },
                        other => other,
                    })
                    .collect();
                Message::User { content, id }
            }
            Message::Assistant {
                content,
                id,
                reasoning,
            } => {
                let content = content
                    .into_iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => AssistantContent::Text {
                            text: self.obfuscator.obfuscate(&text),
                        },
                        AssistantContent::ToolCall(tc) => AssistantContent::ToolCall(ToolCall {
                            id: tc.id,
                            function: ToolCallFunction {
                                name: tc.function.name,
                                arguments: self.obfuscator.obfuscate(&tc.function.arguments),
                            },
                        }),
                        AssistantContent::Reasoning { reasoning } => AssistantContent::Reasoning {
                            reasoning: reasoning
                                .into_iter()
                                .map(|r| self.obfuscator.obfuscate(&r))
                                .collect(),
                        },
                    })
                    .collect();
                let reasoning = reasoning.map(|r| self.obfuscator.obfuscate(&r));
                Message::Assistant {
                    content,
                    id,
                    reasoning,
                }
            }
        }
    }

    /// De-obfuscate text content in a CompletionResponse.
    fn deobfuscate_response<R>(
        &self,
        mut response: CompletionResponse<R>,
    ) -> CompletionResponse<R> {
        response.message = self.deobfuscate_message(response.message);

        if let Some(ref reasoning) = response.reasoning_content {
            response.reasoning_content = Some(self.obfuscator.deobfuscate(reasoning));
        }

        response
    }

    /// De-obfuscate text content in a single Message.
    fn deobfuscate_message(&self, msg: Message) -> Message {
        match msg {
            Message::Assistant {
                content,
                id,
                reasoning,
            } => {
                let content = content
                    .into_iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => AssistantContent::Text {
                            text: self.obfuscator.deobfuscate(&text),
                        },
                        AssistantContent::ToolCall(tc) => AssistantContent::ToolCall(ToolCall {
                            id: tc.id,
                            function: ToolCallFunction {
                                name: tc.function.name,
                                arguments: self.obfuscator.deobfuscate(&tc.function.arguments),
                            },
                        }),
                        AssistantContent::Reasoning { reasoning } => AssistantContent::Reasoning {
                            reasoning: reasoning
                                .into_iter()
                                .map(|r| self.obfuscator.deobfuscate(&r))
                                .collect(),
                        },
                    })
                    .collect();
                let reasoning = reasoning.map(|r| self.obfuscator.deobfuscate(&r));
                Message::Assistant {
                    content,
                    id,
                    reasoning,
                }
            }
            // User messages in response are unexpected, pass through
            other => other,
        }
    }
}

impl<M: CompletionModel, O: Obfuscator> CompletionModel for ObfuscatingCompletionModel<M, O> {
    type Response = M::Response;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        tracing::debug!(
            model_id = %self.inner.model_id(),
            provider = %self.inner.provider(),
            message_count = request.messages.len(),
            "obfuscating completion request"
        );
        let obfuscated_request = self.obfuscate_request(request);
        let response = self.inner.completion(obfuscated_request).await?;
        Ok(self.deobfuscate_response(response))
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ToolCall;
    use crate::provider::Usage;

    // -- MapObfuscator tests --

    #[test]
    fn test_map_obfuscator_basic() {
        let mut map = HashMap::new();
        map.insert("secret".to_string(), "REDACTED".to_string());
        let obf = MapObfuscator::new(map);

        assert_eq!(obf.obfuscate("my secret data"), "my REDACTED data");
        assert_eq!(obf.deobfuscate("my REDACTED data"), "my secret data");
    }

    #[test]
    fn test_map_obfuscator_roundtrip() {
        let mut map = HashMap::new();
        map.insert("0xdead".to_string(), "ADDR_A".to_string());
        map.insert("0xbeef".to_string(), "ADDR_B".to_string());
        let obf = MapObfuscator::new(map);

        let original = "Transfer from 0xdead to 0xbeef";
        let obfuscated = obf.obfuscate(original);
        assert_eq!(obfuscated, "Transfer from ADDR_A to ADDR_B");
        assert_eq!(obf.deobfuscate(&obfuscated), original);
    }

    #[test]
    fn test_map_obfuscator_longest_first() {
        let mut map = HashMap::new();
        map.insert("0xdead".to_string(), "SHORT".to_string());
        map.insert("0xdeadbeef".to_string(), "LONG".to_string());
        let obf = MapObfuscator::new(map);

        // "0xdeadbeef" should match before "0xdead"
        assert_eq!(obf.obfuscate("address: 0xdeadbeef"), "address: LONG");
    }

    #[test]
    fn test_map_obfuscator_no_match() {
        let mut map = HashMap::new();
        map.insert("secret".to_string(), "REDACTED".to_string());
        let obf = MapObfuscator::new(map);

        assert_eq!(obf.obfuscate("nothing here"), "nothing here");
        assert_eq!(obf.deobfuscate("nothing here"), "nothing here");
    }

    #[test]
    fn test_map_obfuscator_empty_map() {
        let obf = MapObfuscator::new(HashMap::new());
        assert_eq!(obf.obfuscate("hello"), "hello");
        assert_eq!(obf.deobfuscate("hello"), "hello");
    }

    #[test]
    fn test_map_obfuscator_no_double_substitution() {
        let mut map = HashMap::new();
        map.insert("A".to_string(), "B".to_string());
        map.insert("B".to_string(), "C".to_string());
        let obf = MapObfuscator::new(map);

        // Single-pass: "A" matches "A"->"B", cursor advances past it.
        // The output "B" is never re-scanned, so "B"->"C" does NOT apply.
        assert_eq!(obf.obfuscate("A"), "B");

        // Each input token is consumed independently
        assert_eq!(obf.obfuscate("B"), "C");
        assert_eq!(obf.obfuscate("AB"), "BC");
    }

    #[test]
    fn test_map_obfuscator_mixed_length_single_pass() {
        let mut map = HashMap::new();
        map.insert("/xxx/A".to_string(), "/yyy".to_string());
        map.insert("A".to_string(), "B".to_string());
        let obf = MapObfuscator::new(map);

        // Longer "/xxx/A" wins at that position; standalone "A" replaced separately
        assert_eq!(obf.obfuscate("found A at /xxx/A"), "found B at /yyy");
    }

    #[test]
    fn test_map_obfuscator_chained_no_cascade() {
        let mut map = HashMap::new();
        map.insert("foo".to_string(), "bar".to_string());
        map.insert("bar".to_string(), "baz".to_string());
        map.insert("baz".to_string(), "qux".to_string());
        let obf = MapObfuscator::new(map);

        // Each input token consumed exactly once
        assert_eq!(obf.obfuscate("foo"), "bar");
        assert_eq!(obf.obfuscate("bar"), "baz");
        assert_eq!(obf.obfuscate("baz"), "qux");
        assert_eq!(obf.obfuscate("foo bar baz"), "bar baz qux");
    }

    #[test]
    fn test_map_obfuscator_overlapping_at_same_position() {
        let mut map = HashMap::new();
        map.insert("ab".to_string(), "X".to_string());
        map.insert("abc".to_string(), "Y".to_string());
        let obf = MapObfuscator::new(map);

        // "abc" is longer, wins at position 0
        assert_eq!(obf.obfuscate("abcd"), "Yd");
        // "ab" matches where "abc" doesn't
        assert_eq!(obf.obfuscate("abd"), "Xd");
    }

    #[test]
    fn test_map_obfuscator_utf8_safe() {
        let mut map = HashMap::new();
        map.insert("caf\u{00e9}".to_string(), "COFFEE".to_string());
        map.insert("\u{1f600}".to_string(), "SMILE".to_string());
        let obf = MapObfuscator::new(map);

        assert_eq!(obf.obfuscate("I like caf\u{00e9}"), "I like COFFEE");
        assert_eq!(obf.obfuscate("Hello \u{1f600} world"), "Hello SMILE world");
    }

    // -- FnObfuscator tests --

    #[test]
    fn test_fn_obfuscator_basic() {
        let obf = FnObfuscator::new(
            |text| text.replace("secret", "***"),
            |text| text.replace("***", "secret"),
        );

        assert_eq!(obf.obfuscate("my secret"), "my ***");
        assert_eq!(obf.deobfuscate("my ***"), "my secret");
    }

    #[test]
    fn test_fn_obfuscator_roundtrip() {
        let obf = FnObfuscator::new(|text| text.to_uppercase(), |text| text.to_lowercase());

        assert_eq!(obf.obfuscate("hello"), "HELLO");
        assert_eq!(obf.deobfuscate("HELLO"), "hello");
    }

    // -- ObfuscatingCompletionModel tests --

    /// Mock model that captures the request it receives
    #[derive(Clone)]
    struct CapturingMockModel {
        captured: std::sync::Arc<tokio::sync::Mutex<Option<CompletionRequest>>>,
    }

    impl CapturingMockModel {
        fn new() -> Self {
            Self {
                captured: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            }
        }

        async fn captured_request(&self) -> CompletionRequest {
            self.captured.lock().await.take().unwrap()
        }
    }

    impl CompletionModel for CapturingMockModel {
        type Response = serde_json::Value;

        async fn completion(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            // Capture the request the model received (after obfuscation)
            *self.captured.lock().await = Some(request.clone());

            // Echo back the last message text (which should be obfuscated)
            let last_text = request
                .messages
                .last()
                .map(|m| m.text())
                .unwrap_or_default();

            Ok(CompletionResponse::new(
                Message::assistant(format!("Echo: {}", last_text)),
                Usage::new(10, 20),
                serde_json::json!({}),
            ))
        }

        fn model_id(&self) -> &str {
            "capturing-mock"
        }

        fn provider(&self) -> &str {
            "mock"
        }
    }

    fn test_obfuscator() -> MapObfuscator {
        let mut map = HashMap::new();
        map.insert("secret_addr".to_string(), "ADDR_X".to_string());
        map.insert("secret_key".to_string(), "KEY_Y".to_string());
        MapObfuscator::new(map)
    }

    #[tokio::test]
    async fn test_obfuscate_request_preamble() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest::new("hello")
            .with_preamble("Analyze secret_addr for vulnerabilities");

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        assert_eq!(
            captured.preamble.unwrap(),
            "Analyze ADDR_X for vulnerabilities"
        );
    }

    #[tokio::test]
    async fn test_obfuscate_request_user_text() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest::new("Check secret_addr and secret_key");

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        assert_eq!(
            captured.messages.last().unwrap().text(),
            "Check ADDR_X and KEY_Y"
        );
    }

    #[tokio::test]
    async fn test_obfuscate_request_tool_result() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message::tool_result("call-1", "Result for secret_addr: OK")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            cache: Default::default(),
        };

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        if let Message::User { content, .. } = &captured.messages[0] {
            if let UserContent::ToolResult { content, .. } = &content[0] {
                assert_eq!(content, "Result for ADDR_X: OK");
            } else {
                panic!("Expected ToolResult");
            }
        } else {
            panic!("Expected User message");
        }
    }

    #[tokio::test]
    async fn test_obfuscate_request_assistant_history() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest {
            preamble: None,
            messages: vec![
                Message::assistant("I found secret_addr in the code"),
                Message::user("Tell me more about secret_key"),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            cache: Default::default(),
        };

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        assert_eq!(captured.messages[0].text(), "I found ADDR_X in the code");
        assert_eq!(captured.messages[1].text(), "Tell me more about KEY_Y");
    }

    #[tokio::test]
    async fn test_obfuscate_request_tool_call_args() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message::Assistant {
                content: vec![AssistantContent::ToolCall(ToolCall::new(
                    "call-1",
                    "analyze",
                    r#"{"address": "secret_addr"}"#,
                ))],
                id: None,
                reasoning: None,
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            cache: Default::default(),
        };

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        let tool_calls = captured.messages[0].tool_calls();
        assert_eq!(tool_calls[0].function.arguments, r#"{"address": "ADDR_X"}"#);
    }

    #[tokio::test]
    async fn test_obfuscate_preserves_tool_names() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request = CompletionRequest {
            preamble: None,
            messages: vec![Message::Assistant {
                content: vec![AssistantContent::ToolCall(ToolCall::new(
                    "call-1",
                    "analyze_contract",
                    "{}",
                ))],
                id: None,
                reasoning: None,
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            cache: Default::default(),
        };

        let _ = model.completion(request).await.unwrap();

        let captured = mock.captured_request().await;
        let tool_calls = captured.messages[0].tool_calls();
        assert_eq!(tool_calls[0].function.name, "analyze_contract");
    }

    #[tokio::test]
    async fn test_deobfuscate_response_text() {
        // Mock that returns obfuscated tokens in response
        #[derive(Clone)]
        struct ObfuscatedResponseModel;

        impl CompletionModel for ObfuscatedResponseModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                Ok(CompletionResponse::new(
                    Message::assistant("Found vulnerability at ADDR_X using KEY_Y"),
                    Usage::new(10, 20),
                    serde_json::json!({}),
                ))
            }

            fn model_id(&self) -> &str {
                "mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = ObfuscatingCompletionModel::new(ObfuscatedResponseModel, test_obfuscator());
        let response = model
            .completion(CompletionRequest::new("test"))
            .await
            .unwrap();

        assert_eq!(
            response.content(),
            "Found vulnerability at secret_addr using secret_key"
        );
    }

    #[tokio::test]
    async fn test_deobfuscate_response_tool_call_args() {
        #[derive(Clone)]
        struct ToolCallResponseModel;

        impl CompletionModel for ToolCallResponseModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let msg = Message::Assistant {
                    content: vec![AssistantContent::ToolCall(ToolCall::new(
                        "call-1",
                        "write_file",
                        r#"{"path": "ADDR_X.txt"}"#,
                    ))],
                    id: None,
                    reasoning: None,
                };
                Ok(CompletionResponse::new(
                    msg,
                    Usage::new(10, 20),
                    serde_json::json!({}),
                ))
            }

            fn model_id(&self) -> &str {
                "mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = ObfuscatingCompletionModel::new(ToolCallResponseModel, test_obfuscator());
        let response = model
            .completion(CompletionRequest::new("test"))
            .await
            .unwrap();

        let tool_calls = response.tool_calls();
        assert_eq!(
            tool_calls[0].function.arguments,
            r#"{"path": "secret_addr.txt"}"#
        );
        // Tool name must NOT be de-obfuscated
        assert_eq!(tool_calls[0].function.name, "write_file");
    }

    #[tokio::test]
    async fn test_deobfuscate_response_reasoning() {
        #[derive(Clone)]
        struct ReasoningResponseModel;

        impl CompletionModel for ReasoningResponseModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                Ok(CompletionResponse {
                    message: Message::assistant("result"),
                    usage: Usage::new(10, 20),
                    raw: serde_json::json!({}),
                    reasoning_content: Some("Thinking about ADDR_X...".to_string()),
                    finish_reason: Some("stop".to_string()),
                })
            }

            fn model_id(&self) -> &str {
                "mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = ObfuscatingCompletionModel::new(ReasoningResponseModel, test_obfuscator());
        let response = model
            .completion(CompletionRequest::new("test"))
            .await
            .unwrap();

        assert_eq!(
            response.reasoning_content.unwrap(),
            "Thinking about secret_addr..."
        );
    }

    #[tokio::test]
    async fn test_full_obfuscation_roundtrip() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock.clone(), test_obfuscator());

        let request =
            CompletionRequest::new("Analyze secret_addr").with_preamble("You analyze secret_key");

        let response = model.completion(request).await.unwrap();

        // The mock echoes back, so the model received obfuscated text
        let captured = mock.captured_request().await;
        assert_eq!(captured.preamble.unwrap(), "You analyze KEY_Y");
        assert_eq!(captured.messages[0].text(), "Analyze ADDR_X");

        // The response should be de-obfuscated
        // Mock echoes "Echo: Analyze ADDR_X" -> de-obfuscated to "Echo: Analyze secret_addr"
        assert_eq!(response.content(), "Echo: Analyze secret_addr");
    }

    #[tokio::test]
    async fn test_model_id_and_provider_delegate() {
        let mock = CapturingMockModel::new();
        let model = ObfuscatingCompletionModel::new(mock, test_obfuscator());

        assert_eq!(model.model_id(), "capturing-mock");
        assert_eq!(model.provider(), "mock");
    }

    #[test]
    fn test_obfuscate_reasoning_in_history() {
        let obf = test_obfuscator();
        let model = ObfuscatingCompletionModel::new(
            // We only need the model for type checking, won't call completion
            CapturingMockModel::new(),
            obf,
        );

        let msg = Message::Assistant {
            content: vec![AssistantContent::Reasoning {
                reasoning: vec![
                    "Step 1: check secret_addr".to_string(),
                    "Step 2: use secret_key".to_string(),
                ],
            }],
            id: None,
            reasoning: Some("Overall: secret_addr analysis".to_string()),
        };

        let obfuscated = model.obfuscate_message(msg);

        if let Message::Assistant {
            content, reasoning, ..
        } = &obfuscated
        {
            if let AssistantContent::Reasoning { reasoning: steps } = &content[0] {
                assert_eq!(steps[0], "Step 1: check ADDR_X");
                assert_eq!(steps[1], "Step 2: use KEY_Y");
            } else {
                panic!("Expected Reasoning content");
            }
            assert_eq!(reasoning.as_deref().unwrap(), "Overall: ADDR_X analysis");
        } else {
            panic!("Expected Assistant message");
        }
    }

    #[test]
    fn test_obfuscate_preserves_image_document() {
        let obf = test_obfuscator();
        let model = ObfuscatingCompletionModel::new(CapturingMockModel::new(), obf);

        let msg = Message::User {
            content: vec![
                UserContent::Text {
                    text: "secret_addr".to_string(),
                },
                UserContent::Image {
                    data: "secret_addr_base64".to_string(),
                    media_type: "image/png".to_string(),
                },
                UserContent::Document {
                    data: "secret_key_doc".to_string(),
                    media_type: "application/pdf".to_string(),
                    name: Some("report".to_string()),
                },
            ],
            id: None,
        };

        let obfuscated = model.obfuscate_message(msg);

        if let Message::User { content, .. } = &obfuscated {
            // Text is obfuscated
            if let UserContent::Text { text } = &content[0] {
                assert_eq!(text, "ADDR_X");
            }
            // Image data is NOT obfuscated
            if let UserContent::Image { data, .. } = &content[1] {
                assert_eq!(data, "secret_addr_base64");
            }
            // Document data is NOT obfuscated
            if let UserContent::Document { data, .. } = &content[2] {
                assert_eq!(data, "secret_key_doc");
            }
        } else {
            panic!("Expected User message");
        }
    }
}
