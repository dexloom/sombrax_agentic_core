//! Message types for conversation content
//!
//! Provides the Message enum and content types for user and assistant messages,
//! along with validation helpers to enforce message integrity.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

/// Error returned when message validation fails
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Message content is empty
    EmptyContent {
        /// Role of the message with empty content
        role: &'static str,
    },
    /// Text content is empty or whitespace-only
    EmptyText {
        /// Role of the message containing empty text
        role: &'static str,
    },
    /// Tool call references an unknown tool
    UnknownTool {
        /// The tool call ID
        tool_call_id: String,
        /// The unknown tool name
        tool_name: String,
    },
    /// Tool result ID doesn't match any previous tool call
    UnmatchedToolResult {
        /// The tool result ID that doesn't match
        tool_result_id: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyContent { role } => {
                write!(f, "{} message has empty content", role)
            }
            ValidationError::EmptyText { role } => {
                write!(f, "{} message contains empty or whitespace-only text", role)
            }
            ValidationError::UnknownTool {
                tool_call_id,
                tool_name,
            } => {
                write!(
                    f,
                    "tool call '{}' references unknown tool '{}'",
                    tool_call_id, tool_name
                )
            }
            ValidationError::UnmatchedToolResult { tool_result_id } => {
                write!(
                    f,
                    "tool result '{}' does not match any previous tool call",
                    tool_result_id
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A user message
    User {
        /// Message content items
        content: Vec<UserContent>,
        /// Optional message ID
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// An assistant message
    Assistant {
        /// Message content items
        content: Vec<AssistantContent>,
        /// Optional message ID
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Optional reasoning/thinking content (for models that return it as separate field)
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
}

impl Message {
    /// Create a new user message with text content
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::Text { text: text.into() }],
            id: None,
        }
    }

    /// Create a new user message with text content and an ID
    pub fn user_with_id(text: impl Into<String>, id: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::Text { text: text.into() }],
            id: Some(id.into()),
        }
    }

    /// Create a new assistant message with text content
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![AssistantContent::Text { text: text.into() }],
            id: None,
            reasoning: None,
        }
    }

    /// Create a new assistant message with text content and an ID
    pub fn assistant_with_id(text: impl Into<String>, id: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![AssistantContent::Text { text: text.into() }],
            id: Some(id.into()),
            reasoning: None,
        }
    }

    /// Create a new assistant message with text content and reasoning
    pub fn assistant_with_reasoning(text: impl Into<String>, reasoning: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![AssistantContent::Text { text: text.into() }],
            id: None,
            reasoning: Some(reasoning.into()),
        }
    }

    /// Create a tool result message
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::ToolResult {
                id: tool_call_id.into(),
                content: content.into(),
            }],
            id: None,
        }
    }

    /// Get the message ID if present
    pub fn id(&self) -> Option<&str> {
        match self {
            Message::User { id, .. } => id.as_deref(),
            Message::Assistant { id, .. } => id.as_deref(),
        }
    }

    /// Set the message ID
    pub fn with_id(mut self, new_id: impl Into<String>) -> Self {
        match &mut self {
            Message::User { id, .. } => *id = Some(new_id.into()),
            Message::Assistant { id, .. } => *id = Some(new_id.into()),
        }
        self
    }

    /// Check if this is a user message
    pub fn is_user(&self) -> bool {
        matches!(self, Message::User { .. })
    }

    /// Check if this is an assistant message
    pub fn is_assistant(&self) -> bool {
        matches!(self, Message::Assistant { .. })
    }

    /// Get the message role as a string
    pub fn role(&self) -> &'static str {
        match self {
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
        }
    }

    /// Get the text content of the message
    pub fn text(&self) -> String {
        match self {
            Message::User { content, .. } => content
                .iter()
                .filter_map(|c| match c {
                    UserContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Message::Assistant { content, .. } => content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Get the approximate content length (for token estimation)
    pub fn content_length(&self) -> usize {
        self.text().len()
    }

    /// Prepend text to the message content
    pub fn prepend_text(&mut self, prefix: &str) {
        match self {
            Message::User { content, .. } => {
                if let Some(UserContent::Text { text }) = content.first_mut() {
                    *text = format!("{}{}", prefix, text);
                } else {
                    content.insert(
                        0,
                        UserContent::Text {
                            text: prefix.to_string(),
                        },
                    );
                }
            }
            Message::Assistant { content, .. } => {
                if let Some(AssistantContent::Text { text }) = content.first_mut() {
                    *text = format!("{}{}", prefix, text);
                } else {
                    content.insert(
                        0,
                        AssistantContent::Text {
                            text: prefix.to_string(),
                        },
                    );
                }
            }
        }
    }

    /// Append text to the message content
    pub fn append_text(&mut self, suffix: &str) {
        match self {
            Message::User { content, .. } => {
                if let Some(UserContent::Text { text }) = content.last_mut() {
                    text.push_str(suffix);
                } else {
                    content.push(UserContent::Text {
                        text: suffix.to_string(),
                    });
                }
            }
            Message::Assistant { content, .. } => {
                if let Some(AssistantContent::Text { text }) = content.last_mut() {
                    text.push_str(suffix);
                } else {
                    content.push(AssistantContent::Text {
                        text: suffix.to_string(),
                    });
                }
            }
        }
    }

    /// Create a copy of this message with reasoning content set
    ///
    /// For assistant messages, this sets the reasoning field to preserve
    /// the model's thought process in conversation history for continuity
    /// after tool calls.
    pub fn with_reasoning(&self, reasoning_content: &str) -> Self {
        match self {
            Message::Assistant { content, id, .. } => Message::Assistant {
                content: content.clone(),
                id: id.clone(),
                reasoning: if reasoning_content.is_empty() {
                    None
                } else {
                    Some(reasoning_content.to_string())
                },
            },
            // For non-assistant messages, just return a clone
            _ => self.clone(),
        }
    }

    /// Get the reasoning content from an assistant message
    pub fn reasoning(&self) -> Option<&str> {
        match self {
            Message::Assistant { reasoning, .. } => reasoning.as_deref(),
            _ => None,
        }
    }

    /// Set reasoning on an assistant message (mutates in place)
    pub fn set_reasoning(&mut self, reasoning_content: Option<String>) {
        if let Message::Assistant { reasoning, .. } = self {
            *reasoning = reasoning_content;
        }
    }

    /// Get tool calls from an assistant message
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        match self {
            Message::Assistant { content, .. } => content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::ToolCall(tc) => Some(tc),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Check if this message contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls().is_empty()
    }

    /// Remap all tool call IDs in this message using the provided generator function.
    ///
    /// The generator receives the old ID and returns the new ID.
    /// This is useful for normalizing provider-specific tool call IDs to a
    /// simpler incremental format (e.g., `tool_call_1`, `tool_call_2`).
    ///
    /// Only affects assistant messages containing tool calls.
    pub fn remap_tool_call_ids<F>(&mut self, mut id_generator: F)
    where
        F: FnMut(&str) -> String,
    {
        if let Message::Assistant { content, .. } = self {
            for item in content.iter_mut() {
                if let AssistantContent::ToolCall(tc) = item {
                    tc.id = id_generator(&tc.id);
                }
            }
        }
    }

    /// Get tool result IDs from a user message
    pub fn tool_result_ids(&self) -> Vec<&str> {
        match self {
            Message::User { content, .. } => content
                .iter()
                .filter_map(|c| match c {
                    UserContent::ToolResult { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Validate the message content
    ///
    /// Checks that:
    /// - The message has at least one content item
    /// - Text content is non-empty (not just whitespace)
    ///
    /// Returns `Ok(())` if valid, or a list of validation errors.
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        match self {
            Message::User { content, .. } => {
                if content.is_empty() {
                    errors.push(ValidationError::EmptyContent { role: "user" });
                }
                for item in content {
                    if let UserContent::Text { text } = item {
                        if text.trim().is_empty() {
                            errors.push(ValidationError::EmptyText { role: "user" });
                            break;
                        }
                    }
                }
            }
            Message::Assistant { content, .. } => {
                if content.is_empty() {
                    errors.push(ValidationError::EmptyContent { role: "assistant" });
                }
                for item in content {
                    if let AssistantContent::Text { text } = item {
                        if text.trim().is_empty() {
                            errors.push(ValidationError::EmptyText { role: "assistant" });
                            break;
                        }
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate tool calls against a set of registered tool names
    ///
    /// Returns `Ok(())` if all tool calls reference registered tools,
    /// or a list of validation errors for unknown tools.
    pub fn validate_tool_calls(
        &self,
        registered_tools: &HashSet<&str>,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        for tc in self.tool_calls() {
            if !registered_tools.contains(tc.function.name.as_str()) {
                errors.push(ValidationError::UnknownTool {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.function.name.clone(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Validate that tool result IDs in a message match previous tool call IDs
///
/// This function checks the conversation history to ensure that any
/// `ToolResult` in the given message has an ID that matches a previous
/// `ToolCall` in the history.
///
/// # Arguments
/// * `message` - The message to validate
/// * `history` - Previous messages in the conversation
///
/// # Returns
/// `Ok(())` if all tool result IDs match, or a list of unmatched IDs.
pub fn validate_tool_result_ids(
    message: &Message,
    history: &[Message],
) -> Result<(), Vec<ValidationError>> {
    // Collect all tool call IDs from history
    let tool_call_ids: HashSet<&str> = history
        .iter()
        .flat_map(|msg| msg.tool_calls())
        .map(|tc| tc.id.as_str())
        .collect();

    let mut errors = Vec::new();

    for result_id in message.tool_result_ids() {
        if !tool_call_ids.contains(result_id) {
            errors.push(ValidationError::UnmatchedToolResult {
                tool_result_id: result_id.to_string(),
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Message::user(s)
    }
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Message::user(s)
    }
}

/// Content types for user messages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Tool result from a previous tool call
    ToolResult {
        /// The ID of the tool call this is a result for
        id: String,
        /// The tool result content
        content: String,
    },
    /// Image content
    Image {
        /// Base64-encoded image data
        data: String,
        /// MIME type of the image
        media_type: String,
    },
    /// Document content
    Document {
        /// Base64-encoded document data
        data: String,
        /// MIME type of the document
        media_type: String,
        /// Optional document name
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

/// Content types for assistant messages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Tool call request
    ToolCall(ToolCall),
    /// Reasoning/thinking content (for models that support it)
    Reasoning {
        /// The reasoning steps
        reasoning: Vec<String>,
    },
}

/// A tool call request from the assistant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// The function to call
    pub function: ToolCallFunction,
}

/// Function details for a tool call
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    /// Name of the function to call
    pub name: String,
    /// JSON-encoded arguments
    pub arguments: String,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    /// Parse the arguments as a specific type
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.function.arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message() {
        let msg = Message::user("Hello, world!");
        assert!(msg.is_user());
        assert!(!msg.is_assistant());
        assert_eq!(msg.text(), "Hello, world!");
    }

    #[test]
    fn test_assistant_message() {
        let msg = Message::assistant("Hello back!");
        assert!(msg.is_assistant());
        assert!(!msg.is_user());
        assert_eq!(msg.text(), "Hello back!");
    }

    #[test]
    fn test_message_with_id() {
        let msg = Message::user_with_id("Test", "msg-123");
        assert_eq!(msg.id(), Some("msg-123"));
    }

    #[test]
    fn test_prepend_text() {
        let mut msg = Message::user("world");
        msg.prepend_text("Hello, ");
        assert_eq!(msg.text(), "Hello, world");
    }

    #[test]
    fn test_append_text() {
        let mut msg = Message::user("Hello");
        msg.append_text(", world!");
        assert_eq!(msg.text(), "Hello, world!");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Test message");
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, parsed);
    }

    #[test]
    fn test_tool_call() {
        let tc = ToolCall::new("call-1", "get_weather", r#"{"city": "NYC"}"#);
        assert_eq!(tc.id, "call-1");
        assert_eq!(tc.function.name, "get_weather");

        #[derive(Deserialize)]
        struct Args {
            city: String,
        }
        let args: Args = tc.parse_arguments().unwrap();
        assert_eq!(args.city, "NYC");
    }

    #[test]
    fn test_validate_valid_user_message() {
        let msg = Message::user("Hello, world!");
        assert!(msg.validate().is_ok());
    }

    #[test]
    fn test_validate_valid_assistant_message() {
        let msg = Message::assistant("Hello back!");
        assert!(msg.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_user_content() {
        let msg = Message::User {
            content: vec![],
            id: None,
        };
        let result = msg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            ValidationError::EmptyContent { role: "user" }
        ));
    }

    #[test]
    fn test_validate_empty_assistant_content() {
        let msg = Message::Assistant {
            content: vec![],
            id: None,
            reasoning: None,
        };
        let result = msg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(matches!(
            &errors[0],
            ValidationError::EmptyContent { role: "assistant" }
        ));
    }

    #[test]
    fn test_validate_whitespace_only_text() {
        let msg = Message::User {
            content: vec![UserContent::Text {
                text: "   \t\n  ".to_string(),
            }],
            id: None,
        };
        let result = msg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(matches!(
            &errors[0],
            ValidationError::EmptyText { role: "user" }
        ));
    }

    #[test]
    fn test_validate_tool_calls_known_tool() {
        let msg = Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-1",
                "get_weather",
                "{}",
            ))],
            id: None,
            reasoning: None,
        };
        let tools: HashSet<&str> = ["get_weather", "search"].into_iter().collect();
        assert!(msg.validate_tool_calls(&tools).is_ok());
    }

    #[test]
    fn test_validate_tool_calls_unknown_tool() {
        let msg = Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-1",
                "unknown_tool",
                "{}",
            ))],
            id: None,
            reasoning: None,
        };
        let tools: HashSet<&str> = ["get_weather", "search"].into_iter().collect();
        let result = msg.validate_tool_calls(&tools);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(matches!(
            &errors[0],
            ValidationError::UnknownTool { tool_name, .. } if tool_name == "unknown_tool"
        ));
    }

    #[test]
    fn test_validate_tool_result_ids_matching() {
        // History with a tool call
        let history = vec![Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-123",
                "get_weather",
                "{}",
            ))],
            id: None,
            reasoning: None,
        }];

        // Tool result matching the call
        let msg = Message::tool_result("call-123", "Sunny");
        assert!(validate_tool_result_ids(&msg, &history).is_ok());
    }

    #[test]
    fn test_validate_tool_result_ids_unmatched() {
        // Empty history
        let history = vec![];

        // Tool result with no matching call
        let msg = Message::tool_result("call-unknown", "Result");
        let result = validate_tool_result_ids(&msg, &history);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(matches!(
            &errors[0],
            ValidationError::UnmatchedToolResult { tool_result_id } if tool_result_id == "call-unknown"
        ));
    }

    #[test]
    fn test_tool_result_ids_extraction() {
        let msg = Message::tool_result("call-123", "Result");
        let ids = msg.tool_result_ids();
        assert_eq!(ids, vec!["call-123"]);
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::EmptyContent { role: "user" };
        assert_eq!(err.to_string(), "user message has empty content");

        let err = ValidationError::EmptyText { role: "assistant" };
        assert_eq!(
            err.to_string(),
            "assistant message contains empty or whitespace-only text"
        );

        let err = ValidationError::UnknownTool {
            tool_call_id: "call-1".to_string(),
            tool_name: "foo".to_string(),
        };
        assert!(err.to_string().contains("call-1"));
        assert!(err.to_string().contains("foo"));

        let err = ValidationError::UnmatchedToolResult {
            tool_result_id: "result-1".to_string(),
        };
        assert!(err.to_string().contains("result-1"));
    }

    #[test]
    fn test_remap_tool_call_ids() {
        // Create assistant message with multiple tool calls
        let mut msg = Message::Assistant {
            content: vec![
                AssistantContent::Text {
                    text: "Let me help".to_string(),
                },
                AssistantContent::ToolCall(ToolCall::new(
                    "call_abc123xyz",
                    "get_weather",
                    r#"{"city": "NYC"}"#,
                )),
                AssistantContent::ToolCall(ToolCall::new(
                    "call_def456uvw",
                    "get_time",
                    r#"{"timezone": "EST"}"#,
                )),
            ],
            id: None,
            reasoning: None,
        };

        // Remap using an incrementing counter
        let mut counter = 0;
        msg.remap_tool_call_ids(|_old_id| {
            counter += 1;
            format!("tool_call_{}", counter)
        });

        // Verify tool calls have new IDs
        let tool_calls = msg.tool_calls();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, "tool_call_1");
        assert_eq!(tool_calls[1].id, "tool_call_2");

        // Verify other content is unchanged
        assert_eq!(msg.text(), "Let me help");
    }

    #[test]
    fn test_remap_tool_call_ids_user_message_noop() {
        // User messages should not be affected
        let mut msg = Message::user("Hello");
        msg.remap_tool_call_ids(|_| "should_not_be_called".to_string());

        // Should still be the same user message
        assert!(msg.is_user());
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn test_remap_tool_call_ids_no_tool_calls() {
        // Assistant message without tool calls
        let mut msg = Message::assistant("Just text");
        msg.remap_tool_call_ids(|_| "should_not_be_called".to_string());

        assert_eq!(msg.text(), "Just text");
        assert!(msg.tool_calls().is_empty());
    }
}
