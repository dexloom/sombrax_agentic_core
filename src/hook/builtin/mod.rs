//! Built-in hook implementations
//!
//! Provides commonly used hooks out of the box.

use crate::context::HookContext;
use crate::error::{HookError, HookResult, HookStage};
use crate::hook::{Hook, ToolCallDecision};
use crate::message::{validate_tool_result_ids, Message};
#[allow(unused_imports)]
use crate::provider::CompletionResponse;
use std::collections::HashSet;
use std::path::PathBuf;

/// A simple logging hook that logs messages and responses
#[derive(Clone, Debug, Default)]
pub struct LoggingHook {
    /// Log level for messages
    log_level: LogLevel,
}

/// Log level for the logging hook
#[derive(Clone, Debug, Default)]
pub enum LogLevel {
    /// Debug level
    Debug,
    /// Info level (default)
    #[default]
    Info,
    /// Warn level
    Warn,
}

impl LoggingHook {
    /// Create a new logging hook
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the log level
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.log_level = level;
        self
    }
}

impl Hook for LoggingHook {
    async fn pre_completion(
        &self,
        message: Message,
        _history: &[Message],
        ctx: &mut HookContext,
    ) -> HookResult<Message> {
        match self.log_level {
            LogLevel::Debug => {
                tracing::debug!(
                    request_id = %ctx.request_id,
                    message = ?message.text(),
                    "pre_completion: processing message"
                );
            }
            LogLevel::Info => {
                tracing::info!(
                    request_id = %ctx.request_id,
                    message_len = %message.content_length(),
                    "pre_completion: processing message"
                );
            }
            LogLevel::Warn => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    "pre_completion: processing message"
                );
            }
        }
        Ok(message)
    }

    async fn post_completion_message(
        &self,
        message: Message,
        ctx: &mut HookContext,
    ) -> HookResult<Message> {
        match self.log_level {
            LogLevel::Debug => {
                tracing::debug!(
                    request_id = %ctx.request_id,
                    elapsed_ms = %ctx.elapsed().as_millis(),
                    response = ?message.text(),
                    "post_completion: received response"
                );
            }
            LogLevel::Info => {
                tracing::info!(
                    request_id = %ctx.request_id,
                    elapsed_ms = %ctx.elapsed().as_millis(),
                    response_len = %message.content_length(),
                    "post_completion: received response"
                );
            }
            LogLevel::Warn => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    elapsed_ms = %ctx.elapsed().as_millis(),
                    "post_completion: received response"
                );
            }
        }
        Ok(message)
    }

    async fn on_assistant_message(
        &self,
        message: &Message,
        ctx: &mut HookContext,
    ) -> HookResult<()> {
        let text = message.text();
        let has_tool_calls = message.has_tool_calls();

        match self.log_level {
            LogLevel::Debug => {
                tracing::debug!(
                    request_id = %ctx.request_id,
                    text = %text,
                    has_tool_calls = %has_tool_calls,
                    "assistant message"
                );
            }
            LogLevel::Info => {
                if !text.is_empty() {
                    tracing::info!(
                        request_id = %ctx.request_id,
                        text_len = %text.len(),
                        has_tool_calls = %has_tool_calls,
                        "assistant message"
                    );
                }
            }
            LogLevel::Warn => {
                if !text.is_empty() {
                    tracing::warn!(
                        request_id = %ctx.request_id,
                        has_tool_calls = %has_tool_calls,
                        "assistant message"
                    );
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "LoggingHook"
    }
}

/// A hook that adds a prefix to all user messages
#[derive(Clone, Debug)]
pub struct PrefixHook {
    prefix: String,
}

impl PrefixHook {
    /// Create a new prefix hook
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl Hook for PrefixHook {
    async fn pre_completion(
        &self,
        mut message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        message.prepend_text(&self.prefix);
        Ok(message)
    }

    fn name(&self) -> &str {
        "PrefixHook"
    }
}

/// A hook that adds a suffix to all user messages
#[derive(Clone, Debug)]
pub struct SuffixHook {
    suffix: String,
}

impl SuffixHook {
    /// Create a new suffix hook
    pub fn new(suffix: impl Into<String>) -> Self {
        Self {
            suffix: suffix.into(),
        }
    }
}

impl Hook for SuffixHook {
    async fn pre_completion(
        &self,
        mut message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        message.append_text(&self.suffix);
        Ok(message)
    }

    fn name(&self) -> &str {
        "SuffixHook"
    }
}

/// A hook that validates messages according to message validation rules
///
/// Validation rules enforced:
/// - User messages must have non-empty content
/// - Assistant messages must have non-empty content
/// - Text content must not be empty or whitespace-only
/// - Tool result IDs must match previous tool call IDs in history
/// - Tool calls must reference registered tools (when tool names are configured)
///
/// # Example
///
/// ```ignore
/// use rig_agent::hook::builtin::ValidationHook;
///
/// let hook = ValidationHook::new();
///
/// // Or with registered tool names for tool call validation
/// let hook = ValidationHook::with_tools(vec!["get_weather", "search"]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ValidationHook {
    /// Registered tool names for tool call validation
    registered_tools: HashSet<String>,
}

impl ValidationHook {
    /// Create a new validation hook
    ///
    /// This will validate:
    /// - Non-empty message content
    /// - Non-empty text
    /// - Tool result ID matching against history
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a validation hook with registered tool names
    ///
    /// This additionally validates that tool calls reference known tools.
    pub fn with_tools<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            registered_tools: tools.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Add a tool name to the set of registered tools
    pub fn add_tool(&mut self, tool_name: impl Into<String>) {
        self.registered_tools.insert(tool_name.into());
    }
}

impl Hook for ValidationHook {
    async fn pre_completion(
        &self,
        message: Message,
        history: &[Message],
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        // Validate message content (non-empty, non-whitespace text)
        if let Err(errors) = message.validate() {
            let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(HookError::hook_failed(
                "ValidationHook",
                HookStage::PreCompletion,
                error_msgs.join("; "),
            ));
        }

        // Validate tool result IDs match previous tool calls
        if let Err(errors) = validate_tool_result_ids(&message, history) {
            let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(HookError::hook_failed(
                "ValidationHook",
                HookStage::PreCompletion,
                error_msgs.join("; "),
            ));
        }

        Ok(message)
    }

    async fn post_completion_message(
        &self,
        message: Message,
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        // Validate assistant response content
        if let Err(errors) = message.validate() {
            let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(HookError::hook_failed(
                "ValidationHook",
                HookStage::PostCompletion,
                error_msgs.join("; "),
            ));
        }

        // Validate tool calls reference registered tools (if configured)
        if !self.registered_tools.is_empty() {
            let tools_ref: HashSet<&str> =
                self.registered_tools.iter().map(|s| s.as_str()).collect();
            if let Err(errors) = message.validate_tool_calls(&tools_ref) {
                let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                return Err(HookError::hook_failed(
                    "ValidationHook",
                    HookStage::PostCompletion,
                    error_msgs.join("; "),
                ));
            }
        }

        Ok(message)
    }

    fn name(&self) -> &str {
        "ValidationHook"
    }
}

/// A hook that enforces workspace boundary for file operations.
///
/// This hook intercepts file-related tool calls (read, write, edit, grep, glob)
/// and ensures that all file paths are within the configured workspace directory.
/// This prevents agents from accessing files outside the allowed directory tree.
///
/// # Security
///
/// This hook provides a defense-in-depth measure. The tools themselves also
/// validate paths, but this hook adds an additional layer at the agent level.
///
/// # Example
///
/// ```ignore
/// use sombrax_agentic_core::hook::builtin::WorkspaceBoundaryHook;
/// use std::path::PathBuf;
///
/// // Restrict to current working directory
/// let hook = WorkspaceBoundaryHook::new(PathBuf::from("."));
///
/// // Or allow specific additional directories
/// let hook = WorkspaceBoundaryHook::new(PathBuf::from("."))
///     .allow_path("/usr/share/dict");
///
/// let agent = Agent::builder(model)
///     .hook(hook)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct WorkspaceBoundaryHook {
    /// The workspace root directory (canonicalized)
    workspace: PathBuf,
    /// Additional allowed paths outside workspace
    allowed_paths: Vec<PathBuf>,
    /// Tools to check (defaults to file operation tools)
    file_tools: HashSet<String>,
}

impl WorkspaceBoundaryHook {
    /// Create a new workspace boundary hook.
    ///
    /// The workspace path will be canonicalized to resolve symlinks and relative paths.
    pub fn new(workspace: PathBuf) -> Self {
        let canonical = workspace.canonicalize().unwrap_or(workspace);

        // Default file operation tools
        let file_tools: HashSet<String> = ["read", "write", "edit", "grep", "glob"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        Self {
            workspace: canonical,
            allowed_paths: Vec::new(),
            file_tools,
        }
    }

    /// Allow access to an additional path outside the workspace.
    ///
    /// The path will be canonicalized. If canonicalization fails,
    /// the original path is used.
    pub fn allow_path(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let canonical = path.canonicalize().unwrap_or(path);
        self.allowed_paths.push(canonical);
        self
    }

    /// Add a custom tool name to be checked for path validation.
    pub fn add_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.file_tools.insert(tool_name.into());
        self
    }

    /// Check if a path is within the allowed boundaries.
    fn is_path_allowed(&self, path: &str) -> bool {
        let path = PathBuf::from(path);

        // Make path absolute if relative
        let absolute_path = if path.is_absolute() {
            path
        } else {
            self.workspace.join(&path)
        };

        // Try to canonicalize (resolves symlinks and ..)
        let canonical = match absolute_path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // For new files, check if parent is allowed
                if let Some(parent) = absolute_path.parent() {
                    if let Ok(parent_canonical) = parent.canonicalize() {
                        // Check parent against workspace and allowed paths
                        if parent_canonical.starts_with(&self.workspace) {
                            return true;
                        }
                        for allowed in &self.allowed_paths {
                            if parent_canonical.starts_with(allowed) {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }
        };

        // Check if within workspace
        if canonical.starts_with(&self.workspace) {
            return true;
        }

        // Check if within any allowed path
        for allowed in &self.allowed_paths {
            if canonical.starts_with(allowed) {
                return true;
            }
        }

        false
    }

    /// Extract path from tool arguments based on tool name.
    fn extract_path(&self, tool_name: &str, args: &serde_json::Value) -> Option<String> {
        match tool_name {
            "read" | "write" | "edit" => args
                .get("file_path")
                .and_then(|v| v.as_str())
                .map(String::from),
            "grep" | "glob" => {
                // These tools have optional "path" parameter
                args.get("path").and_then(|v| v.as_str()).map(String::from)
            }
            _ => None,
        }
    }
}

impl Hook for WorkspaceBoundaryHook {
    async fn pre_tool_call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        _ctx: &mut HookContext,
    ) -> HookResult<ToolCallDecision> {
        // Only check file operation tools
        if !self.file_tools.contains(tool_name) {
            return Ok(ToolCallDecision::Proceed(args));
        }

        // Extract path from arguments
        if let Some(path) = self.extract_path(tool_name, &args) {
            if !self.is_path_allowed(&path) {
                tracing::warn!(
                    tool = %tool_name,
                    path = %path,
                    workspace = %self.workspace.display(),
                    "Blocked tool call: path outside workspace boundary"
                );
                return Ok(ToolCallDecision::Block(format!(
                    "Access denied: path '{}' is outside the allowed workspace. \
                     Only paths within '{}' are permitted.",
                    path,
                    self.workspace.display()
                )));
            }
        }

        Ok(ToolCallDecision::Proceed(args))
    }

    fn name(&self) -> &str {
        "WorkspaceBoundaryHook"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefix_hook() {
        let hook = PrefixHook::new("[IMPORTANT] ");
        let message = Message::user("Hello");
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(message, &[], &mut ctx).await.unwrap();
        assert_eq!(result.text(), "[IMPORTANT] Hello");
    }

    #[tokio::test]
    async fn test_suffix_hook() {
        let hook = SuffixHook::new(" [END]");
        let message = Message::user("Hello");
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(message, &[], &mut ctx).await.unwrap();
        assert_eq!(result.text(), "Hello [END]");
    }

    #[tokio::test]
    async fn test_logging_hook() {
        let hook = LoggingHook::new().with_level(LogLevel::Debug);
        let message = Message::user("Hello");
        let mut ctx = HookContext::new("test-123");

        // Should not modify the message
        let result = hook
            .pre_completion(message.clone(), &[], &mut ctx)
            .await
            .unwrap();
        assert_eq!(result.text(), message.text());
    }

    #[tokio::test]
    async fn test_validation_hook_valid_message() {
        let hook = ValidationHook::new();
        let message = Message::user("Hello, world!");
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(message.clone(), &[], &mut ctx).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_validation_hook_empty_content() {
        let hook = ValidationHook::new();
        // Create a message with empty content directly
        let message = Message::User {
            content: vec![],
            id: None,
        };
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(message, &[], &mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty content"));
    }

    #[tokio::test]
    async fn test_validation_hook_empty_text() {
        use crate::message::UserContent;

        let hook = ValidationHook::new();
        let message = Message::User {
            content: vec![UserContent::Text {
                text: "   ".to_string(),
            }],
            id: None,
        };
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(message, &[], &mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty or whitespace"));
    }

    #[tokio::test]
    async fn test_validation_hook_tool_result_matching() {
        use crate::message::{AssistantContent, ToolCall};

        let hook = ValidationHook::new();

        // Create history with a tool call
        let assistant_msg = Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-123",
                "get_weather",
                r#"{"city": "NYC"}"#,
            ))],
            id: None,
            reasoning: None,
        };

        // Create a tool result that matches
        let tool_result = Message::tool_result("call-123", "Sunny, 72°F");
        let mut ctx = HookContext::new("test-123");

        let result = hook
            .pre_completion(tool_result, &[assistant_msg], &mut ctx)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validation_hook_unmatched_tool_result() {
        let hook = ValidationHook::new();

        // Create a tool result without matching tool call in history
        let tool_result = Message::tool_result("call-unknown", "Some result");
        let mut ctx = HookContext::new("test-123");

        let result = hook.pre_completion(tool_result, &[], &mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("call-unknown"));
        assert!(err.to_string().contains("does not match"));
    }

    #[tokio::test]
    async fn test_validation_hook_unknown_tool_call() {
        use crate::message::{AssistantContent, ToolCall};

        let hook = ValidationHook::with_tools(vec!["get_weather", "search"]);

        // Create an assistant message with an unknown tool call
        let assistant_msg = Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-456",
                "unknown_tool",
                "{}",
            ))],
            id: None,
            reasoning: None,
        };
        let mut ctx = HookContext::new("test-123");

        let result = hook.post_completion_message(assistant_msg, &mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unknown_tool"));
    }

    #[tokio::test]
    async fn test_validation_hook_valid_tool_call() {
        use crate::message::{AssistantContent, ToolCall};

        let hook = ValidationHook::with_tools(vec!["get_weather", "search"]);

        // Create an assistant message with a known tool call
        let assistant_msg = Message::Assistant {
            content: vec![AssistantContent::ToolCall(ToolCall::new(
                "call-789",
                "get_weather",
                r#"{"city": "NYC"}"#,
            ))],
            id: None,
            reasoning: None,
        };
        let mut ctx = HookContext::new("test-123");

        let result = hook.post_completion_message(assistant_msg, &mut ctx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_allows_cwd() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // Create a test file path within workspace
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "test").unwrap();

        let args = serde_json::json!({
            "file_path": file_path.to_string_lossy()
        });

        let result = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();
        assert!(result.should_proceed());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_blocks_outside() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // Try to read a file outside workspace
        let args = serde_json::json!({
            "file_path": "/etc/passwd"
        });

        let result = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();
        assert!(!result.should_proceed());
        assert!(result.block_reason().unwrap().contains("outside"));
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_allows_relative_paths() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        let file_path = subdir.join("test.txt");
        std::fs::write(&file_path, "test").unwrap();

        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // Relative path within workspace
        let args = serde_json::json!({
            "file_path": "subdir/test.txt"
        });

        let result = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();
        assert!(result.should_proceed());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_blocks_path_traversal() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // Try path traversal attack
        let args = serde_json::json!({
            "file_path": "../../../etc/passwd"
        });

        let result = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();
        assert!(!result.should_proceed());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_allows_extra_path() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let extra_dir = tempfile::TempDir::new().unwrap();
        let extra_file = extra_dir.path().join("allowed.txt");
        std::fs::write(&extra_file, "allowed").unwrap();

        let hook =
            WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf()).allow_path(extra_dir.path());
        let mut ctx = HookContext::new("test-123");

        // Access file in allowed extra path
        let args = serde_json::json!({
            "file_path": extra_file.to_string_lossy()
        });

        let result = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();
        assert!(result.should_proceed());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_grep_glob_path() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // grep/glob use "path" parameter
        let args = serde_json::json!({
            "pattern": "*.rs",
            "path": "/etc"
        });

        let result = hook.pre_tool_call("grep", args, &mut ctx).await.unwrap();
        assert!(!result.should_proceed());

        let args2 = serde_json::json!({
            "pattern": "*.rs",
            "path": "/etc"
        });

        let result2 = hook.pre_tool_call("glob", args2, &mut ctx).await.unwrap();
        assert!(!result2.should_proceed());
    }

    #[tokio::test]
    async fn test_workspace_boundary_hook_ignores_other_tools() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let hook = WorkspaceBoundaryHook::new(temp_dir.path().to_path_buf());
        let mut ctx = HookContext::new("test-123");

        // Other tools should pass through
        let args = serde_json::json!({
            "command": "ls /etc"
        });

        let result = hook.pre_tool_call("bash", args, &mut ctx).await.unwrap();
        assert!(result.should_proceed());
    }
}
