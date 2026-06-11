//! Hook Integration for File-History Context Management
//!
//! This module provides a hook implementation that integrates the classification
//! system with the agent's hook chain. It intercepts tool calls and results to:
//!
//! 1. Classify each file operation
//! 2. Track file artifacts
//! 3. Optimize context before sending to the LLM
//!
//! # Usage
//!
//! ```ignore
//! use sombrax_agentic_core::context::classification::FileContextHook;
//! use sombrax_agentic_core::AgentBuilder;
//!
//! let agent = AgentBuilder::new(model)
//!     .hook(FileContextHook::new())
//!     .build();
//! ```
//!
//! # Important Notes
//!
//! The hook classifies tool operations as they occur. The resulting optimization
//! is advisory - it tells you which file-related messages can be dropped or moved.
//! Unclassified messages (normal chat, non-file tools) are not tracked and should
//! be preserved by the caller.

use super::{ClassificationKey, ContextManager, SupersedenceRules};
use crate::context::HookContext;
use crate::error::{HookError, HookResult};
use crate::hook::{Hook, ToolCallDecision};
use crate::message::Message;
use crate::tool::ToolDefinition;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Context key for storing the classification manager in HookContext
pub const FILE_CONTEXT_KEY: &str = "file_context_manager";

/// Global counter for generating unique IDs
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Hook for file-history context management
///
/// This hook intercepts tool calls and results related to file operations,
/// classifies them, and optimizes the context before LLM calls.
#[derive(Clone)]
pub struct FileContextHook {
    /// The context manager (shared across hook invocations)
    manager: Arc<RwLock<ContextManager>>,
    /// Whether to automatically optimize context on pre_completion
    auto_optimize: bool,
    /// Whether to store manager in HookContext for external access
    expose_manager: bool,
}

impl FileContextHook {
    /// Create a new file context hook with default settings
    pub fn new() -> Self {
        Self {
            manager: Arc::new(RwLock::new(ContextManager::new())),
            auto_optimize: true,
            expose_manager: true,
        }
    }

    /// Create with custom supersedence rules
    pub fn with_rules(rules: SupersedenceRules) -> Self {
        Self {
            manager: Arc::new(RwLock::new(ContextManager::with_rules(rules))),
            auto_optimize: true,
            expose_manager: true,
        }
    }

    /// Disable automatic context optimization
    ///
    /// When disabled, the hook will only classify operations but not
    /// modify the context automatically. Use `get_optimization()` to
    /// retrieve optimization recommendations manually.
    pub fn without_auto_optimize(mut self) -> Self {
        self.auto_optimize = false;
        self
    }

    /// Disable exposing the manager in HookContext
    pub fn without_expose_manager(mut self) -> Self {
        self.expose_manager = false;
        self
    }

    /// Get the current optimization result
    ///
    /// Returns optimization recommendations. Note that this only covers
    /// file-related operations. Use `is_classified()` on the result to check
    /// if a message was tracked.
    pub fn compute_optimization(&self) -> ComputedOptimization {
        let manager = self.manager.read().unwrap();
        let result = manager.compute_optimized_order();

        ComputedOptimization {
            keep: result.keep.clone(),
            drop: result.drop.clone(),
            move_to_end: result.move_to_end.clone(),
            classified_messages: manager
                .classifier()
                .classifications()
                .values()
                .map(|c| c.message_id.clone())
                .collect(),
        }
    }

    /// Get files that need read-after-write
    pub fn files_needing_read(&self) -> Vec<std::path::PathBuf> {
        let manager = self.manager.read().unwrap();
        manager.files_needing_read()
    }

    /// Clear all tracked state
    pub fn clear(&self) {
        let mut manager = self.manager.write().unwrap();
        manager.clear();
    }

    /// Get a clone of the manager for external use
    pub fn manager(&self) -> Arc<RwLock<ContextManager>> {
        Arc::clone(&self.manager)
    }

    /// Extract tool name and args from an assistant message's tool calls
    fn extract_tool_calls(message: &Message) -> Vec<(String, String, serde_json::Value)> {
        let tool_calls = message.tool_calls();
        tool_calls
            .into_iter()
            .filter_map(|tc| {
                let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).ok()?;
                Some((tc.id.clone(), tc.function.name.clone(), args))
            })
            .collect()
    }

    /// Generate a unique ID for tool call tracking
    fn generate_unique_id() -> u64 {
        UNIQUE_COUNTER.fetch_add(1, Ordering::SeqCst)
    }
}

impl Default for FileContextHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for FileContextHook {
    fn name(&self) -> &str {
        "FileContextHook"
    }

    /// Classify any tool calls in the message before completion
    ///
    /// This doesn't modify the message, just records classifications.
    async fn pre_completion(
        &self,
        message: Message,
        history: &[Message],
        ctx: &mut HookContext,
    ) -> HookResult<Message> {
        // Store manager reference in context if enabled
        if self.expose_manager && !ctx.contains(FILE_CONTEXT_KEY) {
            // We can't store Arc<RwLock<_>> directly, so store a marker
            // The actual manager is accessed via the hook
            ctx.set(FILE_CONTEXT_KEY, true)
                .map_err(|e| HookError::HookFailed {
                    hook_name: self.name().to_string(),
                    stage: crate::error::HookStage::PreCompletion,
                    message: format!("Failed to store context key: {}", e),
                    source: None,
                })?;
        }

        // Process any tool calls in the history that haven't been classified
        // This handles cases where messages are added externally
        for hist_msg in history {
            if let Some(msg_id) = hist_msg.id() {
                let tool_calls = Self::extract_tool_calls(hist_msg);
                for (call_id, tool_name, args) in tool_calls {
                    let mut manager = self.manager.write().unwrap();
                    // Check if already classified using the composite key
                    let key = ClassificationKey::new(msg_id, &call_id);
                    if manager
                        .classifier()
                        .get_classification_by_key(&key)
                        .is_none()
                    {
                        manager
                            .classifier_mut()
                            .classify_tool_call(&tool_name, &args, msg_id, &call_id);
                    }
                }
            }
        }

        Ok(message)
    }

    /// Classify tool calls made by the assistant
    async fn post_completion_message(
        &self,
        message: Message,
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        // Extract and classify any tool calls in the response
        if let Some(msg_id) = message.id() {
            let tool_calls = Self::extract_tool_calls(&message);
            let mut manager = self.manager.write().unwrap();

            for (call_id, tool_name, args) in tool_calls {
                manager
                    .classifier_mut()
                    .classify_tool_call(&tool_name, &args, msg_id, &call_id);
            }
        }

        Ok(message)
    }

    /// Record tool call arguments for classification
    async fn pre_tool_call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        ctx: &mut HookContext,
    ) -> HookResult<ToolCallDecision> {
        // Generate a unique call ID for this specific invocation
        let unique_id = Self::generate_unique_id();
        let tracking_key = format!("tool_call_tracking_{}", unique_id);

        // Store the tool name, args, and unique ID in context for post_tool_call
        let call_info = serde_json::json!({
            "tool_name": tool_name,
            "args": args.clone(),
            "unique_id": unique_id
        });
        ctx.set(&tracking_key, call_info).ok();

        // Also store the current unique_id as the "latest" for this request
        // This allows post_tool_call to find it
        ctx.set(&format!("latest_tool_call_{}", ctx.request_id), unique_id)
            .ok();

        Ok(ToolCallDecision::Proceed(args))
    }

    /// Classify tool results
    async fn post_tool_call(
        &self,
        tool_name: &str,
        result: String,
        ctx: &mut HookContext,
    ) -> HookResult<String> {
        // Get the unique ID from the latest pre_tool_call
        let unique_id: u64 = ctx
            .get(&format!("latest_tool_call_{}", ctx.request_id))
            .and_then(|r| r.ok())
            .unwrap_or_else(Self::generate_unique_id);

        // Generate unique IDs for this tool result
        // Use the tool_call_id from context if available, or generate one
        let call_id = format!("call_{}_{}", ctx.request_id, unique_id);
        let msg_id = format!("result_{}_{}", ctx.request_id, unique_id);

        let mut manager = self.manager.write().unwrap();
        manager
            .classifier_mut()
            .classify_tool_result(tool_name, &result, &msg_id, &call_id);

        Ok(result)
    }

    /// Filter tools (pass-through, no modification)
    async fn filter_tools(
        &self,
        tools: Vec<ToolDefinition>,
        _ctx: &mut HookContext,
    ) -> HookResult<Vec<ToolDefinition>> {
        Ok(tools)
    }

    /// Handle assistant messages (for logging/debugging)
    async fn on_assistant_message(
        &self,
        message: &Message,
        _ctx: &mut HookContext,
    ) -> HookResult<()> {
        // Log file operation classifications for debugging
        if let Some(msg_id) = message.id() {
            let manager = self.manager.read().unwrap();
            let classifications = manager.classifier().get_classifications_for_message(msg_id);
            for classification in classifications {
                tracing::debug!(
                    message_id = %msg_id,
                    tool_call_id = %classification.tool_call_id,
                    operation = ?classification.operation,
                    file_path = ?classification.file_path,
                    "File operation classified"
                );
            }
        }
        Ok(())
    }
}

/// Computed optimization result that can be used independently of the manager
#[derive(Debug, Clone)]
pub struct ComputedOptimization {
    /// Classification keys to keep
    pub keep: Vec<ClassificationKey>,
    /// Classification keys to drop
    pub drop: Vec<ClassificationKey>,
    /// Classification keys to move to end
    pub move_to_end: Vec<ClassificationKey>,
    /// Set of all classified message IDs
    pub classified_messages: std::collections::HashSet<String>,
}

impl ComputedOptimization {
    /// Check if a message ID was classified (is a file operation)
    pub fn is_classified(&self, message_id: &str) -> bool {
        self.classified_messages.contains(message_id)
    }

    /// Check if a classification key should be included
    pub fn should_include_key(&self, key: &ClassificationKey) -> bool {
        self.keep.contains(key) || self.move_to_end.contains(key)
    }

    /// Check if a classification key should be dropped
    pub fn should_drop_key(&self, key: &ClassificationKey) -> bool {
        self.drop.contains(key)
    }

    /// Check if a message should be dropped
    ///
    /// Returns true only if the message is classified AND all its classifications
    /// should be dropped. Unclassified messages return false (don't drop them).
    pub fn should_drop(&self, message_id: &str) -> bool {
        if !self.is_classified(message_id) {
            return false;
        }

        // Check if all keys for this message are in the drop list
        let keys_for_message: Vec<_> = self
            .drop
            .iter()
            .chain(self.keep.iter())
            .chain(self.move_to_end.iter())
            .filter(|k| k.message_id == message_id)
            .collect();

        !keys_for_message.is_empty() && keys_for_message.iter().all(|k| self.drop.contains(k))
    }

    /// Check if a message should be included
    ///
    /// Returns true if the message is classified AND at least one classification
    /// should be kept. Unclassified messages return false (caller decides).
    pub fn should_include(&self, message_id: &str) -> bool {
        if !self.is_classified(message_id) {
            return false;
        }

        self.keep
            .iter()
            .chain(self.move_to_end.iter())
            .any(|k| k.message_id == message_id)
    }
}

/// Helper trait for accessing file context from HookContext
pub trait FileContextExt {
    /// Get the file context optimization result
    fn file_context_optimization(&self, hook: &FileContextHook) -> ComputedOptimization;

    /// Check if a message should be included in context
    fn should_include_message(&self, hook: &FileContextHook, message_id: &str) -> bool;
}

impl FileContextExt for HookContext {
    fn file_context_optimization(&self, hook: &FileContextHook) -> ComputedOptimization {
        hook.compute_optimization()
    }

    fn should_include_message(&self, hook: &FileContextHook, message_id: &str) -> bool {
        let result = hook.compute_optimization();
        // If not classified, default to include
        if !result.is_classified(message_id) {
            return true;
        }
        result.should_include(message_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_context_hook_creation() {
        let hook = FileContextHook::new();
        assert!(hook.auto_optimize);
        assert!(hook.expose_manager);
    }

    #[test]
    fn test_file_context_hook_with_rules() {
        let rules = SupersedenceRules {
            preserve_edit_history: true,
            ..Default::default()
        };
        let hook = FileContextHook::with_rules(rules);
        assert!(hook.manager.read().unwrap().rules().preserve_edit_history);
    }

    #[test]
    fn test_file_context_hook_clear() {
        let hook = FileContextHook::new();

        // Add some classifications
        {
            let mut manager = hook.manager.write().unwrap();
            manager.classifier_mut().classify_tool_call(
                "read",
                &serde_json::json!({"file_path": "/test.rs"}),
                "msg-1",
                "call-1",
            );
        }

        assert!(!hook
            .manager
            .read()
            .unwrap()
            .classifier()
            .classifications()
            .is_empty());

        hook.clear();

        assert!(hook
            .manager
            .read()
            .unwrap()
            .classifier()
            .classifications()
            .is_empty());
    }

    #[tokio::test]
    async fn test_file_context_hook_tool_call_classification() {
        let hook = FileContextHook::new();
        let args = serde_json::json!({"file_path": "/src/main.rs"});

        // Simulate pre_tool_call
        let mut ctx = HookContext::new("test-request");
        let decision = hook.pre_tool_call("read", args, &mut ctx).await.unwrap();

        assert!(decision.should_proceed());

        // Simulate post_tool_call
        let result = hook
            .post_tool_call(
                "read",
                r#"{"file_path":"/src/main.rs"}"#.to_string(),
                &mut ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_empty());

        // Verify classification was recorded
        let optimization = hook.compute_optimization();
        assert!(!optimization.keep.is_empty() || !optimization.move_to_end.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_tool_calls_unique_ids() {
        let hook = FileContextHook::new();
        let mut ctx = HookContext::new("test-request");

        // First tool call
        let args1 = serde_json::json!({"file_path": "/src/file1.rs"});
        hook.pre_tool_call("read", args1, &mut ctx).await.unwrap();
        hook.post_tool_call(
            "read",
            r#"{"file_path":"/src/file1.rs"}"#.to_string(),
            &mut ctx,
        )
        .await
        .unwrap();

        // Second tool call (same request, different file)
        let args2 = serde_json::json!({"file_path": "/src/file2.rs"});
        hook.pre_tool_call("read", args2, &mut ctx).await.unwrap();
        hook.post_tool_call(
            "read",
            r#"{"file_path":"/src/file2.rs"}"#.to_string(),
            &mut ctx,
        )
        .await
        .unwrap();

        // Both should be recorded with unique IDs
        let manager = hook.manager.read().unwrap();
        let classifications: Vec<_> = manager.classifier().classifications().values().collect();
        assert_eq!(classifications.len(), 2);

        // Verify they have different tool_call_ids
        let call_ids: std::collections::HashSet<_> =
            classifications.iter().map(|c| &c.tool_call_id).collect();
        assert_eq!(call_ids.len(), 2, "Tool calls should have unique IDs");
    }

    #[test]
    fn test_computed_optimization_unclassified() {
        let opt = ComputedOptimization {
            keep: vec![],
            drop: vec![],
            move_to_end: vec![],
            classified_messages: std::collections::HashSet::new(),
        };

        // Unclassified messages should not be dropped
        assert!(!opt.should_drop("unclassified-msg"));
        assert!(!opt.is_classified("unclassified-msg"));
    }
}
