//! Context types for hooks and agents
//!
//! Provides HookContext for per-request state and SharedContext for cross-agent state.

pub mod classification;
mod optimizer;

pub use optimizer::{
    ContextOptimizer, OptimizationConfig, PriorityOptimizer, RecencyOptimizer, TruncationOptimizer,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cancellation signal for graceful interruption
#[derive(Clone, Debug)]
pub struct CancelSignal(Arc<AtomicBool>);

impl CancelSignal {
    /// Create a new cancellation signal
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Signal cancellation
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Reset the cancellation signal
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl Default for CancelSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutable context passed to all hooks in a chain (FR-003a)
///
/// Enables state sharing and inter-hook communication within a request.
///
/// # Example
///
/// ```ignore
/// impl Hook for TranslationHook {
///     async fn pre_completion(
///         &self,
///         message: Message,
///         _history: &[Message],
///         ctx: &mut HookContext,
///     ) -> HookResult<Message> {
///         // Store detected language for post-processing
///         ctx.set("source_lang", "es")?;
///         Ok(message)
///     }
/// }
/// ```
#[derive(Debug)]
pub struct HookContext {
    /// Unique identifier for this request
    pub request_id: String,

    /// When the request started
    pub started_at: Instant,

    /// Cancellation signal
    pub cancel_signal: CancelSignal,

    /// OpenTelemetry span for tracing (FR-020)
    pub span: tracing::Span,

    /// User-defined key-value storage for inter-hook communication
    extras: HashMap<String, serde_json::Value>,
}

impl HookContext {
    /// Create a new hook context with the given request ID
    pub fn new(request_id: impl Into<String>) -> Self {
        let request_id = request_id.into();
        Self {
            span: tracing::info_span!("hook_chain", request_id = %request_id),
            request_id,
            started_at: Instant::now(),
            cancel_signal: CancelSignal::new(),
            extras: HashMap::new(),
        }
    }

    /// Create a new hook context with a generated UUID
    pub fn new_with_uuid() -> Self {
        Self::new(uuid::Uuid::new_v4().to_string())
    }

    /// Store a value in the context
    pub fn set<T: serde::Serialize>(
        &mut self,
        key: &str,
        value: T,
    ) -> Result<(), serde_json::Error> {
        self.extras
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    /// Retrieve a value from the context
    pub fn get<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<T, serde_json::Error>> {
        self.extras
            .get(key)
            .map(|v| serde_json::from_value(v.clone()))
    }

    /// Remove a value from the context
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.extras.remove(key)
    }

    /// Check if a key exists
    pub fn contains(&self, key: &str) -> bool {
        self.extras.contains_key(key)
    }

    /// Get elapsed time since request started
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Check if the request has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_signal.is_cancelled()
    }

    /// Get all keys in the extras store
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.extras.keys().map(|s| s.as_str())
    }

    /// Generate the next incremental tool call ID for this request.
    ///
    /// Returns IDs in the format `tool_call_1`, `tool_call_2`, etc.
    /// The counter is per-request, stored in the context extras.
    pub fn next_tool_call_id(&mut self) -> String {
        const TOOL_CALL_COUNTER_KEY: &str = "__tool_call_counter";
        let counter: u32 = self
            .get(TOOL_CALL_COUNTER_KEY)
            .and_then(|r| r.ok())
            .unwrap_or(0);
        let next = counter + 1;
        // This should always succeed for a simple u32
        let _ = self.set(TOOL_CALL_COUNTER_KEY, next);
        format!("tool_call_{}", next)
    }
}

/// Session-scoped shared context (FR-011)
///
/// Persists for the lifetime of the AgentRegistry, accessible by all registered agents.
#[derive(Debug, Default)]
pub struct SharedContext {
    data: HashMap<String, serde_json::Value>,
}

impl SharedContext {
    /// Create a new empty shared context
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a value by key, deserializing to the requested type
    pub fn get<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<T, serde_json::Error>> {
        self.data
            .get(key)
            .map(|v| serde_json::from_value(v.clone()))
    }

    /// Set a value, serializing from the given type
    pub fn set<T: serde::Serialize>(
        &mut self,
        key: &str,
        value: T,
    ) -> Result<(), serde_json::Error> {
        self.data
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    /// Remove a value by key
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.data.remove(key)
    }

    /// List all keys
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.data.keys().map(|s| s.as_str())
    }

    /// Check if a key exists
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the context is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Thread-safe wrapper for SharedContext
pub type SharedContextHandle = Arc<RwLock<SharedContext>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_signal() {
        let signal = CancelSignal::new();
        assert!(!signal.is_cancelled());

        signal.cancel();
        assert!(signal.is_cancelled());

        signal.reset();
        assert!(!signal.is_cancelled());
    }

    #[test]
    fn test_hook_context_set_get() {
        let mut ctx = HookContext::new("test-123");

        ctx.set("key1", "value1").unwrap();
        ctx.set("key2", 42i32).unwrap();

        assert_eq!(ctx.get::<String>("key1").unwrap().unwrap(), "value1");
        assert_eq!(ctx.get::<i32>("key2").unwrap().unwrap(), 42);
        assert!(ctx.get::<String>("nonexistent").is_none());
    }

    #[test]
    fn test_hook_context_remove() {
        let mut ctx = HookContext::new("test-123");
        ctx.set("key", "value").unwrap();

        assert!(ctx.contains("key"));
        ctx.remove("key");
        assert!(!ctx.contains("key"));
    }

    #[test]
    fn test_shared_context() {
        let mut ctx = SharedContext::new();

        ctx.set("user_id", "user-123").unwrap();
        ctx.set("count", 5i32).unwrap();

        assert_eq!(ctx.get::<String>("user_id").unwrap().unwrap(), "user-123");
        assert_eq!(ctx.get::<i32>("count").unwrap().unwrap(), 5);
        assert_eq!(ctx.len(), 2);

        ctx.remove("user_id");
        assert_eq!(ctx.len(), 1);

        ctx.clear();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_next_tool_call_id() {
        let mut ctx = HookContext::new("test-123");

        // First call should return tool_call_1
        assert_eq!(ctx.next_tool_call_id(), "tool_call_1");

        // Subsequent calls should increment
        assert_eq!(ctx.next_tool_call_id(), "tool_call_2");
        assert_eq!(ctx.next_tool_call_id(), "tool_call_3");

        // New context should start from 1 again
        let mut ctx2 = HookContext::new("test-456");
        assert_eq!(ctx2.next_tool_call_id(), "tool_call_1");
    }
}
