//! Hook trait and chain execution
//!
//! Provides the Hook trait for content-modifying hooks and HookChain for
//! sequential hook execution.

pub mod builtin;
pub mod context;

pub use context::HookContext;

use crate::error::{HookError, HookResult, HookStage};
use crate::message::Message;
use crate::provider::CompletionResponse;
use crate::telemetry::Metrics;
use crate::tool::ToolDefinition;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

/// Re-export HookContext from context module
pub use crate::context::HookContext as ContextHookContext;

/// Decision returned by pre_tool_call hook
#[derive(Debug, Clone)]
pub enum ToolCallDecision {
    /// Proceed with the tool call using (possibly modified) arguments
    Proceed(serde_json::Value),
    /// Block the tool call with a reason (returned to model as error)
    Block(String),
}

impl ToolCallDecision {
    /// Check if this decision allows proceeding
    pub fn should_proceed(&self) -> bool {
        matches!(self, ToolCallDecision::Proceed(_))
    }

    /// Get the arguments if proceeding
    pub fn args(&self) -> Option<&serde_json::Value> {
        match self {
            ToolCallDecision::Proceed(args) => Some(args),
            ToolCallDecision::Block(_) => None,
        }
    }

    /// Get the block reason if blocked
    pub fn block_reason(&self) -> Option<&str> {
        match self {
            ToolCallDecision::Block(reason) => Some(reason),
            ToolCallDecision::Proceed(_) => None,
        }
    }
}

/// Content-modifying hook trait (FR-001, FR-002, FR-003, FR-007, FR-008, FR-016)
///
/// Hooks can intercept and modify content at various points in the agent lifecycle.
/// All methods have default pass-through implementations.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone)]
/// struct LoggingHook;
///
/// impl Hook for LoggingHook {
///     async fn pre_completion(
///         &self,
///         message: Message,
///         history: &[Message],
///         ctx: &mut HookContext,
///     ) -> HookResult<Message> {
///         tracing::info!("Processing message: {:?}", message);
///         Ok(message) // Pass through unchanged
///     }
/// }
/// ```
pub trait Hook: Clone + Send + Sync + 'static {
    /// Called before the completion request is sent to the model (FR-001)
    ///
    /// Can modify the message content before it reaches the LLM.
    fn pre_completion(
        &self,
        message: Message,
        history: &[Message],
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        let _ = (history, ctx);
        async { Ok(message) }
    }

    /// Called after the completion response is received (FR-002)
    ///
    /// Can modify the response content before it's returned to the caller.
    fn post_completion<R: Send + Sync + 'static>(
        &self,
        response: CompletionResponse<R>,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<CompletionResponse<R>>> + Send {
        let _ = ctx;
        async { Ok(response) }
    }

    /// Message-only post-completion hook for dynamic dispatch
    ///
    /// Override this to modify the response message when using dynamic dispatch.
    /// This is called by the hook chain for all hooks.
    fn post_completion_message(
        &self,
        message: Message,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        let _ = ctx;
        async { Ok(message) }
    }

    /// Called before a tool is executed (FR-007)
    ///
    /// Can modify arguments, or block the tool call entirely.
    fn pre_tool_call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<ToolCallDecision>> + Send {
        let _ = (tool_name, ctx);
        async { Ok(ToolCallDecision::Proceed(args)) }
    }

    /// Called after a tool returns its result (FR-008)
    ///
    /// Can modify the tool result before it's added to conversation history.
    fn post_tool_call(
        &self,
        tool_name: &str,
        result: String,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<String>> + Send {
        let _ = (tool_name, ctx);
        async { Ok(result) }
    }

    /// Called before tool definitions are sent to the model (FR-016)
    ///
    /// Can filter or modify which tools the model sees.
    fn filter_tools(
        &self,
        tools: Vec<ToolDefinition>,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<Vec<ToolDefinition>>> + Send {
        let _ = ctx;
        async { Ok(tools) }
    }

    /// Called after the model returns a response (including during tool loop)
    ///
    /// This is called for every model response, whether or not it contains tool calls.
    /// Useful for displaying/logging assistant text content before tool execution.
    fn on_assistant_message(
        &self,
        message: &Message,
        ctx: &mut crate::context::HookContext,
    ) -> impl Future<Output = HookResult<()>> + Send {
        let _ = (message, ctx);
        async { Ok(()) }
    }

    /// Returns the hook name for error reporting
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// No-op hook implementation for unit type.
///
/// This allows `()` to be used as a default generic parameter for hooks,
/// enabling patterns like `AgentBuildOptions<H: Hook = ()>` where no
/// hook is provided.
impl Hook for () {
    fn name(&self) -> &str {
        "NoOpHook"
    }
}

/// Dynamic dispatch trait for hooks (type-erased)
pub trait HookDyn: Send + Sync {
    /// Returns the hook name
    fn name(&self) -> &str;

    /// Pre-completion hook with dynamic dispatch
    fn pre_completion<'a>(
        &'a self,
        message: Message,
        history: &'a [Message],
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Message>> + Send + 'a>>;

    /// Post-completion hook with dynamic dispatch
    fn post_completion_dyn<'a>(
        &'a self,
        message: Message,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Message>> + Send + 'a>>;

    /// Pre-tool-call hook with dynamic dispatch
    fn pre_tool_call<'a>(
        &'a self,
        tool_name: &'a str,
        args: serde_json::Value,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<ToolCallDecision>> + Send + 'a>>;

    /// Post-tool-call hook with dynamic dispatch
    fn post_tool_call<'a>(
        &'a self,
        tool_name: &'a str,
        result: String,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<String>> + Send + 'a>>;

    /// Filter-tools hook with dynamic dispatch
    fn filter_tools<'a>(
        &'a self,
        tools: Vec<ToolDefinition>,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Vec<ToolDefinition>>> + Send + 'a>>;

    /// On-assistant-message hook with dynamic dispatch
    fn on_assistant_message<'a>(
        &'a self,
        message: &'a Message,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<()>> + Send + 'a>>;
}

/// Wrapper to implement HookDyn for any Hook
struct HookWrapper<H: Hook>(H);

impl<H: Hook> HookDyn for HookWrapper<H> {
    fn name(&self) -> &str {
        Hook::name(&self.0)
    }

    fn pre_completion<'a>(
        &'a self,
        message: Message,
        history: &'a [Message],
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Message>> + Send + 'a>> {
        Box::pin(Hook::pre_completion(&self.0, message, history, ctx))
    }

    fn post_completion_dyn<'a>(
        &'a self,
        message: Message,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Message>> + Send + 'a>> {
        Box::pin(Hook::post_completion_message(&self.0, message, ctx))
    }

    fn pre_tool_call<'a>(
        &'a self,
        tool_name: &'a str,
        args: serde_json::Value,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<ToolCallDecision>> + Send + 'a>> {
        Box::pin(Hook::pre_tool_call(&self.0, tool_name, args, ctx))
    }

    fn post_tool_call<'a>(
        &'a self,
        tool_name: &'a str,
        result: String,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<String>> + Send + 'a>> {
        Box::pin(Hook::post_tool_call(&self.0, tool_name, result, ctx))
    }

    fn filter_tools<'a>(
        &'a self,
        tools: Vec<ToolDefinition>,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<Vec<ToolDefinition>>> + Send + 'a>> {
        Box::pin(Hook::filter_tools(&self.0, tools, ctx))
    }

    fn on_assistant_message<'a>(
        &'a self,
        message: &'a Message,
        ctx: &'a mut crate::context::HookContext,
    ) -> Pin<Box<dyn Future<Output = HookResult<()>> + Send + 'a>> {
        Box::pin(Hook::on_assistant_message(&self.0, message, ctx))
    }
}

/// An ordered chain of hooks
///
/// Hooks are executed sequentially, with each hook receiving the output of the previous.
/// The first error stops the chain and propagates.
#[derive(Default)]
pub struct HookChain {
    hooks: Vec<Arc<dyn HookDyn>>,
    metrics: Option<Metrics>,
}

impl HookChain {
    /// Create a new empty hook chain
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            metrics: Some(Metrics::global()),
        }
    }

    /// Add a hook to the chain
    pub fn add<H: Hook>(&mut self, hook: H) {
        self.hooks.push(Arc::new(HookWrapper(hook)));
    }

    /// Create a new chain with a hook added
    pub fn with<H: Hook>(mut self, hook: H) -> Self {
        self.add(hook);
        self
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Get the number of hooks in the chain
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Execute pre-completion hooks in order
    pub async fn execute_pre_completion(
        &self,
        mut message: Message,
        history: &[Message],
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<Message> {
        tracing::debug!("executing pre_completion hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();

            tracing::debug!(
                hook_name = %hook_name,
                stage = "pre_completion",
                "executing hook"
            );

            message = hook
                .pre_completion(message, history, ctx)
                .await
                .map_err(|e| match e {
                    HookError::HookFailed {
                        message, source, ..
                    } => HookError::HookFailed {
                        hook_name: hook_name.clone(),
                        stage: HookStage::PreCompletion,
                        message,
                        source,
                    },
                    other => other,
                })?;

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                "pre_completion hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "pre_completion", duration);
            }
        }

        Ok(message)
    }

    /// Execute post-completion hooks in order
    pub async fn execute_post_completion<R: Send + Sync + 'static>(
        &self,
        mut response: CompletionResponse<R>,
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<CompletionResponse<R>> {
        tracing::debug!("executing post_completion hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();

            tracing::debug!(
                hook_name = %hook_name,
                stage = "post_completion",
                "executing hook"
            );

            // For dynamic dispatch, we process the message only
            let new_message = hook
                .post_completion_dyn(response.message, ctx)
                .await
                .map_err(|e| match e {
                    HookError::HookFailed {
                        message, source, ..
                    } => HookError::HookFailed {
                        hook_name: hook_name.clone(),
                        stage: HookStage::PostCompletion,
                        message,
                        source,
                    },
                    other => other,
                })?;

            response = CompletionResponse {
                message: new_message,
                usage: response.usage,
                raw: response.raw,
                reasoning_content: response.reasoning_content,
                finish_reason: response.finish_reason,
            };

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                "post_completion hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "post_completion", duration);
            }
        }

        Ok(response)
    }

    /// Execute pre-tool-call hooks in order
    pub async fn execute_pre_tool_call(
        &self,
        tool_name: &str,
        mut args: serde_json::Value,
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<ToolCallDecision> {
        tracing::debug!(tool_name = %tool_name, "executing pre_tool_call hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();

            let decision = hook
                .pre_tool_call(tool_name, args, ctx)
                .await
                .map_err(|e| match e {
                    HookError::HookFailed {
                        message, source, ..
                    } => HookError::HookFailed {
                        hook_name: hook_name.clone(),
                        stage: HookStage::PreToolCall,
                        message,
                        source,
                    },
                    other => other,
                })?;

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                "pre_tool_call hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "pre_tool_call", duration);
            }

            match decision {
                ToolCallDecision::Block(reason) => {
                    tracing::info!(
                        hook_name = %hook_name,
                        tool_name = %tool_name,
                        reason = %reason,
                        "tool call blocked by hook"
                    );
                    return Ok(ToolCallDecision::Block(reason));
                }
                ToolCallDecision::Proceed(new_args) => {
                    args = new_args;
                }
            }
        }

        Ok(ToolCallDecision::Proceed(args))
    }

    /// Execute post-tool-call hooks in order
    pub async fn execute_post_tool_call(
        &self,
        tool_name: &str,
        mut result: String,
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<String> {
        tracing::debug!(tool_name = %tool_name, "executing post_tool_call hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();

            result = hook
                .post_tool_call(tool_name, result, ctx)
                .await
                .map_err(|e| match e {
                    HookError::HookFailed {
                        message, source, ..
                    } => HookError::HookFailed {
                        hook_name: hook_name.clone(),
                        stage: HookStage::PostToolCall,
                        message,
                        source,
                    },
                    other => other,
                })?;

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                "post_tool_call hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "post_tool_call", duration);
            }
        }

        Ok(result)
    }

    /// Execute filter-tools hooks in order
    pub async fn execute_filter_tools(
        &self,
        mut tools: Vec<ToolDefinition>,
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<Vec<ToolDefinition>> {
        tracing::debug!("executing filter_tools hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();
            let tools_before = tools.len();

            tools = hook.filter_tools(tools, ctx).await.map_err(|e| match e {
                HookError::HookFailed {
                    message, source, ..
                } => HookError::HookFailed {
                    hook_name: hook_name.clone(),
                    stage: HookStage::FilterTools,
                    message,
                    source,
                },
                other => other,
            })?;

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                tools_before = %tools_before,
                tools_after = %tools.len(),
                "filter_tools hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "filter_tools", duration);
            }
        }

        Ok(tools)
    }

    /// Execute on-assistant-message hooks in order
    ///
    /// Called after the model returns a response, before tool execution begins.
    /// This allows hooks to display or log assistant text content.
    pub async fn execute_on_assistant_message(
        &self,
        message: &Message,
        ctx: &mut crate::context::HookContext,
    ) -> HookResult<()> {
        tracing::debug!("executing on_assistant_message hook chain");

        for hook in &self.hooks {
            if ctx.is_cancelled() {
                return Err(HookError::Cancelled);
            }

            let hook_name = hook.name().to_string();
            let start = Instant::now();

            tracing::debug!(
                hook_name = %hook_name,
                stage = "on_assistant_message",
                "executing hook"
            );

            hook.on_assistant_message(message, ctx)
                .await
                .map_err(|e| match e {
                    HookError::HookFailed {
                        message, source, ..
                    } => HookError::HookFailed {
                        hook_name: hook_name.clone(),
                        stage: HookStage::OnAssistantMessage,
                        message,
                        source,
                    },
                    other => other,
                })?;

            let duration = start.elapsed();
            tracing::debug!(
                hook_name = %hook_name,
                duration_ms = %duration.as_millis(),
                "on_assistant_message hook completed"
            );

            // Record hook duration metric (FR-021)
            if let Some(metrics) = &self.metrics {
                metrics.record_hook_duration(&hook_name, "on_assistant_message", duration);
            }
        }

        Ok(())
    }
}

impl Clone for HookChain {
    fn clone(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
            metrics: Some(Metrics::global()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestHook {
        prefix: String,
    }

    impl Hook for TestHook {
        async fn pre_completion(
            &self,
            mut message: Message,
            _history: &[Message],
            _ctx: &mut crate::context::HookContext,
        ) -> HookResult<Message> {
            message.prepend_text(&self.prefix);
            Ok(message)
        }
    }

    #[tokio::test]
    async fn test_hook_chain_pre_completion() {
        let mut chain = HookChain::new();
        chain.add(TestHook {
            prefix: "A: ".to_string(),
        });
        chain.add(TestHook {
            prefix: "B: ".to_string(),
        });

        let mut ctx = crate::context::HookContext::new("test-123");
        let message = Message::user("Hello");

        let result = chain
            .execute_pre_completion(message, &[], &mut ctx)
            .await
            .unwrap();

        // B runs after A, so "B: " is prepended to "A: Hello"
        assert_eq!(result.text(), "B: A: Hello");
    }

    #[test]
    fn test_tool_call_decision() {
        let proceed = ToolCallDecision::Proceed(serde_json::json!({"a": 1}));
        assert!(proceed.should_proceed());
        assert!(proceed.args().is_some());
        assert!(proceed.block_reason().is_none());

        let block = ToolCallDecision::Block("Not allowed".to_string());
        assert!(!block.should_proceed());
        assert!(block.args().is_none());
        assert_eq!(block.block_reason(), Some("Not allowed"));
    }

    #[test]
    fn test_hook_chain_operations() {
        let mut chain = HookChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        chain.add(TestHook {
            prefix: "A".to_string(),
        });
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);

        let chain2 = chain.with(TestHook {
            prefix: "B".to_string(),
        });
        assert_eq!(chain2.len(), 2);
    }
}
