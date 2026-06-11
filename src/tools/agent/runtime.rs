//! Agent runtime abstraction for sub-agent spawning
//!
//! This module provides the [`AgentRuntime`] trait that allows TaskTool to delegate
//! sub-agent execution to a real agent orchestrator. This decouples the tools module
//! from any specific agent runtime implementation.

use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request to spawn a sub-agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentRequest {
    /// Short description of the task (3-5 words)
    pub description: String,
    /// Detailed prompt for the sub-agent
    pub prompt: String,
    /// Type of sub-agent to spawn (e.g., "general-purpose", "explore", "plan")
    pub subagent_type: String,
    /// Override model (optional, inherits from parent if None)
    pub model: Option<String>,
    /// Parent session ID for tracking hierarchy
    pub parent_session_id: String,
    /// Current recursion depth
    pub current_depth: usize,
    /// Initial messages to seed the sub-agent's conversation history.
    /// These are typically source code context or other pre-loaded content
    /// that should be available to the sub-agent before processing the prompt.
    #[serde(default)]
    pub initial_messages: Vec<Message>,
}

/// Response from a sub-agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentResponse {
    /// Child session ID created for this task
    pub child_session_id: String,
    /// Response content from the sub-agent
    pub response: String,
    /// Number of conversation turns in child execution (if available)
    pub conversation_turns: Option<usize>,
}

/// Trait for delegating sub-agent execution to a runtime
///
/// Implementations of this trait connect TaskTool to an actual agent runtime
/// that can spawn and execute sub-agents. Without a runtime, TaskTool operates
/// in no-op mode.
///
/// # Example Implementation
///
/// ```rust,ignore
/// use sombrax_agentic_core::{AgentRegistry, AgentRequest};
/// use sombrax_agentic_core::tools::agent::{AgentRuntime, SubAgentRequest, SubAgentResponse};
/// use std::sync::Arc;
///
/// struct RegistryRuntime {
///     registry: Arc<AgentRegistry>,
/// }
///
/// #[async_trait::async_trait]
/// impl AgentRuntime for RegistryRuntime {
///     async fn spawn_subagent(&self, request: SubAgentRequest) -> Result<SubAgentResponse, String> {
///         let agent_request = AgentRequest::new(&request.prompt);
///         let response = self.registry
///             .invoke(&request.subagent_type, agent_request)
///             .await
///             .map_err(|e| e.to_string())?;
///
///         Ok(SubAgentResponse {
///             child_session_id: format!("{}-child", request.parent_session_id),
///             response: response.content,
///             conversation_turns: None,
///         })
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait AgentRuntime: Send + Sync {
    /// Spawn a sub-agent to handle the given request
    ///
    /// # Arguments
    /// * `request` - The sub-agent request containing prompt, type, and context
    ///
    /// # Returns
    /// * `Ok(SubAgentResponse)` - The sub-agent completed successfully
    /// * `Err(String)` - The sub-agent failed with the given error message
    async fn spawn_subagent(&self, request: SubAgentRequest) -> Result<SubAgentResponse, String>;

    /// Check if this runtime supports the given sub-agent type
    ///
    /// Default implementation returns true for all types.
    fn supports_subagent_type(&self, _subagent_type: &str) -> bool {
        true
    }

    /// Get the maximum recursion depth supported by this runtime
    ///
    /// Default is 5, matching FR-021 requirement.
    fn max_depth(&self) -> usize {
        5
    }
}

/// No-op runtime that returns a placeholder response
///
/// This runtime is used when no real agent runtime is configured.
/// It clearly indicates that sub-agent execution is not available
/// and provides guidance on how to enable it.
#[derive(Debug, Clone, Default)]
pub struct NoOpRuntime;

impl NoOpRuntime {
    /// Create a new no-op runtime
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AgentRuntime for NoOpRuntime {
    async fn spawn_subagent(&self, request: SubAgentRequest) -> Result<SubAgentResponse, String> {
        tracing::warn!(
            subagent_type = %request.subagent_type,
            description = %request.description,
            "TaskTool operating in no-op mode: no AgentRuntime configured. \
             To enable real sub-agent execution, inject an AgentRuntime via TaskTool::with_runtime()"
        );

        Err(format!(
            "Sub-agent execution is not available. No AgentRuntime is configured. \
             Task '{}' (type: {}) cannot be executed. \
             Configure an AgentRuntime to enable sub-agent spawning.",
            request.description, request.subagent_type
        ))
    }

    fn supports_subagent_type(&self, _subagent_type: &str) -> bool {
        false
    }
}

/// Create a shared no-op runtime instance
pub fn no_op_runtime() -> Arc<dyn AgentRuntime> {
    Arc::new(NoOpRuntime::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_no_op_runtime_returns_error() {
        let runtime = NoOpRuntime::new();
        let request = SubAgentRequest {
            description: "Test task".to_string(),
            prompt: "Do something".to_string(),
            subagent_type: "general-purpose".to_string(),
            model: None,
            parent_session_id: "test-session".to_string(),
            current_depth: 0,
            initial_messages: vec![],
        };

        let result = runtime.spawn_subagent(request).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("No AgentRuntime is configured"));
    }

    #[test]
    fn test_no_op_runtime_does_not_support_types() {
        let runtime = NoOpRuntime::new();
        assert!(!runtime.supports_subagent_type("general-purpose"));
        assert!(!runtime.supports_subagent_type("explore"));
    }

    #[test]
    fn test_default_max_depth() {
        let runtime = NoOpRuntime::new();
        assert_eq!(runtime.max_depth(), 5);
    }
}
