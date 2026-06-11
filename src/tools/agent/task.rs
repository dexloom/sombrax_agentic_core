//! Task tool for spawning sub-agents
//!
//! The TaskTool enables spawning sub-agents to handle complex, multi-step tasks.
//! It requires an [`AgentRuntime`] to be injected for real execution; otherwise,
//! it operates in no-op mode.
//!
//! # Example
//!
//! ```rust,no_run
//! use sombrax_agentic_core::tools::{ToolContext, TaskTool};
//! use sombrax_agentic_core::tools::agent::{AgentRuntime, SubAgentRequest, SubAgentResponse};
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! // With a real runtime
//! # struct MyRuntime;
//! # #[async_trait::async_trait]
//! # impl AgentRuntime for MyRuntime {
//! #     async fn spawn_subagent(&self, _: SubAgentRequest) -> Result<SubAgentResponse, String> { todo!() }
//! # }
//! let context = ToolContext::new("session".into(), PathBuf::from("."));
//! let runtime: Arc<dyn AgentRuntime> = Arc::new(MyRuntime);
//! let task_tool = TaskTool::new(context).with_runtime(runtime);
//! ```

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::instrument;

use super::runtime::{no_op_runtime, AgentRuntime, SubAgentRequest};
use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Spawn sub-agents for complex tasks
#[derive(Clone)]
pub struct TaskTool {
    context: ToolContext,
    runtime: Arc<dyn AgentRuntime>,
}

impl TaskTool {
    /// Create a new task tool with no-op runtime
    ///
    /// The tool will operate in no-op mode until a runtime is injected via
    /// [`with_runtime`](Self::with_runtime).
    pub fn new(context: ToolContext) -> Self {
        Self {
            context,
            runtime: no_op_runtime(),
        }
    }

    /// Set the agent runtime for sub-agent execution
    ///
    /// This enables real sub-agent spawning. Without a runtime, the tool
    /// returns an error indicating that execution is not available.
    pub fn with_runtime(mut self, runtime: Arc<dyn AgentRuntime>) -> Self {
        self.runtime = runtime;
        self
    }

    /// Set maximum recursion depth
    ///
    /// Note: The actual max depth is determined by the runtime.
    /// This method is kept for API compatibility.
    #[deprecated(note = "Max depth is now controlled by the AgentRuntime")]
    pub fn with_max_depth(self, _depth: usize) -> Self {
        self
    }

    /// Check if a runtime is configured (not no-op)
    pub fn has_runtime(&self) -> bool {
        // Check if the runtime supports any subagent type
        // NoOpRuntime returns false for all types
        self.runtime.supports_subagent_type("general-purpose")
    }
}

/// Arguments for the task tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TaskArgs {
    /// Short description (3-5 words)
    pub description: String,
    /// Detailed prompt for the sub-agent
    pub prompt: String,
    /// Type of sub-agent to spawn
    pub subagent_type: String,
    /// Override model (optional, inherits from parent)
    #[serde(default)]
    pub model: Option<String>,
}

/// Output of the task tool
#[derive(Debug, Serialize)]
pub struct TaskOutput {
    /// Child session ID
    pub child_session_id: String,
    /// Sub-agent type used
    pub subagent_type: String,
    /// Task description
    pub description: String,
    /// Response from sub-agent
    pub response: String,
    /// Number of conversation turns (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_turns: Option<usize>,
}

impl Tool for TaskTool {
    const NAME: &'static str = "task";
    type Args = TaskArgs;
    type Output = TaskOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(TaskArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Launch a sub-agent to handle complex, multi-step tasks autonomously.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. Is this task complex enough to need a sub-agent?
2. What type of sub-agent is best suited for this task?
3. What clear instructions should I provide?

## PARAMETERS

- `description` (REQUIRED, STRING): Short description (3-5 words)
  CORRECT: "Analyze security vulnerabilities"
  CORRECT: "Search for API endpoints"
  WRONG: {"description": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `prompt` (REQUIRED, STRING): Detailed instructions for the sub-agent
  Be specific about what the sub-agent should do and return.

- `subagent_type` (REQUIRED, STRING): Type of sub-agent to spawn
  Common types: "general-purpose", "explore", "research"
  The available types depend on the runtime configuration.

- `model` (optional, STRING): Override the model for this sub-agent
  If not specified, inherits from parent agent.

## EXAMPLES

Launch an exploration sub-agent:
  description: "Find auth handlers"
  prompt: "Search the codebase for authentication and authorization handlers. Return a list of files and their purposes."
  subagent_type: "explore"

Launch a research sub-agent:
  description: "Research API patterns"
  prompt: "Analyze the existing API patterns in this codebase and summarize the conventions used."
  subagent_type: "general-purpose"

## WHEN TO USE THIS TOOL

- Complex tasks requiring multiple steps
- Tasks that benefit from focused, autonomous execution
- Parallel exploration of different aspects of a problem

## WHEN NOT TO USE THIS TOOL

- Simple, single-step operations
- Tasks you can do directly with other tools
- When sub-agent runtime is not configured

## COMMON MISTAKES TO AVOID

1. Do NOT pass JSON objects as parameters - use plain strings
2. Do NOT use empty or vague descriptions
3. Do NOT spawn sub-agents for trivial tasks
4. Do NOT exceed maximum recursion depth (sub-agents spawning sub-agents)
"#.to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "task", subagent_type = %args.subagent_type))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Check recursion depth
        let max_depth = self.runtime.max_depth();
        if self.context.current_depth() >= max_depth {
            tracing::warn!(
                current_depth = %self.context.current_depth(),
                max_depth = %max_depth,
                "Max recursion depth exceeded"
            );
            return Err(ToolError::MaxRecursionDepth);
        }

        // Validate arguments
        if args.description.trim().is_empty() {
            return Err(ToolError::Validation("description cannot be empty".into()));
        }
        if args.prompt.trim().is_empty() {
            return Err(ToolError::Validation("prompt cannot be empty".into()));
        }
        if args.subagent_type.trim().is_empty() {
            return Err(ToolError::Validation(
                "subagent_type cannot be empty".into(),
            ));
        }

        // Validate subagent type is supported
        if !self.runtime.supports_subagent_type(&args.subagent_type) {
            // Log warning for unknown types but allow them through
            // The runtime will handle actual validation
            tracing::warn!(
                subagent_type = %args.subagent_type,
                "Unknown or unsupported subagent type"
            );
        }

        // Build the request
        let request = SubAgentRequest {
            description: args.description.clone(),
            prompt: args.prompt,
            subagent_type: args.subagent_type.clone(),
            model: args.model,
            parent_session_id: self.context.session_id().to_string(),
            current_depth: self.context.current_depth(),
            initial_messages: self.context.initial_messages().to_vec(),
        };

        // Delegate to runtime
        tracing::info!(
            description = %args.description,
            subagent_type = %args.subagent_type,
            "Spawning sub-agent via runtime"
        );

        let response = self
            .runtime
            .spawn_subagent(request)
            .await
            .map_err(|e| ToolError::Validation(format!("Sub-agent execution failed: {}", e)))?;

        tracing::info!(
            child_session_id = %response.child_session_id,
            response_len = %response.response.len(),
            "Sub-agent completed"
        );

        Ok(TaskOutput {
            child_session_id: response.child_session_id,
            subagent_type: args.subagent_type,
            description: args.description,
            response: response.response,
            conversation_turns: response.conversation_turns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::agent::runtime::SubAgentResponse;
    use std::path::PathBuf;

    struct MockRuntime {
        response: String,
    }

    #[async_trait::async_trait]
    impl AgentRuntime for MockRuntime {
        async fn spawn_subagent(
            &self,
            request: SubAgentRequest,
        ) -> Result<SubAgentResponse, String> {
            Ok(SubAgentResponse {
                child_session_id: format!("{}-child-mock", request.parent_session_id),
                response: self.response.clone(),
                conversation_turns: Some(1),
            })
        }
    }

    #[tokio::test]
    async fn test_task_tool_with_runtime() {
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let runtime: Arc<dyn AgentRuntime> = Arc::new(MockRuntime {
            response: "Task completed successfully".into(),
        });
        let tool = TaskTool::new(context).with_runtime(runtime);

        assert!(tool.has_runtime());

        let args = TaskArgs {
            description: "Test task".into(),
            prompt: "Do something useful".into(),
            subagent_type: "general-purpose".into(),
            model: None,
        };

        let result = tool.call(args).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.child_session_id, "test-session-child-mock");
        assert_eq!(output.response, "Task completed successfully");
        assert_eq!(output.conversation_turns, Some(1));
    }

    #[tokio::test]
    async fn test_task_tool_without_runtime() {
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let tool = TaskTool::new(context);

        assert!(!tool.has_runtime());

        let args = TaskArgs {
            description: "Test task".into(),
            prompt: "Do something".into(),
            subagent_type: "general-purpose".into(),
            model: None,
        };

        let result = tool.call(args).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("No AgentRuntime is configured"));
    }

    #[tokio::test]
    async fn test_task_tool_validates_empty_fields() {
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let runtime: Arc<dyn AgentRuntime> = Arc::new(MockRuntime {
            response: "ok".into(),
        });
        let tool = TaskTool::new(context).with_runtime(runtime);

        // Empty description
        let args = TaskArgs {
            description: "".into(),
            prompt: "Do something".into(),
            subagent_type: "general-purpose".into(),
            model: None,
        };
        assert!(tool.call(args).await.is_err());

        // Empty prompt
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let runtime: Arc<dyn AgentRuntime> = Arc::new(MockRuntime {
            response: "ok".into(),
        });
        let tool = TaskTool::new(context).with_runtime(runtime);
        let args = TaskArgs {
            description: "Test".into(),
            prompt: "".into(),
            subagent_type: "general-purpose".into(),
            model: None,
        };
        assert!(tool.call(args).await.is_err());

        // Empty subagent_type
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let runtime: Arc<dyn AgentRuntime> = Arc::new(MockRuntime {
            response: "ok".into(),
        });
        let tool = TaskTool::new(context).with_runtime(runtime);
        let args = TaskArgs {
            description: "Test".into(),
            prompt: "Do something".into(),
            subagent_type: "".into(),
            model: None,
        };
        assert!(tool.call(args).await.is_err());
    }

    #[tokio::test]
    async fn test_definition() {
        let context = ToolContext::new("test-session".into(), PathBuf::from("."));
        let tool = TaskTool::new(context);

        let def = tool.definition("test prompt".into()).await;
        assert_eq!(def.name, "task");
        assert!(def.description.contains("sub-agent"));
    }

    #[tokio::test]
    async fn test_task_tool_passes_initial_messages() {
        use crate::message::{AssistantContent, Message};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Runtime that captures the initial_messages count
        struct CapturingRuntime {
            message_count: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl AgentRuntime for CapturingRuntime {
            async fn spawn_subagent(
                &self,
                request: SubAgentRequest,
            ) -> Result<SubAgentResponse, String> {
                self.message_count
                    .store(request.initial_messages.len(), Ordering::SeqCst);
                Ok(SubAgentResponse {
                    child_session_id: format!("{}-child", request.parent_session_id),
                    response: "done".into(),
                    conversation_turns: Some(1),
                })
            }
        }

        // Create context with initial messages
        let initial_msgs = vec![
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::Text {
                    text: "Source code context".into(),
                }],
                reasoning: None,
            },
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::Text {
                    text: "More context".into(),
                }],
                reasoning: None,
            },
        ];

        let context = ToolContext::new("test-session".into(), PathBuf::from("."))
            .with_initial_messages(initial_msgs);

        let message_count = Arc::new(AtomicUsize::new(0));
        let runtime: Arc<dyn AgentRuntime> = Arc::new(CapturingRuntime {
            message_count: message_count.clone(),
        });
        let tool = TaskTool::new(context).with_runtime(runtime);

        let args = TaskArgs {
            description: "Test".into(),
            prompt: "Do task".into(),
            subagent_type: "general-purpose".into(),
            model: None,
        };

        let result = tool.call(args).await;
        assert!(result.is_ok());
        assert_eq!(message_count.load(Ordering::SeqCst), 2);
    }
}
