//! # Agent Tools
//!
//! Agent tools for file operations, shell commands, HTTP requests, and task coordination.
//!
//! This module provides a comprehensive set of tools that can be used by LLM agents
//! to interact with the file system, execute commands, make HTTP requests, and
//! coordinate sub-agents.
//!
//! ## Tool Categories
//!
//! - **File Tools**: `ReadTool`, `WriteTool`, `EditTool`, `GlobTool`, `GrepTool`
//! - **Shell Tools**: `BashTool` (with safety validation to block dangerous commands)
//! - **Web Tools**: `FetchTool` (HTTP client supporting all methods)
//! - **Agent Tools**: `TaskTool` (sub-agent spawning), `TodoReadTool`, `TodoWriteTool`
//!
//! ## Key Features
//!
//! - **Workspace Boundary Enforcement**: All file operations are confined to the workspace directory
//! - **Command Safety**: Dangerous shell commands (rm -rf /, fork bombs, etc.) are automatically blocked
//! - **Session Isolation**: Todos and context are scoped to sessions
//! - **Recursion Limits**: Task tool enforces depth limits to prevent infinite loops
//! - **Timeout Support**: Shell and HTTP operations support configurable timeouts
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use sombrax_agentic_core::tools::{ToolContext, create_tool_set};
//! use std::path::PathBuf;
//!
//! // Create a context bound to a workspace directory
//! let context = ToolContext::new(
//!     "session-123".to_string(),
//!     PathBuf::from("/path/to/workspace"),
//! );
//!
//! // Create a registry with all tools
//! let tools = create_tool_set(context);
//! ```
//!
//! ## Using Individual Tools
//!
//! ```rust,no_run
//! use sombrax_agentic_core::tools::{ToolContext, ReadTool, BashTool};
//! use sombrax_agentic_core::tools::file::ReadArgs;
//! use sombrax_agentic_core::tools::shell::BashArgs;
//! use sombrax_agentic_core::tools::registry::Tool;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let context = ToolContext::new("session".to_string(), PathBuf::from("."));
//!
//! // Read a file
//! let read_tool = ReadTool::new(context.clone());
//! let output = read_tool.call(ReadArgs {
//!     file_path: "Cargo.toml".to_string(),
//!     description: None,
//!     offset: None,
//!     limit: None,
//! }).await?;
//! println!("Lines: {}", output.lines_read);
//!
//! // Execute a safe command
//! let bash_tool = BashTool::new(context);
//! let result = bash_tool.call(BashArgs {
//!     command: "ls -la".to_string(),
//!     timeout: Some(30_000),
//!     description: Some("List files".to_string()),
//! }).await?;
//! println!("Exit code: {}", result.exit_code);
//! # Ok(())
//! # }
//! ```
//!
//! ## Sub-Agent Execution with AgentRuntime
//!
//! The TaskTool requires an [`AgentRuntime`](agent::AgentRuntime) to spawn sub-agents.
//! Without a configured runtime, the tool returns an error.
//!
//! ```rust,no_run
//! use sombrax_agentic_core::tools::{ToolContext, TaskTool, create_tool_set_with_runtime};
//! use sombrax_agentic_core::tools::agent::{AgentRuntime, SubAgentRequest, SubAgentResponse};
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! // Implement AgentRuntime for your orchestrator
//! struct MyRuntime;
//!
//! #[async_trait::async_trait]
//! impl AgentRuntime for MyRuntime {
//!     async fn spawn_subagent(&self, request: SubAgentRequest) -> Result<SubAgentResponse, String> {
//!         // Delegate to your agent registry/orchestrator
//!         todo!()
//!     }
//! }
//!
//! // Create tool set with runtime
//! let context = ToolContext::new("session".into(), PathBuf::from("."));
//! let runtime: Arc<dyn AgentRuntime> = Arc::new(MyRuntime);
//! let tools = create_tool_set_with_runtime(context, runtime);
//! ```
//!
//! ## Architecture
//!
//! All tools implement the [`registry::Tool`] trait which provides:
//! - `definition()` - Returns JSON Schema for the tool's parameters
//! - `call()` - Executes the tool with provided arguments
//!
//! The [`ToolContext`] provides:
//! - Workspace boundary enforcement
//! - Session-scoped todo storage
//! - Child context creation for sub-agents
//! - Recursion depth tracking

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub mod context;
pub mod error;
pub mod registry;
pub mod resolver;
pub mod serde_flexible;

/// Wrapper to convert `tools::registry::Tool` to `tool::ToolDyn`.
///
/// This bridges the two tool systems in sac:
/// - `sombrax_agentic_core::tools::registry::Tool` - what built-in tools implement
/// - `sombrax_agentic_core::tool::ToolDyn` - what Agent uses for execution
struct ToolBridge<T: registry::Tool>(T);

impl<T: registry::Tool> crate::tool::ToolDyn for ToolBridge<T> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn definition<'a>(
        &'a self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = crate::tool::ToolDefinition> + Send + 'a>> {
        Box::pin(async move {
            let def = self.0.definition(prompt).await;
            crate::tool::ToolDefinition {
                name: def.name,
                description: def.description,
                parameters: def.parameters,
            }
        })
    }

    fn call<'a>(
        &'a self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, crate::error::ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let parsed_args: T::Args = serde_json::from_str(&args).map_err(|e| {
                crate::error::ToolError::ExecutionFailed(format!("Invalid arguments: {}", e))
            })?;
            let result = self
                .0
                .call(parsed_args)
                .await
                .map_err(|e| crate::error::ToolError::ExecutionFailed(e.to_string()))?;
            serde_json::to_string(&result).map_err(|e| {
                crate::error::ToolError::ExecutionFailed(format!(
                    "Failed to serialize output: {}",
                    e
                ))
            })
        })
    }
}

/// Convert a registry Tool to an Arc<dyn ToolDyn> for use with Agent.
///
/// This allows using any tool that implements `sombrax_agentic_core::tools::registry::Tool`
/// with the Agent's tool execution system.
///
/// # Example
///
/// ```rust,no_run
/// use sombrax_agentic_core::tools::{ToolContext, ReadTool, into_tool_dyn};
/// use std::path::PathBuf;
///
/// let context = ToolContext::new("session".to_string(), PathBuf::from("."));
/// let read_tool = ReadTool::new(context);
/// let tool_dyn = into_tool_dyn(read_tool);
/// // Now tool_dyn can be used with Agent::tools()
/// ```
pub fn into_tool_dyn<T: registry::Tool>(tool: T) -> Arc<dyn crate::tool::ToolDyn> {
    Arc::new(ToolBridge(tool))
}

pub mod agent;
pub mod file;
pub mod shell;
/// Skill tool for executing discovered skills.
pub mod skill;
pub mod web;

// Re-export main types
pub use context::ToolContext;
pub use error::ToolError;
pub use registry::ToolRegistry;
pub use resolver::{FileResolver, MatchType, PathResolution};

// Re-export file tools
pub use file::{EditTool, GlobTool, GrepTool, ReadTool, WriteTool};

// Re-export shell tools
pub use shell::BashTool;

// Re-export web tools
pub use web::FetchTool;

// Re-export agent tools
pub use agent::{TaskTool, TodoReadTool, TodoWriteTool};

// Re-export agent runtime types
pub use agent::{AgentRuntime, NoOpRuntime, SubAgentRequest, SubAgentResponse};

// Re-export skill tool
pub use skill::SkillTool;

/// Create a standard tool set with all available tools
///
/// Note: The TaskTool will operate in no-op mode without an AgentRuntime.
/// Use [`create_tool_set_with_runtime`] for full sub-agent support.
pub fn create_tool_set(context: ToolContext) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // File tools
    registry.register(ReadTool::new(context.clone()));
    registry.register(WriteTool::new(context.clone()));
    registry.register(EditTool::new(context.clone()));
    registry.register(GlobTool::new(context.clone()));
    registry.register(GrepTool::new(context.clone()));

    // Shell tools
    registry.register(BashTool::new(context.clone()));

    // Web tools
    registry.register(FetchTool::new(context.clone()));

    // Agent tools (TaskTool without runtime - no-op mode)
    registry.register(TaskTool::new(context.clone()));
    registry.register(TodoReadTool::new(context.clone()));
    registry.register(TodoWriteTool::new(context));

    registry
}

/// Create a standard tool set with an AgentRuntime for sub-agent execution
///
/// This function creates a complete tool set where TaskTool is wired to the
/// provided runtime for real sub-agent spawning.
///
/// # Arguments
/// * `context` - The tool execution context
/// * `runtime` - The agent runtime for spawning sub-agents
///
/// # Example
///
/// ```rust,no_run
/// use sombrax_agentic_core::tools::{ToolContext, create_tool_set_with_runtime};
/// use sombrax_agentic_core::tools::agent::{AgentRuntime, SubAgentRequest, SubAgentResponse};
/// use std::sync::Arc;
/// use std::path::PathBuf;
///
/// # struct MyRuntime;
/// # #[async_trait::async_trait]
/// # impl AgentRuntime for MyRuntime {
/// #     async fn spawn_subagent(&self, _: SubAgentRequest) -> Result<SubAgentResponse, String> { todo!() }
/// # }
///
/// let context = ToolContext::new("session".into(), PathBuf::from("."));
/// let runtime: Arc<dyn AgentRuntime> = Arc::new(MyRuntime);
/// let tools = create_tool_set_with_runtime(context, runtime);
/// ```
pub fn create_tool_set_with_runtime(
    context: ToolContext,
    runtime: Arc<dyn agent::AgentRuntime>,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // File tools
    registry.register(ReadTool::new(context.clone()));
    registry.register(WriteTool::new(context.clone()));
    registry.register(EditTool::new(context.clone()));
    registry.register(GlobTool::new(context.clone()));
    registry.register(GrepTool::new(context.clone()));

    // Shell tools
    registry.register(BashTool::new(context.clone()));

    // Web tools
    registry.register(FetchTool::new(context.clone()));

    // Agent tools (TaskTool with runtime for real execution)
    registry.register(TaskTool::new(context.clone()).with_runtime(runtime));
    registry.register(TodoReadTool::new(context.clone()));
    registry.register(TodoWriteTool::new(context));

    registry
}

/// Create a minimal tool set with file tools only
pub fn create_file_tool_set(context: ToolContext) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    registry.register(ReadTool::new(context.clone()));
    registry.register(WriteTool::new(context.clone()));
    registry.register(EditTool::new(context.clone()));
    registry.register(GlobTool::new(context.clone()));
    registry.register(GrepTool::new(context));

    registry
}
