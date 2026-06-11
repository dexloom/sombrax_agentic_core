//! Agent coordination tools
//!
//! Tools for spawning sub-agents and tracking task progress.
//!
//! ## AgentRuntime
//!
//! The [`AgentRuntime`] trait enables the TaskTool to delegate sub-agent execution
//! to a real agent runtime. Without an injected runtime, TaskTool operates in a
//! no-op mode that returns a placeholder response.
//!
//! ```rust,no_run
//! use sombrax_agentic_core::tools::agent::{AgentRuntime, SubAgentRequest, SubAgentResponse};
//! use sombrax_agentic_core::tools::{ToolContext, TaskTool};
//! use std::sync::Arc;
//!
//! // Your runtime implementation
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
//! // Wire up the TaskTool
//! let context = ToolContext::new("session".into(), std::path::PathBuf::from("."));
//! let runtime: Arc<dyn AgentRuntime> = Arc::new(MyRuntime);
//! let task_tool = TaskTool::new(context).with_runtime(runtime);
//! ```

mod runtime;
mod task;
mod todo;

pub use runtime::{AgentRuntime, NoOpRuntime, SubAgentRequest, SubAgentResponse};
pub use task::{TaskArgs, TaskOutput, TaskTool};
pub use todo::{
    TodoItem, TodoReadArgs, TodoReadOutput, TodoReadTool, TodoWriteArgs, TodoWriteItem,
    TodoWriteOutput, TodoWriteTool,
};
