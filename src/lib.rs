//! # SombraX Agentic Core (`sombrax_agentic_core`)
//!
//! SombraX Agentic Core (SAC) is a Rust library for LLM agent orchestration with content-modifying hooks,
//! MCP tool integration, cross-agent communication, and context optimization.
//!
//! ## Features
//!
//! - **Content-Modifying Hooks**: Intercept and modify messages before/after LLM calls
//! - **MCP Tool Integration**: Connect to MCP servers for tool discovery and execution
//! - **Cross-Agent Communication**: Registry for agent discovery and invocation
//! - **Context Optimization**: Automatic context management for long conversations
//! - **OpenTelemetry**: Built-in observability with tracing and metrics
//! - **LLM Providers**: OpenAI, Anthropic, ZAI, Cerebras, OpenRouter (`sombrax_agentic_core::providers`)
//! - **Agent Tools**: File, shell, web, and task tools (`sombrax_agentic_core::tools`)
//!
//! ## Quick Start
//!
//! ```ignore
//! use sombrax_agentic_core::prelude::*;
//!
//! #[derive(Clone)]
//! struct LoggingHook;
//!
//! impl Hook for LoggingHook {
//!     async fn pre_completion(
//!         &self,
//!         message: Message,
//!         _history: &[Message],
//!         ctx: &mut HookContext,
//!     ) -> HookResult<Message> {
//!         println!("[{}] Processing message", ctx.request_id);
//!         Ok(message)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = OpenAIClientBuilder::new(&std::env::var("OPENAI_API_KEY")?).build();
//!     let model = client.completion_model_adapter("gpt-4");
//!
//!     let agent = AgentBuilder::new(model)
//!         .preamble("You are a helpful assistant.")
//!         .hook(LoggingHook)
//!         .build();
//!
//!     let response = agent.prompt("Hello!").await?;
//!     println!("{}", response.content());
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod agent;
pub mod context;
pub mod error;
/// Experience-based learning system for agent self-improvement.
pub mod experience;
pub mod extractor;
pub mod hook;
pub mod message;
pub mod prelude;
/// System prompts as first-class on-disk assets with a name-based
/// resolution ladder — analog of [`skill`] for persona prompts.
pub mod prompt;
pub mod provider;
pub mod providers;
pub mod retry;
/// Skill system for discovering and loading user-defined agent skills.
pub mod skill;
pub mod telemetry;
pub mod tool;
pub mod tools;

/// Pluggable pipeline / bundle / job runtime.
///
/// See [`runs`] (and `src/runs/mod.rs`) for the design and contract.
#[cfg(feature = "runs")]
pub mod runs;

// Re-export commonly used types at crate root
pub use agent::{Agent, AgentBuilder, AgentWrapper, ExecutionStats, PromptResponse};
pub use context::{HookContext, OptimizationConfig, SharedContext};
pub use error::{CompletionError, HookError, HookStage, ToolError};
pub use extractor::{build_extractor, ExtractorBuildError, ExtractorError, ExtractorWrapper};
pub use hook::builtin::{ValidationHook, WorkspaceBoundaryHook};
pub use hook::{Hook, HookChain, ToolCallDecision};
pub use message::{validate_tool_result_ids, Message, ValidationError};
pub use provider::{CompletionModel, CompletionRequest, CompletionResponse, Usage};
pub use providers::{
    build_agent, build_agent_with_options, AgentBuildError, AgentBuildOptions, LlmConfigLike,
    ProviderType, ProviderTypeError,
};
pub use retry::{ResponseValidation, RetryConfig, ValidationResult};
pub use tool::{McpToolSource, StdioMcpClient, Tool, ToolDefinition, ToolDyn};
