//! Prelude module for convenient imports
//!
//! Import all commonly used types with a single use statement:
//!
//! ```ignore
//! use sombrax_agentic_core::prelude::*;
//! ```

// Core types
pub use crate::agent::{Agent, AgentBuilder, AgentWrapper};
pub use crate::message::{
    validate_tool_result_ids, AssistantContent, Message, ToolCall, ToolCallFunction, UserContent,
    ValidationError,
};

// Hook types
pub use crate::context::{
    CancelSignal, ContextOptimizer, HookContext, OptimizationConfig, RecencyOptimizer,
    SharedContext,
};
pub use crate::hook::builtin::{LoggingHook, PrefixHook, SuffixHook, ValidationHook};
pub use crate::hook::{Hook, HookChain, HookDyn, ToolCallDecision};

// Provider trait and types
pub use crate::provider::{
    CompletionModel, CompletionRequest, CompletionResponse, MapObfuscator,
    ObfuscatingCompletionModel, Obfuscator, Usage,
};

// Provider implementations
pub use crate::providers::{
    AnthropicClient, AnthropicClientBuilder, AnthropicClientExt, AnthropicCompletionModelAdapter,
    OpenAIClient, OpenAIClientBuilder, OpenAIClientExt, OpenAICompletionModelAdapter,
};

// Provider type and builder
pub use crate::providers::{
    build_agent, build_agent_with_options, AgentBuildError, AgentBuildOptions, LlmConfigLike,
    ProviderType, ProviderTypeError,
};

// Tool types
pub use crate::tool::{Tool, ToolDefinition, ToolDyn};

// Error types
pub use crate::error::{
    CompletionError, HookError, HookResult, HookStage, RegistryError, ToolError,
};

// Re-export common dependencies
pub use serde::{Deserialize, Serialize};
pub use serde_json::{self, Value as JsonValue};
