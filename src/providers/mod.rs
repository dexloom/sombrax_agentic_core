//! # LLM Providers
//!
//! LLM provider clients for OpenAI, Anthropic (Claude), MiniMax, ZAI, Cerebras, OpenRouter, MLX-LM, and LMStudio.
//!
//! This module provides unified access to multiple LLM providers through
//! a common interface. Each provider has a client builder for configuration
//! and a completion model for making requests.
//!
//! ## Providers
//!
//! | Provider | Models | Key Feature |
//! |----------|--------|-------------|
//! | **OpenAI** | GPT-4, GPT-4o, GPT-3.5 | Industry standard, function calling |
//! | **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | Long context, tool use |
//! | **MiniMax** | MiniMax M2.x | Anthropic-compatible API, interleaved thinking |
//! | **ZAI** | GLM-4.6 | Thinking mode for extended reasoning |
//! | **Cerebras** | Llama models | Fast inference, string-only tool results |
//! | **OpenRouter** | Multi-model | Provider routing (whitelist/blacklist) |
//! | **MLX-LM** | Local MLX models | Apple Silicon, custom chat templates (Minimax) |
//! | **LMStudio** | Local models | Anti-repetition controls, server-side templates |
//! | **Ollama** | Local + cloud models | Native `/api/chat`, thinking traces, one provider for localhost and ollama.com |
//!
//! ## Environment Variables
//!
//! | Variable | Provider | Required |
//! |----------|----------|----------|
//! | `OPENAI_API_KEY` | OpenAI | Yes |
//! | `ANTHROPIC_API_KEY` | Anthropic | Yes |
//! | `MINIMAX_API_KEY` | MiniMax | Yes |
//! | `ZAI_API_KEY` | ZAI | Yes |
//! | `CEREBRAS_API_KEY` | Cerebras | Yes |
//! | `OPENROUTER_API_KEY` | OpenRouter | Yes |
//! | N/A | MLX-LM | No (local server) |
//! | N/A | LMStudio | No (local server) |
//! | `OLLAMA_API_KEY` | Ollama | No for local; yes for cloud (ollama.com) |
//!
//! ## Ollama (native, cloud + local)
//!
//! One provider serves both. Local needs no key; cloud points at
//! `https://ollama.com` with an API key.
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::OllamaClientBuilder;
//!
//! // Local
//! let local = OllamaClientBuilder::new()
//!     .base_url("http://localhost:11434")
//!     .build();
//! let model = local.completion_model("llama3.2");
//!
//! // Cloud
//! let cloud = OllamaClientBuilder::new()
//!     .base_url("https://ollama.com")
//!     .api_key("your-ollama-api-key")
//!     .enable_thinking(true)
//!     .build();
//! let model = cloud.completion_model("gpt-oss:120b");
//! ```
//!
//! ## Quick Start
//!
//! ### OpenAI
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::OpenAIClientBuilder;
//!
//! let client = OpenAIClientBuilder::new("your-api-key")
//!     .temperature(0.7)
//!     .build();
//!
//! let model = client.completion_model("gpt-4o");
//! ```
//!
//! ### Anthropic (Claude)
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::AnthropicClientBuilder;
//!
//! let client = AnthropicClientBuilder::new("your-api-key")
//!     .temperature(0.7)
//!     .max_tokens(4096)
//!     .build();
//!
//! let model = client.completion_model("claude-3-5-sonnet-20241022");
//! ```
//!
//! ### MiniMax (M2.1)
//!
//! MiniMax uses an Anthropic-compatible API and supports interleaved thinking blocks.
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::MinimaxClientBuilder;
//!
//! let client = MinimaxClientBuilder::new("your-api-key")
//!     .temperature(0.7)
//!     .top_p(0.95)
//!     .max_tokens(4096)
//!     .build();
//!
//! let model = client.completion_model("minimax-m2.1");
//! ```
//!
//! For local development with MLX-LM server (OpenAI-compatible):
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::{MlxLmClientBuilder, ChatTemplate};
//!
//! let client = MlxLmClientBuilder::new()
//!     .base_url("http://localhost:1234/v1")
//!     .chat_template(ChatTemplate::Minimax)
//!     .temperature(0.4)
//!     .top_p(0.95)
//!     .top_k(20)
//!     .build();
//!
//! let model = client.completion_model("MiniMax-M2.1-Q8");
//! ```
//!
//! ### ZAI with Thinking Mode
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::ZaiClientBuilder;
//!
//! let client = ZaiClientBuilder::new("api-key")
//!     .enable_thinking(true)
//!     .build();
//!
//! let model = client.completion_model("glm-4.6");
//! ```
//!
//! ### OpenRouter with Provider Routing
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::OpenRouterClientBuilder;
//!
//! let client = OpenRouterClientBuilder::new("api-key")
//!     .whitelist(vec!["anthropic".to_string()])
//!     .allow_fallbacks(false)
//!     .build();
//! ```
//!
//! ## Error Handling
//!
//! All providers use [`ProviderError`] for consistent error handling:
//!
//! - `ProviderError::Authentication` - Invalid API key
//! - `ProviderError::RateLimited` - Rate limit exceeded (with retry hint)
//! - `ProviderError::Http` - HTTP errors with status code
//! - `ProviderError::InvalidResponse` - Malformed response from provider
//!
//! ## Agent Integration
//!
//! All providers can be used with the `Agent` framework via adapters:
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::{ZaiClientBuilder, ZaiClientExt};
//! use sombrax_agentic_core::AgentBuilder;
//!
//! let client = ZaiClientBuilder::new("api-key").build();
//! // Get an adapter that implements CompletionModel
//! let model = client.completion_model_adapter("glm-4.6");
//!
//! // Use with Agent framework
//! let agent = AgentBuilder::new(model).build();
//! ```

pub mod adapter;
pub mod builder;
pub mod config;
pub mod error;
mod http;
pub mod provider_type;

pub mod anthropic;
pub mod cerebras;
pub mod lmstudio;
pub mod minimax;
pub mod mlxlm;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod zai;

// Re-export main types
pub use error::ProviderError;

// Re-export OpenAI
pub use openai::{OpenAIClient, OpenAIClientBuilder, OpenAICompletionModel};

// Re-export Anthropic
pub use anthropic::{AnthropicClient, AnthropicClientBuilder, AnthropicCompletionModel};

// Re-export MiniMax
pub use minimax::{MinimaxClient, MinimaxClientBuilder, MinimaxCompletionModel};

// Re-export ZAI
pub use zai::{ZaiClient, ZaiClientBuilder, ZaiCompletionModel};

// Re-export Cerebras
pub use cerebras::{CerebrasClient, CerebrasClientBuilder, CerebrasCompletionModel};

// Re-export OpenRouter
pub use openrouter::{OpenRouterClient, OpenRouterClientBuilder, OpenRouterCompletionModel};

// Re-export LMStudio
pub use lmstudio::{LmStudioClient, LmStudioClientBuilder, LmStudioCompletionModel};

// Re-export Ollama (native /api/chat — cloud + local)
pub use ollama::{OllamaClient, OllamaClientBuilder, OllamaCompletionModel};

// Re-export MLX-LM
pub use mlxlm::{ChatTemplate, MlxLmClient, MlxLmClientBuilder, MlxLmCompletionModel};

// Re-export adapters for CompletionModel trait support
pub use adapter::{
    AnthropicClientExt, AnthropicCompletionModelAdapter, CerebrasClientExt,
    CerebrasCompletionModelAdapter, LmStudioClientExt, LmStudioCompletionModelAdapter,
    MinimaxClientExt, MinimaxCompletionModelAdapter, MlxLmClientExt, MlxLmCompletionModelAdapter,
    OllamaClientExt, OllamaCompletionModelAdapter, OpenAIClientExt, OpenAICompletionModelAdapter,
    OpenRouterClientExt, OpenRouterCompletionModelAdapter, ZaiClientExt, ZaiCompletionModelAdapter,
};

// Re-export provider type and config
pub use config::LlmConfigLike;
pub use provider_type::{ProviderType, ProviderTypeError};

// Re-export builder functions
pub use builder::{build_agent, build_agent_with_options, AgentBuildError, AgentBuildOptions};
