//! LLM configuration trait for provider-agnostic agent building.
//!
//! This module defines the [`LlmConfigLike`] trait that allows any configuration
//! struct to be used with the agent factory functions. This decouples sac from
//! specific configuration crate dependencies.
//!
//! # Example
//!
//! ```rust,ignore
//! use sombrax_agentic_core::providers::{LlmConfigLike, build_agent};
//!
//! struct MyConfig {
//!     provider: String,
//!     url: String,
//!     model: String,
//!     api_key: Option<String>,
//! }
//!
//! impl LlmConfigLike for MyConfig {
//!     fn provider(&self) -> &str { &self.provider }
//!     fn url(&self) -> &str { &self.url }
//!     fn model(&self) -> &str { &self.model }
//!     fn api_key(&self) -> Option<&str> { self.api_key.as_deref() }
//! }
//!
//! let config = MyConfig { /* ... */ };
//! let agent = build_agent(&config, "system prompt", 8192, vec![]).await?;
//! ```

/// Configuration trait for LLM providers.
///
/// Implement this trait for your configuration struct to use the factory functions.
/// This allows `sac` to remain independent of specific config crate dependencies.
///
/// Required methods provide the essential fields for all providers.
/// Optional methods have default implementations returning `None` and are used
/// by specific providers that support additional features.
pub trait LlmConfigLike {
    /// Provider type string (e.g., "openrouter", "openai", "anthropic", "minimax", "cerebras", "zai", "mlx")
    fn provider(&self) -> &str;

    /// Base URL for the API endpoint
    fn url(&self) -> &str;

    /// Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet", "glm-4.6")
    fn model(&self) -> &str;

    /// API key for authentication (None for local providers like MLX-LM)
    fn api_key(&self) -> Option<&str>;

    /// Sampling temperature (0.0 - 1.0)
    /// Controls randomness: lower values are more deterministic.
    fn temperature(&self) -> Option<f64> {
        None
    }

    /// Top-p (nucleus) sampling (0.0 - 1.0)
    /// Only tokens with cumulative probability <= top_p are considered.
    fn top_p(&self) -> Option<f64> {
        None
    }

    /// Top-k sampling
    /// Limits to the k most likely tokens.
    fn top_k(&self) -> Option<u64> {
        None
    }

    /// Enable thinking/reasoning mode (for ZAI, some Claude models)
    fn thinking(&self) -> Option<bool> {
        None
    }

    /// Thinking budget in tokens for providers that support it (Anthropic, MiniMax, ZAI)
    /// When set with thinking enabled, uses this as the thinking budget.
    fn thinking_budget_tokens(&self) -> Option<u64> {
        None
    }

    /// Enable explicit prompt-cache breakpoints for providers with an explicit
    /// cache protocol (Anthropic, MiniMax). `None` keeps the provider default
    /// (Anthropic on, MiniMax off). Ignored by implicit-cache providers, which
    /// benefit from an append-only message prefix without any wire markers.
    fn prompt_caching(&self) -> Option<bool> {
        None
    }

    // --- OpenRouter-specific ---

    /// Providers to exclude/blacklist (OpenRouter: provider.ignore)
    fn blacklist(&self) -> Option<&[String]> {
        None
    }

    /// Limit to specific providers only (OpenRouter: provider.only)
    fn whitelist(&self) -> Option<&[String]> {
        None
    }

    /// Allow fallback to other providers (OpenRouter: provider.allow_fallbacks)
    fn allow_fallbacks(&self) -> Option<bool> {
        None
    }

    // --- MLX-LM specific ---

    /// Chat template format for MLX-LM provider
    /// Options: "openai", "minimax", "chatml", "glm"
    fn chat_template(&self) -> Option<&str> {
        None
    }

    /// Repetition penalty for MLX-LM provider (1.0-2.0)
    /// Higher values penalize repeated tokens more strongly.
    fn repetition_penalty(&self) -> Option<f64> {
        None
    }

    // --- LMStudio / MLX-LM shared ---

    /// Repeat penalty for LMStudio provider (1.0-2.0)
    /// Applied to tokens from prompt and output. Primary anti-repetition control.
    fn repeat_penalty(&self) -> Option<f64> {
        None
    }

    /// Repetition context size for LMStudio and MLX-LM providers
    /// How many recent tokens to check for repetition penalty.
    /// -1 = full context, 0 = disabled, positive = last N tokens.
    fn repetition_context_size(&self) -> Option<i64> {
        None
    }

    /// Frequency penalty for LMStudio and MLX-LM providers (-2.0 to 2.0)
    /// Penalizes based on token frequency in output only.
    fn frequency_penalty(&self) -> Option<f64> {
        None
    }

    /// Presence penalty for LMStudio and MLX-LM providers (-2.0 to 2.0)
    /// Penalizes based on token presence in output only.
    fn presence_penalty(&self) -> Option<f64> {
        None
    }

    /// Minimum probability floor for sampling (0.0-1.0)
    /// Tokens below this probability are filtered out.
    /// Supported by LMStudio and MLX-LM providers.
    fn min_p(&self) -> Option<f64> {
        None
    }
}
