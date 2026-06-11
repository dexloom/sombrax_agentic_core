//! Agent factory functions for building agents from configuration.
//!
//! This module provides generic factory functions that can build agents from any
//! configuration implementing [`LlmConfigLike`]. This allows applications to use
//! their own configuration structs while leveraging sac's agent infrastructure.
//!
//! # Example
//!
//! ```rust,ignore
//! use sombrax_agentic_core::providers::{build_agent, build_agent_with_options, AgentBuildOptions};
//!
//! // Simple agent
//! let agent = build_agent(&config, "You are helpful", 8192, vec![]).await?;
//!
//! // Agent with options
//! let options = AgentBuildOptions {
//!     max_turns: Some(25),
//!     hook: Some(my_hook),
//!     response_validation: Some(validation),
//! };
//! let agent = build_agent_with_options(&config, "You are helpful", 8192, vec![], options).await?;
//! ```

use crate::agent::wrapper::AgentWrapper;
use crate::context::{ContextOptimizer, OptimizationConfig};
use crate::hook::Hook;
use crate::provider::CompletionModelExt;
use crate::providers::{
    config::LlmConfigLike,
    provider_type::{ProviderType, ProviderTypeError},
    AnthropicClientBuilder, AnthropicClientExt, CerebrasClientBuilder, CerebrasClientExt,
    ChatTemplate, LmStudioClientBuilder, LmStudioClientExt, MinimaxClientBuilder, MinimaxClientExt,
    MlxLmClientBuilder, MlxLmClientExt, OllamaClientBuilder, OllamaClientExt, OpenAIClientBuilder,
    OpenAIClientExt, OpenRouterClientBuilder, OpenRouterClientExt, ZaiClientBuilder, ZaiClientExt,
};
use crate::retry::ResponseValidation;
use crate::tool::ToolDyn;
use crate::AgentBuilder;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info};

/// Error type for agent building operations.
#[derive(Error, Debug)]
pub enum AgentBuildError {
    /// Provider type error (unknown provider)
    #[error("Provider error: {0}")]
    ProviderType(#[from] ProviderTypeError),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Options for building an agent with hooks and execution limits.
///
/// This is a generic version that accepts any [`Hook`] implementation.
/// All fields are optional and have sensible defaults.
pub struct AgentBuildOptions<H: Hook + Clone = ()> {
    /// Maximum number of turns for the tool execution loop.
    /// When set, the agent will stop executing tools after this many turns.
    pub max_turns: Option<usize>,

    /// Optional hook for tracking, display, and notifications.
    /// Hooks can intercept and modify messages, tool calls, and responses.
    pub hook: Option<H>,

    /// Optional response validation for retrying on empty/low-quality responses.
    pub response_validation: Option<ResponseValidation>,

    /// Optional context optimizer for managing conversation context.
    /// Useful for reducing context size during long conversations.
    pub optimizer: Option<Arc<dyn ContextOptimizer>>,

    /// Optional optimization configuration for the context optimizer.
    pub optimization_config: Option<OptimizationConfig>,
}

impl<H: Hook + Clone> Default for AgentBuildOptions<H> {
    fn default() -> Self {
        Self {
            max_turns: None,
            hook: None,
            response_validation: None,
            optimizer: None,
            optimization_config: None,
        }
    }
}

/// Build an agent from any configuration implementing [`LlmConfigLike`].
///
/// This is the primary factory function for creating agents with the correct
/// provider client based on the `provider()` method of the configuration.
///
/// # Arguments
///
/// * `config` - Configuration implementing [`LlmConfigLike`]
/// * `system_prompt` - System prompt/preamble for the agent
/// * `max_tokens` - Maximum tokens for completion
/// * `tools` - Vector of tools to register with the agent
///
/// # Supported Providers
///
/// - `openrouter` - OpenRouter API (default)
/// - `openai` - Direct OpenAI API
/// - `anthropic` / `claude` - Anthropic Claude API
/// - `minimax` - MiniMax Anthropic-compatible API
/// - `cerebras` - Cerebras with custom tool content serialization
/// - `ollama` - Local Ollama (via OpenAI-compatible API)
/// - `zai` - ZAI with GLM-4.6 thinking mode support
/// - `mlx` / `mlxlm` - Local MLX-LM for Apple Silicon
///
/// # Example
///
/// ```rust,ignore
/// let agent = build_agent(&config, "You are a helpful assistant.", 8192, vec![]).await?;
/// let (response, stats) = agent.execute("Hello!", &[]).await?;
/// ```
pub async fn build_agent<C: LlmConfigLike>(
    config: &C,
    system_prompt: &str,
    max_tokens: u64,
    tools: Vec<Arc<dyn ToolDyn>>,
) -> Result<AgentWrapper, AgentBuildError> {
    build_agent_with_options(
        config,
        system_prompt,
        max_tokens,
        tools,
        AgentBuildOptions::<()>::default(),
    )
    .await
}

/// Build an agent with additional options (hooks, max_turns, validation).
///
/// This function extends [`build_agent`] with support for:
/// - Hooks (for tracking, display, ACP notifications)
/// - Maximum turns limit (to prevent infinite tool loops)
/// - Response validation (retry on empty responses)
///
/// # Arguments
///
/// * `config` - Configuration implementing [`LlmConfigLike`]
/// * `system_prompt` - System prompt/preamble for the agent
/// * `max_tokens` - Maximum tokens for completion
/// * `tools` - Vector of tools to register with the agent
/// * `options` - Additional options (hooks, max_turns, validation)
///
/// # Example
///
/// ```rust,ignore
/// let options = AgentBuildOptions {
///     max_turns: Some(25),
///     hook: Some(my_tracking_hook),
///     response_validation: Some(ResponseValidation::min_length(100)),
/// };
///
/// let agent = build_agent_with_options(&config, &prompt, 8192, tools, options).await?;
/// let (response, stats) = agent.execute(&prompt, &history).await?;
/// ```
pub async fn build_agent_with_options<C: LlmConfigLike, H: Hook + Clone>(
    config: &C,
    system_prompt: &str,
    max_tokens: u64,
    tools: Vec<Arc<dyn ToolDyn>>,
    options: AgentBuildOptions<H>,
) -> Result<AgentWrapper, AgentBuildError> {
    let provider_type = ProviderType::from_str(config.provider())?;
    let api_key = config.api_key().unwrap_or("none");

    info!(
        "Building agent: provider={}, model={}, think={:?} url={} temp={:?} top_k={:?} top_p={:?}",
        provider_type,
        config.model(),
        config.thinking(),
        config.url(),
        config.temperature(),
        config.top_k(),
        config.top_p()
    );

    match provider_type {
        ProviderType::OpenRouter => {
            debug!("Creating OpenRouter client");
            let mut builder = OpenRouterClientBuilder::new(api_key).base_url(config.url());

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            // Apply provider routing configuration if set
            if let Some(blacklist) = config.blacklist() {
                debug!("OpenRouter blacklist: {:?}", blacklist);
                builder = builder.blacklist(blacklist.to_vec());
            }
            if let Some(whitelist) = config.whitelist() {
                debug!("OpenRouter whitelist: {:?}", whitelist);
                builder = builder.whitelist(whitelist.to_vec());
            }
            if let Some(allow_fallbacks) = config.allow_fallbacks() {
                debug!("OpenRouter allow_fallbacks: {}", allow_fallbacks);
                builder = builder.allow_fallbacks(allow_fallbacks);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::OpenRouter(agent_builder.build()))
        }
        ProviderType::OpenAI => {
            debug!("Creating OpenAI client");
            let mut builder = OpenAIClientBuilder::new(api_key)
                .base_url(config.url())
                .max_tokens(max_tokens);

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            // top_k and repetition_penalty are not part of the canonical
            // OpenAI Chat Completions schema, but every OpenAI-compatible
            // local server we target (mlx-lm forks, vllm, sglang,
            // llama.cpp `--jinja`) accepts them as extension fields.
            // Forward when configured; upstream OpenAI silently ignores
            // unknown fields, so this is safe even against api.openai.com.
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }
            if let Some(rep) = config.repetition_penalty() {
                builder = builder.repetition_penalty(rep);
            }

            // Forward explicit thinking-mode toggle so OpenAI-compatible
            // servers fronting thinking-by-default models (mlx_fun
            // serving GLM-5.1, Qwen-thinking, …) emit `content` instead
            // of reasoning-only output. Skip when unset — preserves
            // upstream OpenAI behaviour where the field is meaningless.
            if let Some(enabled) = config.thinking() {
                debug!("OpenAI client enable_thinking={}", enabled);
                builder = builder.enable_thinking(enabled);
            }

            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::OpenAI(agent_builder.build()))
        }
        ProviderType::Anthropic => {
            let enable_thinking = config.thinking().unwrap_or(false);
            debug!(
                "Creating Anthropic client, thinking mode: {}",
                enable_thinking
            );
            let mut builder = AnthropicClientBuilder::new(api_key)
                .base_url(config.url())
                .enable_thinking(enable_thinking);

            if let Some(budget) = config.thinking_budget_tokens() {
                builder = builder.thinking_budget_tokens(budget);
            }

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::Anthropic(agent_builder.build()))
        }
        ProviderType::Minimax => {
            let enable_thinking = config.thinking().unwrap_or(false);
            debug!(
                "Creating MiniMax client, thinking mode: {}",
                enable_thinking
            );
            let mut builder = MinimaxClientBuilder::new(api_key)
                .base_url(config.url())
                .enable_thinking(enable_thinking);

            if let Some(budget) = config.thinking_budget_tokens() {
                builder = builder.thinking_budget_tokens(budget);
            }

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::Minimax(agent_builder.build()))
        }
        ProviderType::Cerebras => {
            debug!("Creating Cerebras client");
            let mut builder = CerebrasClientBuilder::new(api_key).base_url(config.url());

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::Cerebras(agent_builder.build()))
        }
        ProviderType::Ollama => {
            // Native Ollama /api/chat — one provider for local + cloud.
            // Cloud (https://ollama.com) needs a Bearer key but configs
            // shouldn't embed the secret; fall back to OLLAMA_API_KEY when
            // the config supplies no key (or the "none" sentinel). Local
            // use stays keyless.
            debug!("Creating native Ollama client");
            let ollama_key = if api_key.is_empty() || api_key == "none" {
                std::env::var("OLLAMA_API_KEY").unwrap_or_else(|_| "none".to_string())
            } else {
                api_key.to_string()
            };
            let mut builder = OllamaClientBuilder::new()
                .base_url(config.url())
                .api_key(&ollama_key)
                .max_tokens(max_tokens);

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }
            if let Some(mp) = config.min_p() {
                builder = builder.min_p(mp);
            }

            // Native thinking traces when requested by config.
            if let Some(enabled) = config.thinking() {
                debug!("Ollama client enable_thinking={}", enabled);
                builder = builder.enable_thinking(enabled);
            }

            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::Ollama(agent_builder.build()))
        }
        ProviderType::Zai => {
            // Determine thinking mode: config override, or default to enabled for ZAI
            let enable_thinking = config.thinking().unwrap_or(true);
            debug!("Creating ZAI client, thinking mode: {}", enable_thinking);

            let mut builder = ZaiClientBuilder::new(api_key)
                .base_url(config.url())
                .enable_thinking(enable_thinking);

            if let Some(budget) = config.thinking_budget_tokens() {
                builder = builder.thinking_budget_tokens(budget);
            }

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::Zai(agent_builder.build()))
        }
        ProviderType::MlxLm => {
            let mut builder = MlxLmClientBuilder::new().base_url(config.url());

            // Detect if this is an iquest model family (uses ChatML with specific stop sequences)
            let is_iquest = config.chat_template() == Some("iquest")
                || config.model().to_lowercase().contains("iquest");

            // Apply chat template: explicit config takes precedence, otherwise auto-detect from model name
            builder = match config.chat_template() {
                Some("minimax") => builder.chat_template(ChatTemplate::Minimax),
                Some("minimax25") | Some("minimax2.5") => {
                    builder.chat_template(ChatTemplate::Minimax25)
                }
                Some("chatml") | Some("iquest") | Some("qwen") => {
                    builder.chat_template(ChatTemplate::ChatML)
                }
                Some("qwen35") | Some("qwen3.5") => builder.chat_template(ChatTemplate::Qwen35),
                Some("glm") => builder.chat_template(ChatTemplate::GLM),
                Some("openai") => builder.chat_template(ChatTemplate::OpenAI),
                None => builder.auto_chat_template(config.model()), // Auto-detect from model name
                Some(other) => {
                    debug!(
                        "Unknown chat_template '{}', auto-detecting from model name",
                        other
                    );
                    builder.auto_chat_template(config.model())
                }
            };

            // Apply iquest-specific stop sequences for ChatML template and anti-repetition
            if is_iquest {
                debug!("Applying iquest model stop sequences (ChatML + anti-repetition)");
                builder = builder
                    .with_chatml_stop_sequences()
                    .with_anti_repetition_stops();
            }

            debug!("Creating MLX-LM client");

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            // Apply repetition penalty only if explicitly configured
            if let Some(repetition_penalty) = config.repetition_penalty() {
                builder = builder.repetition_penalty(repetition_penalty);
            }

            if let Some(ctx_size) = config.repetition_context_size() {
                builder = builder.repetition_context_size(ctx_size);
            }
            if let Some(freq) = config.frequency_penalty() {
                builder = builder.frequency_penalty(freq);
            }
            if let Some(pres) = config.presence_penalty() {
                builder = builder.presence_penalty(pres);
            }
            if let Some(mp) = config.min_p() {
                builder = builder.min_p(mp);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::MlxLm(agent_builder.build()))
        }
        ProviderType::LmStudio => {
            let mut builder = LmStudioClientBuilder::new().base_url(config.url());

            debug!("Creating LMStudio client");

            // Apply sampling parameters from config if set
            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }
            if let Some(top_p) = config.top_p() {
                builder = builder.top_p(top_p);
            }
            if let Some(top_k) = config.top_k() {
                builder = builder.top_k(top_k);
            }

            // Apply repeat penalty only if explicitly configured
            if let Some(repeat_penalty) = config.repeat_penalty() {
                builder = builder.repeat_penalty(repeat_penalty);
            }

            if let Some(ctx_size) = config.repetition_context_size() {
                builder = builder.repetition_context_size(ctx_size);
            }
            if let Some(freq) = config.frequency_penalty() {
                builder = builder.frequency_penalty(freq);
            }
            if let Some(pres) = config.presence_penalty() {
                builder = builder.presence_penalty(pres);
            }
            if let Some(mp) = config.min_p() {
                builder = builder.min_p(mp);
            }

            builder = builder.max_tokens(max_tokens);
            let client = builder.build();
            let model = client
                .completion_model_adapter(config.model())
                .with_metrics();
            let mut agent_builder = AgentBuilder::new(model)
                .preamble(system_prompt)
                .tools(tools);

            if let Some(max) = options.max_turns {
                agent_builder = agent_builder.max_turns(max);
            }
            if let Some(hook) = options.hook.clone() {
                agent_builder = agent_builder.hook(hook);
            }
            if let Some(ref validation) = options.response_validation {
                agent_builder = agent_builder.response_validation(validation.clone());
            }
            if let Some(ref optimizer) = options.optimizer {
                agent_builder = agent_builder.context_optimizer_arc(optimizer.clone());
            }
            if let Some(ref opt_config) = options.optimization_config {
                agent_builder = agent_builder.optimization_config(opt_config.clone());
            }

            Ok(AgentWrapper::LmStudio(agent_builder.build()))
        }
    }
}
