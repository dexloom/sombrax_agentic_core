//! Factory function for building ExtractorWrapper from LLM configuration.

use crate::extractor::ExtractorWrapper;
use crate::providers::{
    AnthropicClientBuilder, CerebrasClientBuilder, ChatTemplate, LlmConfigLike,
    LmStudioClientBuilder, MinimaxClientBuilder, MlxLmClientBuilder, OllamaClientBuilder,
    OpenAIClientBuilder, OpenRouterClientBuilder, ProviderType, ProviderTypeError,
    ZaiClientBuilder,
};
use thiserror::Error;
use tracing::{debug, info};

/// Error type for extractor building operations.
#[derive(Error, Debug)]
pub enum ExtractorBuildError {
    /// Unknown or unsupported provider type
    #[error("Provider type error: {0}")]
    ProviderType(#[from] ProviderTypeError),
}

/// Build an extractor wrapper from LLM configuration.
///
/// This is the primary factory function for creating extractors with the correct
/// provider client based on the `provider` field in the config.
///
/// # Type Parameters
///
/// * `C` - Any type implementing `LlmConfigLike` trait
///
/// # Arguments
///
/// * `config` - LLM configuration containing provider, URL, model, and API key
///
/// # Returns
///
/// Returns an `ExtractorWrapper` that can extract structured data using the
/// configured provider.
///
/// # Supported Providers
///
/// - `openrouter` - OpenRouter API (default)
/// - `openai` - Direct OpenAI API
/// - `anthropic` / `claude` - Anthropic Claude API
/// - `minimax` - MiniMax Anthropic-compatible API
/// - `cerebras` - Cerebras with custom tool content serialization
/// - `ollama` - Local Ollama (uses OpenAI-compatible client)
/// - `zai` - ZAI GLM-4.6 with thinking mode support
/// - `mlx` / `mlxlm` - Local MLX-LM for Apple Silicon
///
/// # Example
///
/// ```rust,ignore
/// use sombrax_agentic_core::extractor::build_extractor;
///
/// let extractor = build_extractor(&llm_config)?;
/// let result: MyResponse = extractor
///     .extract(&llm_config.model(), preamble, prompt)
///     .await?;
/// ```
pub fn build_extractor<C: LlmConfigLike>(
    config: &C,
) -> Result<ExtractorWrapper, ExtractorBuildError> {
    let provider_type = ProviderType::from_str(config.provider())?;
    let api_key = config.api_key().unwrap_or("none");

    info!(
        "Building extractor: provider={}, model={}, url={}",
        provider_type,
        config.model(),
        config.url()
    );

    match provider_type {
        ProviderType::OpenRouter => {
            debug!("Creating OpenRouter extractor client");
            let mut builder = OpenRouterClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::OpenRouter(client))
        }
        ProviderType::OpenAI => {
            debug!("Creating OpenAI extractor client");
            let mut builder = OpenAIClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::OpenAI(client))
        }
        ProviderType::Anthropic => {
            debug!("Creating Anthropic extractor client");
            let mut builder = AnthropicClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::Anthropic(client))
        }
        ProviderType::Minimax => {
            debug!("Creating MiniMax extractor client");
            let mut builder = MinimaxClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::Minimax(client))
        }
        ProviderType::Ollama => {
            // Native Ollama /api/chat — one client for local + cloud.
            debug!("Creating native Ollama extractor client");
            let mut builder = OllamaClientBuilder::new()
                .base_url(config.url())
                .api_key(api_key);

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::Ollama(client))
        }
        ProviderType::Cerebras => {
            debug!("Creating Cerebras extractor client");
            let mut builder = CerebrasClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::Cerebras(client))
        }
        ProviderType::Zai => {
            debug!("Creating ZAI extractor client");
            let mut builder = ZaiClientBuilder::new(api_key).base_url(config.url());

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::Zai(client))
        }
        ProviderType::MlxLm => {
            let mut builder = MlxLmClientBuilder::new().base_url(config.url());

            // Apply chat template: explicit config takes precedence, otherwise auto-detect from model name
            builder = match config.chat_template() {
                Some("minimax") => builder.chat_template(ChatTemplate::Minimax),
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
            debug!("Creating MLX-LM extractor client");

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::MlxLm(client))
        }
        ProviderType::LmStudio => {
            let mut builder = LmStudioClientBuilder::new().base_url(config.url());
            debug!("Creating LMStudio extractor client");

            if let Some(temp) = config.temperature() {
                builder = builder.temperature(temp);
            }

            // Apply anti-repetition parameters
            let repeat_penalty = config.repeat_penalty().unwrap_or(1.1);
            builder = builder.repeat_penalty(repeat_penalty);

            if let Some(ctx_size) = config.repetition_context_size() {
                builder = builder.repetition_context_size(ctx_size);
            }

            let client = builder.build();
            Ok(ExtractorWrapper::LmStudio(client))
        }
    }
}
