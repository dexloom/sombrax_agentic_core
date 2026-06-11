//! Wrapper enum for different agent types providing a unified execution interface.
//!
//! This module provides [`AgentWrapper`], an enum that wraps agents with different
//! provider-specific completion models. This is necessary because different providers
//! return different concrete agent types, and the wrapper provides a unified interface
//! for executing agents regardless of the underlying provider.
//!
//! # Example
//!
//! ```rust,ignore
//! use sombrax_agentic_core::agent::AgentWrapper;
//! use sombrax_agentic_core::providers::{build_agent, ProviderType};
//!
//! // Build an agent from configuration
//! let agent = build_agent(&config, "You are helpful", 8192, vec![]).await?;
//!
//! // Execute with unified interface
//! let (response, stats) = agent.execute("Hello!", &[]).await?;
//! ```

use crate::agent::{Agent, ExecutionStats};
use crate::error::CompletionError;
use crate::message::Message;
use crate::provider::MetricsCompletionModel;
use crate::providers::{
    provider_type::ProviderType, AnthropicCompletionModelAdapter, CerebrasCompletionModelAdapter,
    LmStudioCompletionModelAdapter, MinimaxCompletionModelAdapter, MlxLmCompletionModelAdapter,
    OllamaCompletionModelAdapter, OpenAICompletionModelAdapter, OpenRouterCompletionModelAdapter,
    ZaiCompletionModelAdapter,
};

/// Wrapper enum for different agent types.
///
/// This is necessary because different providers return different concrete agent types
/// (`Agent<MetricsCompletionModel<OpenRouterAdapter>>` vs `Agent<MetricsCompletionModel<OpenAIAdapter>>`).
/// The wrapper provides a unified interface for executing agents.
///
/// All models are wrapped with [`MetricsCompletionModel`] for automatic performance tracking.
pub enum AgentWrapper {
    /// OpenRouter agent using OpenRouterCompletionModelAdapter with metrics.
    OpenRouter(Agent<MetricsCompletionModel<OpenRouterCompletionModelAdapter>>),
    /// OpenAI agent using OpenAICompletionModelAdapter with metrics.
    OpenAI(Agent<MetricsCompletionModel<OpenAICompletionModelAdapter>>),
    /// Anthropic (Claude) agent using AnthropicCompletionModelAdapter with metrics.
    Anthropic(Agent<MetricsCompletionModel<AnthropicCompletionModelAdapter>>),
    /// MiniMax agent using MinimaxCompletionModelAdapter with metrics.
    Minimax(Agent<MetricsCompletionModel<MinimaxCompletionModelAdapter>>),
    /// Cerebras agent using CerebrasCompletionModelAdapter with metrics.
    Cerebras(Agent<MetricsCompletionModel<CerebrasCompletionModelAdapter>>),
    /// Ollama agent using native `/api/chat` (cloud + local) with metrics.
    Ollama(Agent<MetricsCompletionModel<OllamaCompletionModelAdapter>>),
    /// ZAI agent using ZaiCompletionModelAdapter with thinking mode support and metrics.
    Zai(Agent<MetricsCompletionModel<ZaiCompletionModelAdapter>>),
    /// MLX-LM agent using MlxLmCompletionModelAdapter for Apple Silicon local models with metrics.
    MlxLm(Agent<MetricsCompletionModel<MlxLmCompletionModelAdapter>>),
    /// LMStudio agent using LmStudioCompletionModelAdapter with anti-repetition controls and metrics.
    LmStudio(Agent<MetricsCompletionModel<LmStudioCompletionModelAdapter>>),
}

impl AgentWrapper {
    /// Execute the agent with a prompt and optional history.
    ///
    /// Returns the response content and execution statistics from the agent loop.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user prompt to send to the agent
    /// * `history` - Conversation history slice
    ///
    /// # Returns
    ///
    /// Tuple of (response_content, execution_stats).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (content, stats) = agent.execute("What is 2+2?", &history).await?;
    /// println!("Response: {}", content);
    /// println!("Tokens used: {}", stats.total_tokens());
    /// ```
    pub async fn execute(
        &self,
        prompt: &str,
        history: &[Message],
    ) -> Result<(String, ExecutionStats), CompletionError> {
        match self {
            AgentWrapper::OpenRouter(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::OpenAI(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::Anthropic(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::Minimax(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::Cerebras(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::Ollama(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::Zai(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::MlxLm(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
            AgentWrapper::LmStudio(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats)),
        }
    }

    /// Execute the agent and return response with message history.
    ///
    /// This method returns the full message history from the execution,
    /// enabling multi-stage pipelines where one agent's output becomes
    /// another agent's input history.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user prompt to send to the agent
    /// * `history` - Conversation history slice
    ///
    /// # Returns
    ///
    /// Tuple of (response_content, execution_stats, message_history).
    pub async fn execute_with_messages(
        &self,
        prompt: &str,
        history: &[Message],
    ) -> Result<(String, ExecutionStats, Vec<Message>), CompletionError> {
        match self {
            AgentWrapper::OpenRouter(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::OpenAI(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::Anthropic(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::Minimax(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::Cerebras(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::Ollama(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::Zai(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::MlxLm(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
            AgentWrapper::LmStudio(agent) => agent
                .prompt_with_history(prompt, history)
                .await
                .map(|r| (r.content(), r.stats, r.messages)),
        }
    }

    /// Get the provider type for this agent.
    pub fn provider_type(&self) -> ProviderType {
        match self {
            AgentWrapper::OpenRouter(_) => ProviderType::OpenRouter,
            AgentWrapper::OpenAI(_) => ProviderType::OpenAI,
            AgentWrapper::Anthropic(_) => ProviderType::Anthropic,
            AgentWrapper::Minimax(_) => ProviderType::Minimax,
            AgentWrapper::Cerebras(_) => ProviderType::Cerebras,
            AgentWrapper::Ollama(_) => ProviderType::Ollama,
            AgentWrapper::Zai(_) => ProviderType::Zai,
            AgentWrapper::MlxLm(_) => ProviderType::MlxLm,
            AgentWrapper::LmStudio(_) => ProviderType::LmStudio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_mapping() {
        // This test verifies the mapping is complete and correct
        // Actual agent creation requires provider clients
        assert_eq!(ProviderType::OpenRouter.to_string(), "openrouter");
        assert_eq!(ProviderType::OpenAI.to_string(), "openai");
        assert_eq!(ProviderType::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderType::Minimax.to_string(), "minimax");
        assert_eq!(ProviderType::Cerebras.to_string(), "cerebras");
        assert_eq!(ProviderType::Ollama.to_string(), "ollama");
        assert_eq!(ProviderType::Zai.to_string(), "zai");
        assert_eq!(ProviderType::MlxLm.to_string(), "mlx");
    }
}
