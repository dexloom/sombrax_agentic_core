//! Provider type enumeration for LLM client selection.
//!
//! This module defines the [`ProviderType`] enum which represents all supported
//! LLM providers. Use this with the factory functions in [`super::builder`] to
//! create agents from configuration.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Error type for provider type parsing operations.
#[derive(Error, Debug, Clone)]
pub enum ProviderTypeError {
    /// Unknown or unsupported provider type
    #[error("Unknown provider type: '{0}'. Supported: openrouter, openai, anthropic, minimax, cerebras, ollama, zai, mlx, lmstudio")]
    UnknownProvider(String),
}

/// Supported LLM provider types.
///
/// Each provider may have different API requirements, serialization formats,
/// and optional features (like thinking mode for ZAI).
///
/// # Example
///
/// ```rust,ignore
/// use sombrax_agentic_core::providers::ProviderType;
///
/// let provider = ProviderType::from_str("openrouter")?;
/// assert_eq!(provider.default_base_url(), "https://openrouter.ai/api/v1");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenRouter - aggregator supporting multiple models with provider routing
    #[default]
    OpenRouter,
    /// OpenAI - direct OpenAI API (GPT-4, GPT-4o, etc.)
    OpenAI,
    /// Anthropic - Claude models (Opus, Sonnet, Haiku)
    Anthropic,
    /// MiniMax - Anthropic-compatible API (M2.x)
    Minimax,
    /// Cerebras - fast inference, requires simple string tool content
    Cerebras,
    /// Ollama - local model serving (uses OpenAI-compatible API)
    Ollama,
    /// ZAI - GLM-4.6 with thinking mode support
    Zai,
    /// MLX-LM - Apple Silicon local model serving with custom chat templates
    MlxLm,
    /// LMStudio - local model serving with anti-repetition controls
    LmStudio,
}

impl ProviderType {
    /// Parse provider type from string (case-insensitive).
    ///
    /// # Supported Values
    ///
    /// - `"openrouter"` -> [`ProviderType::OpenRouter`]
    /// - `"openai"` -> [`ProviderType::OpenAI`]
    /// - `"anthropic"`, `"claude"` -> [`ProviderType::Anthropic`]
    /// - `"minimax"` -> [`ProviderType::Minimax`]
    /// - `"cerebras"` -> [`ProviderType::Cerebras`]
    /// - `"ollama"` -> [`ProviderType::Ollama`]
    /// - `"zai"` -> [`ProviderType::Zai`]
    /// - `"mlx"`, `"mlxlm"` -> [`ProviderType::MlxLm`]
    /// - `"lmstudio"` -> [`ProviderType::LmStudio`]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ProviderTypeError> {
        match s.to_lowercase().as_str() {
            "openrouter" => Ok(Self::OpenRouter),
            "openai" => Ok(Self::OpenAI),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "minimax" => Ok(Self::Minimax),
            "cerebras" => Ok(Self::Cerebras),
            "ollama" => Ok(Self::Ollama),
            "zai" => Ok(Self::Zai),
            "mlx" | "mlxlm" => Ok(Self::MlxLm),
            "lmstudio" => Ok(Self::LmStudio),
            _ => Err(ProviderTypeError::UnknownProvider(s.to_string())),
        }
    }

    /// Get the default base URL for this provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Minimax => "https://api.minimax.io/anthropic",
            Self::Cerebras => "https://api.cerebras.ai/v1",
            Self::Ollama => "http://localhost:11434",
            Self::Zai => "https://api.z.ai/api/coding/paas/v4",
            Self::MlxLm => "http://localhost:8080/v1",
            Self::LmStudio => "http://localhost:1234/v1",
        }
    }

    /// Check if this provider requires custom tool content serialization.
    ///
    /// Cerebras requires simple string tool content (not arrays).
    pub fn requires_custom_tool_serialization(&self) -> bool {
        matches!(self, Self::Cerebras)
    }
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenRouter => write!(f, "openrouter"),
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Minimax => write!(f, "minimax"),
            Self::Cerebras => write!(f, "cerebras"),
            Self::Ollama => write!(f, "ollama"),
            Self::Zai => write!(f, "zai"),
            Self::MlxLm => write!(f, "mlx"),
            Self::LmStudio => write!(f, "lmstudio"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!(
            ProviderType::from_str("openrouter").unwrap(),
            ProviderType::OpenRouter
        );
        assert_eq!(
            ProviderType::from_str("OPENROUTER").unwrap(),
            ProviderType::OpenRouter
        );
        assert_eq!(
            ProviderType::from_str("OpenRouter").unwrap(),
            ProviderType::OpenRouter
        );
        assert_eq!(
            ProviderType::from_str("openai").unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            ProviderType::from_str("anthropic").unwrap(),
            ProviderType::Anthropic
        );
        assert_eq!(
            ProviderType::from_str("claude").unwrap(),
            ProviderType::Anthropic
        );
        assert_eq!(
            ProviderType::from_str("minimax").unwrap(),
            ProviderType::Minimax
        );
        assert_eq!(
            ProviderType::from_str("cerebras").unwrap(),
            ProviderType::Cerebras
        );
        assert_eq!(
            ProviderType::from_str("ollama").unwrap(),
            ProviderType::Ollama
        );
        assert_eq!(ProviderType::from_str("zai").unwrap(), ProviderType::Zai);
        assert_eq!(ProviderType::from_str("ZAI").unwrap(), ProviderType::Zai);
        assert_eq!(ProviderType::from_str("mlx").unwrap(), ProviderType::MlxLm);
        assert_eq!(
            ProviderType::from_str("mlxlm").unwrap(),
            ProviderType::MlxLm
        );
        assert_eq!(ProviderType::from_str("MLX").unwrap(), ProviderType::MlxLm);
        assert_eq!(
            ProviderType::from_str("lmstudio").unwrap(),
            ProviderType::LmStudio
        );
        assert_eq!(
            ProviderType::from_str("LMSTUDIO").unwrap(),
            ProviderType::LmStudio
        );

        assert!(ProviderType::from_str("invalid").is_err());
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::OpenRouter.to_string(), "openrouter");
        assert_eq!(ProviderType::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderType::Minimax.to_string(), "minimax");
        assert_eq!(ProviderType::Cerebras.to_string(), "cerebras");
        assert_eq!(ProviderType::Zai.to_string(), "zai");
        assert_eq!(ProviderType::MlxLm.to_string(), "mlx");
        assert_eq!(ProviderType::LmStudio.to_string(), "lmstudio");
    }

    #[test]
    fn test_default_base_urls() {
        assert_eq!(
            ProviderType::OpenRouter.default_base_url(),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            ProviderType::Anthropic.default_base_url(),
            "https://api.anthropic.com"
        );
        assert_eq!(
            ProviderType::Minimax.default_base_url(),
            "https://api.minimax.io/anthropic"
        );
        assert_eq!(
            ProviderType::MlxLm.default_base_url(),
            "http://localhost:8080/v1"
        );
        assert_eq!(
            ProviderType::LmStudio.default_base_url(),
            "http://localhost:1234/v1"
        );
        assert_eq!(
            ProviderType::Ollama.default_base_url(),
            "http://localhost:11434"
        );
    }

    #[test]
    fn test_requires_custom_serialization() {
        assert!(!ProviderType::OpenRouter.requires_custom_tool_serialization());
        assert!(!ProviderType::OpenAI.requires_custom_tool_serialization());
        assert!(!ProviderType::Anthropic.requires_custom_tool_serialization());
        assert!(!ProviderType::Minimax.requires_custom_tool_serialization());
        assert!(ProviderType::Cerebras.requires_custom_tool_serialization());
        assert!(!ProviderType::Ollama.requires_custom_tool_serialization());
        assert!(!ProviderType::Zai.requires_custom_tool_serialization());
        assert!(!ProviderType::MlxLm.requires_custom_tool_serialization());
        assert!(!ProviderType::LmStudio.requires_custom_tool_serialization());
    }
}
