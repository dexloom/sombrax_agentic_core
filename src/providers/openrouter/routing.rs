//! OpenRouter provider routing configuration

use serde::{Deserialize, Serialize};

/// Provider routing configuration for OpenRouter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterProviderConfig {
    /// Providers to ignore (blacklist)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Providers to use exclusively (whitelist)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Allow fallback to other providers if preferred is unavailable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
}

impl OpenRouterProviderConfig {
    /// Create a new empty provider config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add providers to the ignore list (blacklist)
    pub fn blacklist(mut self, providers: Vec<String>) -> Self {
        self.ignore = Some(providers);
        self
    }

    /// Set providers to use exclusively (whitelist)
    pub fn whitelist(mut self, providers: Vec<String>) -> Self {
        self.only = Some(providers);
        self
    }

    /// Set whether to allow fallbacks
    pub fn allow_fallbacks(mut self, allow: bool) -> Self {
        self.allow_fallbacks = Some(allow);
        self
    }

    /// Check if config has any routing rules
    pub fn has_rules(&self) -> bool {
        self.ignore.is_some() || self.only.is_some() || self.allow_fallbacks.is_some()
    }
}
