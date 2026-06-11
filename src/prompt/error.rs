//! Errors for the prompt module.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur when working with system prompts.
#[derive(Debug, Error)]
pub enum PromptError {
    /// I/O error reading a prompt file.
    #[error("io error at {path:?}: {source}")]
    Io {
        /// The path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Prompt failed validation (name or body).
    #[error("invalid prompt at {path:?}: {reason}")]
    Invalid {
        /// Path to the offending file.
        path: PathBuf,
        /// Reason it was rejected.
        reason: String,
    },

    /// YAML frontmatter failed to parse.
    #[error("malformed frontmatter at {path:?}: {source}")]
    Frontmatter {
        /// Path to the offending file.
        path: PathBuf,
        /// Underlying YAML parse error.
        #[source]
        source: serde_yaml::Error,
    },
}

impl PromptError {
    /// Construct an [`Io`](Self::Io) error.
    pub fn io(path: PathBuf, source: std::io::Error) -> Self {
        Self::Io { path, source }
    }

    /// Construct an [`Invalid`](Self::Invalid) error.
    pub fn invalid(path: PathBuf, reason: impl Into<String>) -> Self {
        Self::Invalid {
            path,
            reason: reason.into(),
        }
    }

    /// Construct a [`Frontmatter`](Self::Frontmatter) error.
    pub fn frontmatter(path: PathBuf, source: serde_yaml::Error) -> Self {
        Self::Frontmatter { path, source }
    }
}
