use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur when working with skills
#[derive(Debug, Error)]
pub enum SkillError {
    /// Skill file not found
    #[error("Skill file not found: {0}")]
    FileNotFound(PathBuf),

    /// Failed to read skill file
    #[error("Failed to read skill file {path}: {source}")]
    ReadError {
        /// Path to the skill file.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Invalid YAML frontmatter
    #[error("Invalid YAML frontmatter in {path}: {source}")]
    InvalidFrontmatter {
        /// Path to the skill file.
        path: PathBuf,
        /// YAML parsing error.
        source: serde_yaml::Error,
    },

    /// Missing required field in frontmatter
    #[error("Missing required field '{field}' in {path}")]
    MissingField {
        /// Name of the missing field.
        field: String,
        /// Path to the skill file.
        path: PathBuf,
    },

    /// Invalid skill name
    #[error("Invalid skill name '{name}': {reason}")]
    InvalidName {
        /// The invalid skill name.
        name: String,
        /// Reason why the name is invalid.
        reason: String,
    },

    /// Invalid skill description
    #[error("Invalid skill description: {0}")]
    InvalidDescription(String),

    /// Skill not found in registry
    #[error("Skill not found: {0}")]
    NotFound(String),

    /// Failed to load skill instructions
    #[error("Failed to load instructions for skill '{skill}': {source}")]
    LoadInstructionsError {
        /// Skill name.
        skill: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Failed to list skill resources
    #[error("Failed to list resources for skill '{skill}': {source}")]
    ListResourcesError {
        /// Skill name.
        skill: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Directory traversal attack detected
    #[error("Invalid path: directory traversal detected in {0}")]
    DirectoryTraversal(PathBuf),

    /// Skill directory not found
    #[error("Skill directory not found: {0}")]
    DirectoryNotFound(PathBuf),
}

impl SkillError {
    /// Create a ReadError
    pub fn read_error(path: PathBuf, source: std::io::Error) -> Self {
        Self::ReadError { path, source }
    }

    /// Create an InvalidFrontmatter error
    pub fn invalid_frontmatter(path: PathBuf, source: serde_yaml::Error) -> Self {
        Self::InvalidFrontmatter { path, source }
    }

    /// Create a MissingField error
    pub fn missing_field(field: impl Into<String>, path: PathBuf) -> Self {
        Self::MissingField {
            field: field.into(),
            path,
        }
    }

    /// Create an InvalidName error
    pub fn invalid_name(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidName {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Create a LoadInstructionsError
    pub fn load_instructions_error(skill: impl Into<String>, source: std::io::Error) -> Self {
        Self::LoadInstructionsError {
            skill: skill.into(),
            source,
        }
    }

    /// Create a ListResourcesError
    pub fn list_resources_error(skill: impl Into<String>, source: std::io::Error) -> Self {
        Self::ListResourcesError {
            skill: skill.into(),
            source,
        }
    }
}
