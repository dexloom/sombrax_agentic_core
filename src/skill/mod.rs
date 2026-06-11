mod discovery;
mod error;

pub use discovery::{default_search_paths, SkillRegistry};
pub use error::SkillError;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::OnceCell;

/// Maximum length for skill name
const MAX_NAME_LENGTH: usize = 64;

/// Maximum length for skill description
const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// Reserved words that cannot be used in skill names
const RESERVED_WORDS: &[&str] = &["anthropic", "claude"];

/// Metadata extracted from SKILL.md frontmatter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillMetadata {
    /// Unique skill name (lowercase, alphanumeric + hyphens, max 64 chars)
    pub name: String,

    /// When to use this skill (max 1024 chars, no XML tags)
    pub description: String,
}

impl SkillMetadata {
    /// Validate the metadata
    fn validate(&self) -> Result<(), SkillError> {
        // Validate name
        validate_skill_name(&self.name)?;

        // Validate description
        validate_skill_description(&self.description)?;

        Ok(())
    }
}

/// A skill that extends agent functionality
#[derive(Debug, Clone)]
pub struct Skill {
    /// Skill metadata (Level 1 - always loaded)
    metadata: SkillMetadata,

    /// Filesystem path to skill directory
    path: PathBuf,

    /// Cached SKILL.md content without frontmatter (Level 2 - loaded on demand)
    instructions: OnceCell<String>,
}

impl Skill {
    /// Parse a skill from a directory containing SKILL.md
    pub async fn from_path(path: PathBuf) -> Result<Self, SkillError> {
        // Ensure the path is a directory
        if !path.is_dir() {
            return Err(SkillError::DirectoryNotFound(path));
        }

        // Construct path to SKILL.md
        let skill_file = path.join("SKILL.md");
        if !skill_file.exists() {
            return Err(SkillError::FileNotFound(skill_file));
        }

        // Read SKILL.md
        let content = fs::read_to_string(&skill_file)
            .await
            .map_err(|e| SkillError::read_error(skill_file.clone(), e))?;

        // Parse frontmatter and extract metadata
        let metadata = parse_frontmatter(&skill_file, &content)?;

        // Validate metadata
        metadata.validate()?;

        Ok(Self {
            metadata,
            path,
            instructions: OnceCell::new(),
        })
    }

    /// Get skill metadata (Level 1)
    pub fn metadata(&self) -> &SkillMetadata {
        &self.metadata
    }

    /// Get skill name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get skill description
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Get skill path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load instructions on-demand (Level 2)
    pub async fn load_instructions(&self) -> Result<&str, SkillError> {
        self.instructions
            .get_or_try_init(|| async {
                // Read SKILL.md
                let skill_file = self.path.join("SKILL.md");
                let content = fs::read_to_string(&skill_file)
                    .await
                    .map_err(|e| SkillError::load_instructions_error(&self.metadata.name, e))?;

                // Extract content after frontmatter
                let instructions = extract_content_after_frontmatter(&content);

                Ok::<String, SkillError>(instructions.to_string())
            })
            .await
            .map(|s| s.as_str())
    }

    /// Check if skill description matches the given request
    /// Uses basic keyword matching
    pub fn matches(&self, request: &str) -> bool {
        let request_lower = request.to_lowercase();
        let description_lower = self.description().to_lowercase();

        // Check for direct substring match
        if request_lower.contains(&description_lower) || description_lower.contains(&request_lower)
        {
            return true;
        }

        // Extract meaningful keywords from description (filter very short words)
        let keywords: Vec<&str> = description_lower
            .split_whitespace()
            .filter(|w| w.len() > 2) // Filter very short words (a, an, the, etc.)
            .collect();

        // Extract request words for matching
        let request_words: Vec<&str> = request_lower.split_whitespace().collect();

        // Check if any keyword appears in the request (or shares common prefix)
        keywords.iter().any(|kw| {
            // Check if keyword is in request
            if request_lower.contains(kw) {
                return true;
            }
            // Check request words for prefix matching
            request_words.iter().any(|req_word| {
                // Share a common prefix of at least 5 characters
                let min_len = std::cmp::min(kw.len(), req_word.len());
                if min_len >= 5 {
                    let common_prefix_len = kw
                        .chars()
                        .zip(req_word.chars())
                        .take_while(|(a, b)| a == b)
                        .count();
                    common_prefix_len >= 5
                } else {
                    false
                }
            })
        })
    }

    /// List all resource files in the skill directory (Level 3+)
    pub async fn list_resources(&self) -> Result<Vec<String>, SkillError> {
        list_skill_resources(&self.path, &self.metadata.name).await
    }
}

/// Parse YAML frontmatter from SKILL.md content
fn parse_frontmatter(path: &Path, content: &str) -> Result<SkillMetadata, SkillError> {
    // Find frontmatter delimiters
    let lines: Vec<&str> = content.lines().collect();

    // Must start with ---
    if lines.is_empty() || lines[0] != "---" {
        return Err(SkillError::InvalidFrontmatter {
            path: path.to_path_buf(),
            source: serde_yaml::from_str::<()>("invalid").unwrap_err(),
        });
    }

    // Find ending ---
    let end_idx = lines
        .iter()
        .skip(1)
        .position(|&line| line == "---")
        .ok_or_else(|| SkillError::InvalidFrontmatter {
            path: path.to_path_buf(),
            source: serde_yaml::from_str::<()>("invalid").unwrap_err(),
        })?
        + 1;

    // Extract frontmatter content
    let frontmatter_content = lines[1..end_idx].join("\n");

    // Parse YAML
    let metadata: SkillMetadata = serde_yaml::from_str(&frontmatter_content)
        .map_err(|e| SkillError::invalid_frontmatter(path.to_path_buf(), e))?;

    Ok(metadata)
}

/// Extract content after frontmatter
fn extract_content_after_frontmatter(content: &str) -> &str {
    let lines: Vec<&str> = content.lines().collect();

    // Find ending --- of frontmatter
    if let Some(end_idx) = lines.iter().skip(1).position(|&line| line == "---") {
        // Join lines after frontmatter (skip the ending ---)
        let start_idx = end_idx + 2; // +1 for position offset, +1 to skip the --- line
        if start_idx < lines.len() {
            // Find the starting position in the original string
            let offset: usize = lines[..start_idx].iter().map(|l| l.len() + 1).sum();
            return &content[offset..];
        }
    }

    // If no frontmatter found, return whole content
    content
}

/// Validate skill name according to rules
fn validate_skill_name(name: &str) -> Result<(), SkillError> {
    // Check length
    if name.is_empty() {
        return Err(SkillError::invalid_name(name, "Name cannot be empty"));
    }

    if name.len() > MAX_NAME_LENGTH {
        return Err(SkillError::invalid_name(
            name,
            format!("Name exceeds maximum length of {}", MAX_NAME_LENGTH),
        ));
    }

    // Check character set (lowercase, numbers, hyphens only)
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(SkillError::invalid_name(
            name,
            "Name must contain only lowercase letters, numbers, and hyphens",
        ));
    }

    // Check for reserved words
    for reserved in RESERVED_WORDS {
        if name.contains(reserved) {
            return Err(SkillError::invalid_name(
                name,
                format!("Name cannot contain reserved word '{}'", reserved),
            ));
        }
    }

    Ok(())
}

/// Validate skill description according to rules
fn validate_skill_description(description: &str) -> Result<(), SkillError> {
    // Check empty
    if description.trim().is_empty() {
        return Err(SkillError::InvalidDescription(
            "Description cannot be empty".to_string(),
        ));
    }

    // Check length
    if description.len() > MAX_DESCRIPTION_LENGTH {
        return Err(SkillError::InvalidDescription(format!(
            "Description exceeds maximum length of {}",
            MAX_DESCRIPTION_LENGTH
        )));
    }

    // Check for XML tags (simple check)
    if description.contains('<') && description.contains('>') {
        return Err(SkillError::InvalidDescription(
            "Description cannot contain XML tags".to_string(),
        ));
    }

    Ok(())
}

/// List all files in skill directory (for Level 3 discovery)
async fn list_skill_resources(
    skill_path: &Path,
    skill_name: &str,
) -> Result<Vec<String>, SkillError> {
    let mut resources = Vec::new();

    // Read directory entries
    let mut entries = fs::read_dir(skill_path)
        .await
        .map_err(|e| SkillError::list_resources_error(skill_name, e))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| SkillError::list_resources_error(skill_name, e))?
    {
        let path = entry.path();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Skip SKILL.md itself
        if file_name == "SKILL.md" {
            continue;
        }

        // Add files and directories
        if path.is_file() {
            resources.push(file_name);
        } else if path.is_dir() {
            // Recursively list directory contents
            let subdir_resources =
                list_directory_recursive(path, skill_name.to_string(), file_name).await?;
            resources.extend(subdir_resources);
        }
    }

    // Sort for consistent ordering
    resources.sort();

    Ok(resources)
}

/// Recursively list directory contents
fn list_directory_recursive(
    dir_path: PathBuf,
    skill_name: String,
    prefix: String,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>, SkillError>> + Send>> {
    Box::pin(async move {
        let mut resources = Vec::new();

        let mut entries = fs::read_dir(&dir_path)
            .await
            .map_err(|e| SkillError::list_resources_error(&skill_name, e))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| SkillError::list_resources_error(&skill_name, e))?
        {
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            let relative_path = format!("{}/{}", prefix, file_name);

            if path.is_file() {
                resources.push(relative_path);
            } else if path.is_dir() {
                let subdir_resources =
                    list_directory_recursive(path, skill_name.clone(), relative_path).await?;
                resources.extend(subdir_resources);
            }
        }

        Ok(resources)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_skill_name() {
        // Valid names
        assert!(validate_skill_name("hello-world").is_ok());
        assert!(validate_skill_name("test123").is_ok());
        assert!(validate_skill_name("my-skill-v2").is_ok());

        // Invalid names
        assert!(validate_skill_name("").is_err()); // Empty
        assert!(validate_skill_name("Hello-World").is_err()); // Uppercase
        assert!(validate_skill_name("hello_world").is_err()); // Underscore
        assert!(validate_skill_name("hello world").is_err()); // Space
        assert!(validate_skill_name("anthropic-skill").is_err()); // Reserved word
        assert!(validate_skill_name("claude-helper").is_err()); // Reserved word
        assert!(validate_skill_name(&"a".repeat(65)).is_err()); // Too long
    }

    #[test]
    fn test_validate_skill_description() {
        // Valid descriptions
        assert!(validate_skill_description("A simple test skill").is_ok());
        assert!(validate_skill_description("Use when you need to process files").is_ok());

        // Invalid descriptions
        assert!(validate_skill_description("").is_err()); // Empty
        assert!(validate_skill_description("   ").is_err()); // Whitespace only
        assert!(validate_skill_description("<script>alert('xss')</script>").is_err()); // XML tags
        assert!(validate_skill_description(&"a".repeat(1025)).is_err()); // Too long
    }

    #[test]
    fn test_extract_content_after_frontmatter() {
        let content = r#"---
name: test
description: A test skill
---

# Test Skill

This is the content.
"#;

        let extracted = extract_content_after_frontmatter(content);
        assert!(extracted.starts_with("\n# Test Skill"));
        assert!(extracted.contains("This is the content."));
    }
}
