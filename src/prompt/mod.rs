//! System prompt assets — first-class, on-disk prompt files with a
//! deterministic name-based resolution ladder.
//!
//! A `SystemPrompt` is a named markdown file with optional YAML
//! frontmatter (`description` only, for catalog listings). The
//! `PromptRegistry` discovers prompt files in layered search paths and
//! resolves requested names via a hash-strip ladder (see the
//! `discovery` module).

mod discovery;
mod error;

pub use discovery::{default_search_paths, PromptRegistry};
pub use error::PromptError;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::OnceCell;

/// Maximum length for a prompt name (derived from file stem).
const MAX_NAME_LENGTH: usize = 64;

/// Metadata extracted from a prompt file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptMetadata {
    /// Derived from filename stem. Lowercase, `[a-z0-9._-]`, ≤64 chars.
    pub name: String,
    /// Optional YAML frontmatter field.
    #[serde(default)]
    pub description: Option<String>,
}

/// YAML-frontmatter shape (only `description` is honored).
#[derive(Debug, Clone, Deserialize)]
struct Frontmatter {
    #[serde(default)]
    description: Option<String>,
}

/// A system prompt asset loaded from disk.
///
/// The body is lazily loaded on first call to [`body`](Self::body) and cached
/// thereafter.
#[derive(Debug, Clone)]
pub struct SystemPrompt {
    metadata: PromptMetadata,
    path: PathBuf,
    body: OnceCell<String>,
}

impl SystemPrompt {
    /// Parse a prompt from a `.md` file path.
    ///
    /// The file stem becomes the prompt name; optional YAML frontmatter may
    /// set `description`. Validation happens here: names and bodies that
    /// violate `validate_prompt_name`/non-empty-body are rejected.
    pub async fn from_path(path: PathBuf) -> Result<Self, PromptError> {
        if !path.is_file() {
            return Err(PromptError::invalid(path.clone(), "not a regular file"));
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .ok_or_else(|| PromptError::invalid(path.clone(), "filename stem not valid UTF-8"))?;

        validate_prompt_name(&stem).map_err(|reason| PromptError::invalid(path.clone(), reason))?;

        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| PromptError::io(path.clone(), e))?;

        let (frontmatter, body) = split_frontmatter(&path, &content)?;

        if body.trim().is_empty() {
            return Err(PromptError::invalid(path.clone(), "prompt body is empty"));
        }

        let metadata = PromptMetadata {
            name: stem,
            description: frontmatter.and_then(|f| f.description),
        };

        let cell = OnceCell::new();
        // Seed the cell with the pre-read body so `body()` is cheap.
        cell.set(body).expect("OnceCell is fresh");

        Ok(Self {
            metadata,
            path,
            body: cell,
        })
    }

    /// Prompt metadata.
    pub fn metadata(&self) -> &PromptMetadata {
        &self.metadata
    }

    /// Prompt name (filename stem).
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Optional description from frontmatter.
    pub fn description(&self) -> Option<&str> {
        self.metadata.description.as_deref()
    }

    /// Filesystem path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Body (markdown after frontmatter stripping). Lazily loaded.
    pub async fn body(&self) -> Result<&str, PromptError> {
        self.body
            .get_or_try_init(|| async {
                let content = fs::read_to_string(&self.path)
                    .await
                    .map_err(|e| PromptError::io(self.path.clone(), e))?;
                let (_, body) = split_frontmatter(&self.path, &content)?;
                Ok::<String, PromptError>(body)
            })
            .await
            .map(|s| s.as_str())
    }
}

/// Split a prompt file into (optional frontmatter, body).
///
/// If the file begins with `---` on its own line, a YAML frontmatter block is
/// expected, terminated by another `---` line. Body is whatever follows.
fn split_frontmatter(
    path: &Path,
    content: &str,
) -> Result<(Option<Frontmatter>, String), PromptError> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.first().copied() != Some("---") {
        return Ok((None, content.to_string()));
    }

    let end = lines
        .iter()
        .skip(1)
        .position(|l| *l == "---")
        .ok_or_else(|| PromptError::invalid(path.to_path_buf(), "unterminated YAML frontmatter"))?
        + 1;

    let yaml = lines[1..end].join("\n");
    let frontmatter: Frontmatter =
        serde_yaml::from_str(&yaml).map_err(|e| PromptError::frontmatter(path.to_path_buf(), e))?;

    // Reconstruct body after the closing `---` line.
    let body = if end < lines.len() {
        lines[end + 1..].join("\n")
    } else {
        String::new()
    };

    // Preserve a trailing newline if the original file had one (cosmetic).
    let body = if content.ends_with('\n') && !body.ends_with('\n') {
        format!("{body}\n")
    } else {
        body
    };

    Ok((Some(frontmatter), body))
}

/// Validate a prompt name: lowercase, `[a-z0-9._-]`, ≤64 chars, non-empty.
pub(crate) fn validate_prompt_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("name is empty".into());
    }
    if name.len() > MAX_NAME_LENGTH {
        return Err(format!("name exceeds maximum length of {MAX_NAME_LENGTH}"));
    }
    for c in name.chars() {
        let ok = c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_' || c == '.';
        if !ok {
            return Err(format!(
                "name contains invalid char {c:?}; allowed: [a-z0-9._-]"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_names() {
        assert!(validate_prompt_name("whitehut").is_ok());
        assert!(validate_prompt_name("solidity-tester-minimax").is_ok());
        assert!(validate_prompt_name("whitehut-glm-5.1").is_ok());
        assert!(validate_prompt_name("whitehut-glm-5_1").is_ok());

        assert!(validate_prompt_name("").is_err());
        assert!(validate_prompt_name("Whitehut").is_err());
        assert!(validate_prompt_name("white hut").is_err());
        assert!(validate_prompt_name(&"a".repeat(65)).is_err());
    }

    #[test]
    fn split_frontmatter_none() {
        let (fm, body) =
            split_frontmatter(Path::new("x.md"), "hello\nworld\n").expect("no frontmatter");
        assert!(fm.is_none());
        assert_eq!(body, "hello\nworld\n");
    }

    #[test]
    fn split_frontmatter_present() {
        let content = "---\ndescription: short\n---\n\n# body\ncontent\n";
        let (fm, body) = split_frontmatter(Path::new("x.md"), content).expect("ok");
        assert_eq!(fm.unwrap().description.as_deref(), Some("short"));
        assert_eq!(body, "\n# body\ncontent\n");
    }

    #[test]
    fn split_frontmatter_unterminated() {
        let content = "---\ndescription: short\n(no terminator)";
        let err = split_frontmatter(Path::new("x.md"), content).unwrap_err();
        assert!(matches!(err, PromptError::Invalid { .. }));
    }
}
