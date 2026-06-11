//! Grep tool for searching file contents

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use ignore::WalkBuilder;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Maximum matches to return
const MAX_MATCHES: usize = 500;

/// Maximum context lines
const MAX_CONTEXT: usize = 10;

/// Search file contents by regex
#[derive(Clone)]
pub struct GrepTool {
    context: ToolContext,
}

impl GrepTool {
    /// Create a new grep tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for the grep tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Regex pattern to search for
    pub pattern: String,
    /// Description of what this search is looking for
    #[serde(default)]
    pub description: Option<String>,
    /// Directory or file to search in
    #[serde(default)]
    pub path: Option<String>,
    /// Glob filter for files (e.g., "*.rs")
    #[serde(default)]
    pub glob: Option<String>,
    /// Case-insensitive search
    #[serde(default)]
    pub case_insensitive: bool,
    /// Lines of context before match (max: 10)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_usize"
    )]
    pub context_before: Option<usize>,
    /// Lines of context after match (max: 10)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_usize"
    )]
    pub context_after: Option<usize>,
    /// Return only file paths, not content
    #[serde(default)]
    pub files_only: bool,
}

/// A single grep match
#[derive(Debug, Serialize)]
pub struct GrepMatch {
    /// File path
    pub file_path: String,
    /// Line number (1-indexed)
    pub line_number: usize,
    /// Line content
    pub line_content: String,
    /// Context lines before
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_before: Vec<String>,
    /// Context lines after
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_after: Vec<String>,
}

/// Output of the grep tool
#[derive(Debug, Serialize)]
pub struct GrepOutput {
    /// Individual matches (empty if files_only)
    pub matches: Vec<GrepMatch>,
    /// Files with matches
    pub files: Vec<String>,
    /// Total match count
    pub total_matches: usize,
    /// Number of files with matches
    pub files_matched: usize,
    /// Whether results were truncated
    pub truncated: bool,
    /// Files skipped due to binary content
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skipped_files: Vec<String>,
}

impl Tool for GrepTool {
    const NAME: &'static str = "grep";
    type Args = GrepArgs;
    type Output = GrepOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(GrepArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Search file contents using regex patterns.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What text am I searching for?
2. What regex pattern will match it?
3. Which directory should I search in?

## PARAMETERS

- `pattern` (REQUIRED, STRING): A regex pattern as a plain string
  CORRECT: "function\\s+withdraw"
  CORRECT: "transfer.*value"
  CORRECT: "reentrancy"
  WRONG: {"pattern": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!
  WRONG: {"glob":".","sol,":"attern"} <-- This is NOT a pattern!

- `description` (optional, STRING): Brief description of what this search is looking for
- `path` (optional, STRING): Directory to search in. Default: workspace root
- `glob` (optional, STRING): File filter like "*.sol" or "**/*.rs"
- `case_insensitive` (optional, BOOLEAN): Set true for case-insensitive search
- `context_before` / `context_after` (optional, NUMBER): Lines of context (max 10)
- `files_only` (optional, BOOLEAN): Return only file paths, not content

## EXAMPLES

Search for "withdraw" function in Solidity files:
  pattern: "withdraw"
  glob: "*.sol"

Search for reentrancy patterns:
  pattern: "call\\.value|delegatecall"
  path: "src/contracts"

Find all require statements:
  pattern: "require\\("
  glob: "*.sol"

## COMMON MISTAKES TO AVOID

1. Do NOT pass JSON objects as pattern - pattern is a STRING, not an object
2. Do NOT use {} or {"key": "value"} syntax for any parameter
3. Do NOT confuse glob (file filter) with pattern (content search)
4. Escape special regex characters: use \\. for literal dot, \\( for literal parenthesis
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "grep", pattern = %args.pattern))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let search_path = if let Some(path) = &args.path {
            self.context.validate_path(path)?
        } else {
            self.context.workspace_directory().clone()
        };

        // Build regex
        let pattern = if args.case_insensitive {
            format!("(?i){}", args.pattern)
        } else {
            args.pattern.clone()
        };
        let regex = Regex::new(&pattern)?;

        let context_before = args.context_before.unwrap_or(0).min(MAX_CONTEXT);
        let context_after = args.context_after.unwrap_or(0).min(MAX_CONTEXT);

        let mut matches = Vec::new();
        let mut files = Vec::new();
        let mut skipped_files = Vec::new();
        let mut total_matches = 0;
        let mut truncated = false;

        // Build walker with gitignore support
        let mut builder = WalkBuilder::new(&search_path);
        builder.hidden(false).git_ignore(true);

        // Add glob filter if specified
        if let Some(glob_pattern) = &args.glob {
            let mut types_builder = ignore::types::TypesBuilder::new();
            types_builder
                .add("custom", glob_pattern)
                .map_err(|e| ToolError::Validation(format!("Invalid glob: {}", e)))?;
            types_builder.select("custom");
            builder.types(
                types_builder
                    .build()
                    .map_err(|e| ToolError::Validation(format!("Glob build error: {}", e)))?,
            );
        }

        for entry in builder.build() {
            if total_matches >= MAX_MATCHES {
                truncated = true;
                break;
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Skip excluded paths
            if self.context.is_excluded(path) {
                continue;
            }

            // Search this file
            let search_result = search_file(
                path.to_path_buf(),
                &regex,
                context_before,
                context_after,
                args.files_only,
                MAX_MATCHES - total_matches,
            )?;

            match search_result {
                SearchResult::Binary => {
                    skipped_files.push(path.display().to_string());
                }
                SearchResult::Matches(file_matches) => {
                    if !file_matches.is_empty() {
                        files.push(path.display().to_string());
                        total_matches += file_matches.len();
                        if !args.files_only {
                            matches.extend(file_matches);
                        }
                    }
                }
            }
        }

        // Register found files for path resolution
        self.context.register_known_files(&files);

        Ok(GrepOutput {
            matches,
            files_matched: files.len(),
            files,
            total_matches,
            truncated,
            skipped_files,
        })
    }
}

/// Result of searching a file
enum SearchResult {
    /// File was searched and matches found
    Matches(Vec<GrepMatch>),
    /// File was skipped due to binary content
    Binary,
}

/// Check if content appears to be binary by looking for null bytes
/// in the first portion of the file
fn is_binary_file(path: &PathBuf) -> Result<bool, ToolError> {
    let file = File::open(path).map_err(ToolError::Io)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 8192];

    let bytes_read = std::io::Read::read(&mut reader, &mut buffer).map_err(ToolError::Io)?;
    let chunk = &buffer[..bytes_read];

    // Check for null bytes which indicate binary content
    Ok(chunk.contains(&0))
}

fn search_file(
    path: PathBuf,
    regex: &Regex,
    context_before: usize,
    context_after: usize,
    files_only: bool,
    max_matches: usize,
) -> Result<SearchResult, ToolError> {
    // Check if file is binary before attempting to read as text
    if is_binary_file(&path)? {
        return Ok(SearchResult::Binary);
    }

    let file = File::open(&path).map_err(ToolError::Io)?;
    let reader = BufReader::new(file);

    let mut lines = Vec::new();
    for line_result in reader.lines() {
        match line_result {
            Ok(line) => lines.push(line),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                // Invalid UTF-8 encountered, treat as binary
                return Ok(SearchResult::Binary);
            }
            Err(e) => return Err(ToolError::Io(e)),
        }
    }

    let mut matches = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if matches.len() >= max_matches {
            break;
        }

        if regex.is_match(line) {
            if files_only {
                // Just need one match to count the file
                matches.push(GrepMatch {
                    file_path: path.display().to_string(),
                    line_number: i + 1,
                    line_content: String::new(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
                break;
            }

            let ctx_before: Vec<String> = lines
                .iter()
                .skip(i.saturating_sub(context_before))
                .take(context_before.min(i))
                .cloned()
                .collect();

            let ctx_after: Vec<String> = lines
                .iter()
                .skip(i + 1)
                .take(context_after)
                .cloned()
                .collect();

            matches.push(GrepMatch {
                file_path: path.display().to_string(),
                line_number: i + 1,
                line_content: line.clone(),
                context_before: ctx_before,
                context_after: ctx_after,
            });
        }
    }

    Ok(SearchResult::Matches(matches))
}
