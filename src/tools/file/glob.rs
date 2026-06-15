//! Glob tool for finding files by pattern

use std::path::{Component, Path};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Maximum results to return
const MAX_RESULTS: usize = 1000;

/// Hard wall-clock bound on a single glob walk. `glob` only yields *matching*
/// paths, so the `max_results` cap never trips for a rare/zero-match pattern —
/// the iterator keeps walking the tree. A pathological pattern (e.g. an absolute
/// `/Users/**/Foo.sol` before the workspace guard below existed) once pegged a
/// core for over a day. This timeout guarantees an agent can never hang on glob.
const GLOB_TIMEOUT_SECS: u64 = 20;

/// Find files by glob pattern
#[derive(Clone)]
pub struct GlobTool {
    context: ToolContext,
}

impl GlobTool {
    /// Create a new glob tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for the glob tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GlobArgs {
    /// Glob pattern (e.g., "**/*.rs")
    pub pattern: String,
    /// Description of what files are being searched for
    #[serde(default)]
    pub description: Option<String>,
    /// Base directory for search (defaults to workspace)
    #[serde(default)]
    pub path: Option<String>,
    /// Maximum results to return (default: 1000)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_usize"
    )]
    pub max_results: Option<usize>,
}

/// Output of the glob tool
#[derive(Debug, Serialize)]
pub struct GlobOutput {
    /// Matching file paths
    pub matches: Vec<String>,
    /// Pattern used
    pub pattern: String,
    /// Whether results were truncated
    pub truncated: bool,
}

impl Tool for GlobTool {
    const NAME: &'static str = "glob";
    type Args = GlobArgs;
    type Output = GlobOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(GlobArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Find files by matching their paths against a glob pattern.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What files am I looking for?
2. What is the file extension or naming pattern?
3. Which directory should I search in?

## PARAMETERS

- `pattern` (REQUIRED, STRING): A glob pattern as a plain string
  CORRECT: "**/*.sol"
  CORRECT: "contracts/*.sol"
  CORRECT: "src/**/*.rs"
  WRONG: {"pattern": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!
  WRONG: {"glob":".","sol,":"attern"} <-- This is NOT a pattern!

- `description` (optional, STRING): Brief description of what files are being searched for
- `path` (optional, STRING): Base directory to search from (default: workspace root)
- `max_results` (optional, NUMBER): Maximum files to return (default/max: 1000)

## GLOB PATTERN SYNTAX

- "*" matches any characters within a single directory (e.g., "*.sol")
- "**" matches any characters across directories recursively (e.g., "**/*.sol")
- "?" matches exactly one character (e.g., "Token?.sol")

## EXAMPLES

Find all Solidity files:
  pattern: "**/*.sol"

Find files in contracts folder:
  pattern: "contracts/*.sol"

Find test files:
  pattern: "**/test_*.sol"

## COMMON MISTAKES TO AVOID

1. Do NOT pass JSON objects as pattern - pattern is a STRING, not an object
2. Do NOT use {} or {"key": "value"} syntax for any parameter
3. Do NOT confuse glob (file name matching) with grep (content search)
4. Use glob for finding FILES, use grep for searching TEXT inside files

NOTE: This tool finds files by NAME pattern. To search for text INSIDE files, use the 'grep' tool instead.
"#.to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "glob", pattern = %args.pattern))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let base_path = if let Some(path) = &args.path {
            self.context.validate_path(path)?
        } else {
            self.context.workspace_directory().clone()
        };

        let max_results = args.max_results.unwrap_or(MAX_RESULTS).min(MAX_RESULTS);

        // Reject patterns that escape the workspace. An absolute pattern would
        // override `base_path` in the `join` below (Rust's `Path::join` discards
        // the base when the argument is absolute) and let the walk run over the
        // whole filesystem — `/Users/**/Foo.sol` traverses all of $HOME. `..`
        // components climb out the same way, and `~` is not expanded so it can
        // only ever produce a dead path. Keep the walk inside the workspace.
        let pat = Path::new(&args.pattern);
        if pat.is_absolute()
            || args.pattern.starts_with('~')
            || pat.components().any(|c| matches!(c, Component::ParentDir))
        {
            return Err(ToolError::PathOutsideWorkspace(format!(
                "glob pattern must be relative to the workspace (no leading '/' or '~', no '..'): {}. \
                 Use a workspace-relative pattern such as \"**/*.sol\", or set a specific 'path'.",
                args.pattern
            )));
        }

        // Build full pattern
        let full_pattern = base_path.join(&args.pattern);
        let pattern_str = full_pattern.display().to_string();

        // Run the (synchronous) filesystem walk on a blocking thread under a hard
        // wall-clock bound. The workspace guard above keeps the walk within the
        // arena, and the timeout backstops any remaining pathological in-tree
        // pattern so an agent can never hang.
        let walk_pattern = pattern_str.clone();
        let walk = tokio::task::spawn_blocking(move || -> Result<(Vec<String>, bool), String> {
            let paths =
                glob::glob(&walk_pattern).map_err(|e| format!("Invalid glob pattern: {}", e))?;
            let mut matches = Vec::new();
            let mut truncated = false;
            for entry in paths {
                match entry {
                    Ok(path) => {
                        if matches.len() >= max_results {
                            truncated = true;
                            break;
                        }
                        matches.push(path.display().to_string());
                    }
                    // Skip inaccessible files
                    Err(e) => tracing::debug!("Glob error: {}", e),
                }
            }
            Ok((matches, truncated))
        });

        let (matches, truncated) =
            match tokio::time::timeout(Duration::from_secs(GLOB_TIMEOUT_SECS), walk).await {
                Ok(Ok(Ok(result))) => result,
                Ok(Ok(Err(msg))) => return Err(ToolError::Validation(msg)),
                Ok(Err(join_err)) => {
                    return Err(ToolError::Validation(format!(
                        "glob task failed: {join_err}"
                    )))
                }
                Err(_elapsed) => {
                    return Err(ToolError::Validation(format!(
                        "glob exceeded {GLOB_TIMEOUT_SECS}s and was aborted; narrow the pattern \
                     (avoid a bare '**' over the whole workspace) or set a specific 'path'."
                    )));
                }
            };

        // Filter out excluded paths
        let matches = self.context.filter_excluded(matches);

        // Sort by modification time (newest first) if possible
        let mut matches = matches;
        matches.sort_by(|a, b| {
            let a_meta = std::fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_meta = std::fs::metadata(b).and_then(|m| m.modified()).ok();
            b_meta.cmp(&a_meta)
        });

        // Register found files for path resolution
        self.context.register_known_files(&matches);

        Ok(GlobOutput {
            matches,
            pattern: args.pattern,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn tool_in(dir: &TempDir) -> GlobTool {
        let ctx = ToolContext::new("test".to_string(), dir.path().to_path_buf());
        GlobTool::new(ctx)
    }

    async fn call(tool: &GlobTool, pattern: &str) -> Result<GlobOutput, ToolError> {
        tool.call(GlobArgs {
            pattern: pattern.to_string(),
            description: None,
            path: None,
            max_results: None,
        })
        .await
    }

    #[tokio::test]
    async fn rejects_absolute_pattern_escaping_workspace() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        // The exact shape that pegged a core for >1 day in production.
        let err = call(&tool, "/Users/eugene/**/Foo.sol").await.unwrap_err();
        assert!(
            matches!(err, ToolError::PathOutsideWorkspace(_)),
            "absolute pattern must be rejected, got {err:?}"
        );
    }

    #[tokio::test]
    async fn rejects_parent_dir_and_tilde_patterns() {
        let dir = TempDir::new().unwrap();
        let tool = tool_in(&dir);
        for p in ["../**/*.sol", "../../etc/**", "~/**/*.sol"] {
            let err = call(&tool, p).await.unwrap_err();
            assert!(
                matches!(err, ToolError::PathOutsideWorkspace(_)),
                "pattern {p:?} must be rejected, got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn relative_pattern_walks_only_the_workspace() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/Token.sol"), "contract T {}").unwrap();
        fs::write(dir.path().join("README.md"), "x").unwrap();
        let tool = tool_in(&dir);

        let out = call(&tool, "**/*.sol").await.unwrap();
        assert_eq!(out.matches.len(), 1, "should find the one .sol file");
        assert!(out.matches[0].ends_with("Token.sol"));
        assert!(!out.truncated);
    }
}
