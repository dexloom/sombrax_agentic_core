//! Glob tool for finding files by pattern

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Maximum results to return
const MAX_RESULTS: usize = 1000;

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

        // Build full pattern
        let full_pattern = base_path.join(&args.pattern);
        let pattern_str = full_pattern.display().to_string();

        // Use glob crate
        let paths = glob::glob(&pattern_str)
            .map_err(|e| ToolError::Validation(format!("Invalid glob pattern: {}", e)))?;

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
                Err(e) => {
                    // Skip inaccessible files
                    tracing::debug!("Glob error: {}", e);
                }
            }
        }

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
