//! Read tool for reading file contents

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument, Instrument};

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Read file contents with line numbers
#[derive(Clone)]
pub struct ReadTool {
    context: ToolContext,
}

impl ReadTool {
    /// Create a new read tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for the read tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// Absolute path to the file to read
    pub file_path: String,
    /// Brief description of why this file is being read and what to look for
    #[serde(default)]
    pub description: Option<String>,
    /// Starting line number (0-indexed)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_usize"
    )]
    pub offset: Option<usize>,
    /// Maximum number of lines to read
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_usize"
    )]
    pub limit: Option<usize>,
}

/// Output of the read tool
#[derive(Debug, Serialize)]
pub struct ReadOutput {
    /// File contents with line numbers
    pub content: String,
    /// Total number of lines in the file
    pub total_lines: usize,
    /// Number of lines actually read
    pub lines_read: usize,
    /// Path to the file
    pub file_path: String,
    /// Resolution info if path was auto-resolved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

impl Tool for ReadTool {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = ReadOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(ReadArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Reads file contents and returns them with line numbers.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What file do I need to read?
2. Do I have the correct file path?
3. Do I need the entire file or just a section?

IMPORTANT: You MUST read a file before using the 'edit' tool on it.

## PARAMETERS

- `file_path` (REQUIRED, STRING): Path to the file as a plain string
  CORRECT: "src/contracts/Token.sol"
  CORRECT: "./contracts/Pool.sol"
  CORRECT: "/absolute/path/to/file.rs"
  WRONG: {"file_path": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `description` (optional, STRING): Brief description of why this file is being read and what to look for
- `offset` (optional, NUMBER): Starting line number, 0-indexed
- `limit` (optional, NUMBER): Maximum lines to read

## EXAMPLES

Read a Solidity contract:
  file_path: "contracts/Token.sol"

Read lines 100-200 of a large file:
  file_path: "src/main.rs"
  offset: 100
  limit: 100

## OUTPUT FORMAT

- Each line is prefixed with its line number (1-indexed)
- Format: "   1\tcontent" where tab separates line number from content

## WHEN USING WITH EDIT TOOL

1. Read the file (or relevant section)
2. Find the text you want to change
3. Copy the EXACT text (after the tab) for old_string
4. DO NOT include the line number prefix in old_string

## COMMON MISTAKES TO AVOID

1. Do NOT pass JSON objects as file_path - file_path is a STRING
2. Do NOT use {} or {"key": "value"} syntax for any parameter
3. Do NOT include line number prefix when using with edit tool
"#.to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "read"))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let (file_path, resolution) = self
            .context
            .validate_path_with_resolution(&args.file_path)?;

        // Check if path is excluded
        if self.context.is_excluded(&file_path) {
            return Err(ToolError::Validation(format!(
                "Access denied: '{}' is in an excluded directory",
                file_path.display()
            )));
        }

        // Log resolution if it occurred
        let resolution_msg = resolution.as_ref().map(|r| {
            let msg = format!("Resolved: '{}' -> '{}'", r.original_path, r.resolved_path);
            tracing::info!(resolution = %msg, "Path resolved");
            msg
        });

        let content = tokio::fs::read_to_string(&file_path)
            .instrument(info_span!("read_file"))
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    ToolError::FileNotFound(file_path.display().to_string())
                } else {
                    ToolError::Io(e)
                }
            })?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let offset = args.offset.unwrap_or(0);
        let limit = args.limit.unwrap_or(usize::MAX);

        let selected_lines: Vec<&str> = lines.iter().skip(offset).take(limit).copied().collect();

        let lines_read = selected_lines.len();

        // Format with line numbers
        let formatted_content = selected_lines
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:6}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ReadOutput {
            content: formatted_content,
            total_lines,
            lines_read,
            file_path: file_path.display().to_string(),
            resolution: resolution_msg,
        })
    }
}
