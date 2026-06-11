//! Edit tool for replacing text in files

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument, Instrument};

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Edit file by replacing text
#[derive(Clone)]
pub struct EditTool {
    context: ToolContext,
}

impl EditTool {
    /// Create a new edit tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for the edit tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditArgs {
    /// Absolute path to the file to edit
    pub file_path: String,
    /// Text to find and replace
    pub old_string: String,
    /// Replacement text
    pub new_string: String,
    /// Replace all occurrences (default: false, replaces first only)
    #[serde(default)]
    pub replace_all: bool,
}

/// Output of the edit tool
#[derive(Debug, Serialize)]
pub struct EditOutput {
    /// Path to the file
    pub file_path: String,
    /// Number of replacements made
    pub replacements: usize,
    /// Original file length
    pub old_length: usize,
    /// New file length
    pub new_length: usize,
    /// Resolution info if path was auto-resolved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

impl Tool for EditTool {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = EditOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(EditArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Edit file by replacing text with exact string matching.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. Have I read the file first? (REQUIRED - tool will fail otherwise)
2. What exact text do I need to replace?
3. Is the old_string unique in the file?

IMPORTANT: You MUST read the file using the 'read' tool BEFORE calling edit.

## PARAMETERS

- `file_path` (REQUIRED, STRING): Path to the file as a plain string
  CORRECT: "src/contracts/Token.sol"
  CORRECT: "./contracts/Pool.sol"
  WRONG: {"file_path": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `old_string` (REQUIRED, STRING): Exact text to find and replace
  CORRECT: "function withdraw()"
  CORRECT: "    uint256 balance;"
  WRONG: Line numbers from read output like "   42\t..."
  WRONG: Approximate text that doesn't match exactly

- `new_string` (REQUIRED, STRING): Replacement text

- `replace_all` (optional, BOOLEAN): Replace all occurrences (default: false)

## EXAMPLES

Replace a function name:
  file_path: "src/lib.rs"
  old_string: "fn old_name()"
  new_string: "fn new_name()"

Add a new line after existing code:
  file_path: "src/main.rs"
  old_string: "use std::io;"
  new_string: "use std::io;\nuse std::fs;"

## WORKFLOW WITH READ TOOL

1. Read the file: read tool with file_path: "myfile.rs"
2. Find the text (after the tab, NOT including line numbers)
3. Copy EXACT text for old_string
4. Call edit with old_string and new_string

## COMMON MISTAKES TO AVOID

1. Do NOT call edit without reading the file first
2. Do NOT include line number prefixes (like "   42\t") in old_string
3. Do NOT use approximate text - old_string must match EXACTLY
4. If old_string appears multiple times, use replace_all=true or provide more context
5. Do NOT pass JSON objects as parameters - use plain strings
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self, args), fields(tool = "edit", file_path = %args.file_path))]
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

        let old_length = content.len();

        // Count occurrences
        let occurrence_count = content.matches(&args.old_string).count();

        if occurrence_count == 0 {
            return Err(ToolError::Validation(format!(
                "String not found in file: '{}'",
                args.old_string
            )));
        }

        // Check for uniqueness if not replace_all
        if !args.replace_all && occurrence_count > 1 {
            return Err(ToolError::Validation(format!(
                "String '{}' appears {} times. Use replace_all=true or provide more context.",
                args.old_string, occurrence_count
            )));
        }

        let (new_content, replacements) = if args.replace_all {
            (
                content.replace(&args.old_string, &args.new_string),
                occurrence_count,
            )
        } else {
            (content.replacen(&args.old_string, &args.new_string, 1), 1)
        };

        let new_length = new_content.len();

        tokio::fs::write(&file_path, &new_content)
            .instrument(info_span!("write_file"))
            .await
            .map_err(ToolError::Io)?;

        Ok(EditOutput {
            file_path: file_path.display().to_string(),
            replacements,
            old_length,
            new_length,
            resolution: resolution_msg,
        })
    }
}
