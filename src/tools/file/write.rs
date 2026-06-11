//! Write tool for writing file contents

use std::io::Write;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use tracing::{info_span, instrument, Instrument};

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// Write content to a file
#[derive(Clone)]
pub struct WriteTool {
    context: ToolContext,
}

impl WriteTool {
    /// Create a new write tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for the write tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteArgs {
    /// Absolute path to the file to write
    pub file_path: String,
    /// Complete file content
    pub content: String,
}

/// Output of the write tool
#[derive(Debug, Serialize)]
pub struct WriteOutput {
    /// Path to the file
    pub file_path: String,
    /// Number of bytes written
    pub bytes_written: usize,
    /// Number of lines written
    pub lines_written: usize,
    /// Whether the file was created (vs. overwritten)
    pub created: bool,
    /// Resolution info if path was auto-resolved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

impl Tool for WriteTool {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = WriteOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(WriteArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Write complete content to a file (creates new or overwrites existing).

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. Does this file already exist? If yes, use 'edit' tool instead for targeted changes.
2. Do I have the COMPLETE content to write?
3. Is the parent directory valid?

IMPORTANT: This tool OVERWRITES the entire file. Prefer 'edit' for modifying existing files.

## PARAMETERS

- `file_path` (REQUIRED, STRING): Path to the file as a plain string
  CORRECT: "src/new_module.rs"
  CORRECT: "./contracts/NewToken.sol"
  CORRECT: "/absolute/path/to/file.txt"
  WRONG: {"file_path": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `content` (REQUIRED, STRING): Complete file content to write
  The entire content that should be in the file after writing.

## EXAMPLES

Create a new Rust module:
  file_path: "src/utils/helpers.rs"
  content: "pub fn helper() -> bool {\n    true\n}\n"

Create a configuration file:
  file_path: "config.toml"
  content: "[settings]\ndebug = true\n"

## WHEN TO USE THIS TOOL

- Creating NEW files that don't exist
- Completely replacing file content (rare)
- Writing generated content to a new location

## WHEN NOT TO USE THIS TOOL

- Modifying existing files (use 'edit' instead)
- Making targeted changes (use 'edit' instead)
- Appending to files (use 'edit' instead)

## COMMON MISTAKES TO AVOID

1. Do NOT use write to modify existing files - use 'edit' instead
2. Do NOT pass JSON objects as parameters - use plain strings
3. Do NOT forget to include newlines where needed in content
4. Parent directories will be created automatically if needed
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self, args), fields(tool = "write", file_path = %args.file_path))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Validate write path with resolution fallback (checks parent dir exists)
        let (file_path, resolution) = self
            .context
            .validate_write_path_with_resolution(&args.file_path)?;

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

        let created = !file_path.exists();
        let bytes_written = args.content.len();
        let lines_written = args.content.lines().count();

        // Create parent directories if needed
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .instrument(info_span!("create_dirs"))
                .await
                .map_err(ToolError::Io)?;
        }

        // Atomic write: temp file + fsync + rename
        // Create temp file in the same directory to ensure atomic rename works
        // (rename across filesystems is not atomic)
        let parent_dir = file_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let content = args.content.clone();
        let target_path = file_path.clone();

        tokio::task::spawn_blocking(move || {
            let _span = info_span!("atomic_write").entered();

            // Create temp file in the same directory as the target
            let mut temp_file = NamedTempFile::new_in(&parent_dir).map_err(ToolError::Io)?;

            // Write content to temp file
            temp_file
                .write_all(content.as_bytes())
                .map_err(ToolError::Io)?;

            // Sync data to disk (fsync)
            temp_file.as_file().sync_all().map_err(ToolError::Io)?;

            // Atomically rename temp file to target path
            temp_file
                .persist(&target_path)
                .map_err(|e| ToolError::Io(e.error))?;

            Ok::<(), ToolError>(())
        })
        .await
        .map_err(|e| ToolError::Io(std::io::Error::other(e)))??;

        Ok(WriteOutput {
            file_path: file_path.display().to_string(),
            bytes_written,
            lines_written,
            created,
            resolution: resolution_msg,
        })
    }
}
