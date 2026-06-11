//! Bash tool for executing shell commands

use std::process::Stdio;
use std::time::Duration;

use html_escape::decode_html_entities;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{instrument, warn};

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

use super::command_utils::split_command;
use super::safety::is_command_safe;

/// Decode HTML entities in a command string.
///
/// Commands passed through JSON serialization/deserialization or web interfaces
/// may contain HTML entities that need to be decoded before execution.
///
/// Supported entities include:
/// - `&amp;` → `&`
/// - `&lt;` → `<`
/// - `&gt;` → `>`
/// - `&quot;` → `"`
/// - `&#39;` / `&apos;` → `'`
fn decode_html_entities_in_command(command: &str) -> String {
    decode_html_entities(command).to_string()
}

/// Default timeout in milliseconds
const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// Maximum timeout in milliseconds
const MAX_TIMEOUT_MS: u64 = 600_000;

/// Maximum output size in bytes (10KB — keeps LLM context lean for verbose commands like cast logs)
const MAX_OUTPUT_SIZE: usize = 10 * 1024;

/// Execute bash commands safely
#[derive(Clone)]
pub struct BashTool {
    context: ToolContext,
    allowed_commands: Option<Vec<String>>,
    denied_patterns: Vec<String>,
}

impl BashTool {
    /// Create a new bash tool with default safety settings
    pub fn new(context: ToolContext) -> Self {
        Self {
            context,
            allowed_commands: None,
            denied_patterns: Vec::new(),
        }
    }

    /// Create with custom allowed commands (whitelist mode)
    pub fn with_allowed_commands(context: ToolContext, commands: Vec<String>) -> Self {
        Self {
            context,
            allowed_commands: Some(commands),
            denied_patterns: Vec::new(),
        }
    }

    /// Add additional denied patterns
    pub fn with_denied_patterns(mut self, patterns: Vec<String>) -> Self {
        self.denied_patterns = patterns;
        self
    }

    fn validate_command(&self, command: &str) -> Result<(), ToolError> {
        // Split compound commands on shell operators (&&, ||, ;, |) and validate each segment.
        // This prevents bypassing denied patterns by chaining (e.g., "echo hello && rm -rf /").
        let segments = split_command(command);

        for segment in &segments {
            // Check global safety per segment
            is_command_safe(segment).map_err(ToolError::CommandRejected)?;

            // Check custom denied patterns per segment
            let normalized = segment.to_lowercase();
            for pattern in &self.denied_patterns {
                if normalized.contains(&pattern.to_lowercase()) {
                    return Err(ToolError::CommandRejected(format!(
                        "Command matches denied pattern: '{}'",
                        pattern
                    )));
                }
            }

            // Check whitelist if configured — each segment must have an allowed command
            if let Some(allowed) = &self.allowed_commands {
                let first_word = segment.split_whitespace().next().unwrap_or("");
                if !allowed.iter().any(|a| first_word.starts_with(a)) {
                    return Err(ToolError::CommandRejected(format!(
                        "Command '{}' not in allowed list",
                        first_word
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Arguments for the bash tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BashArgs {
    /// Command to execute
    pub command: String,
    /// Brief description of what the command does
    #[serde(default)]
    pub description: Option<String>,
    /// Timeout in milliseconds (default: 120000, max: 600000)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_u64"
    )]
    pub timeout: Option<u64>,
}

/// Output of the bash tool
#[derive(Debug, Serialize)]
pub struct BashOutput {
    /// Command that was executed
    pub command: String,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: i32,
    /// Whether command succeeded (exit code 0)
    pub success: bool,
    /// Timeout used (ms)
    pub timeout_ms: u64,
}

impl Tool for BashTool {
    const NAME: &'static str = "bash";
    type Args = BashArgs;
    type Output = BashOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(BashArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Execute bash commands with safety validation.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What command do I need to run?
2. Is this command safe (no destructive operations)?
3. What output do I expect?

IMPORTANT: Use specialized tools instead of bash for file operations:
- Use 'read' instead of cat/head/tail
- Use 'edit' instead of sed/awk
- Use 'write' instead of echo with redirection
- Use 'glob' instead of find
- Use 'grep' instead of grep/rg

## PARAMETERS

- `command` (REQUIRED, STRING): Shell command as a plain string
  CORRECT: "cargo build --release"
  CORRECT: "git status"
  CORRECT: "npm install"
  WRONG: {"command": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `description` (optional, STRING): Brief description of what the command does

- `timeout` (optional, NUMBER): Timeout in milliseconds (default: 120000, max: 600000)

## EXAMPLES

Build a Rust project:
  command: "cargo build --release"
  description: "Build release binary"

Check git status:
  command: "git status"

Run tests:
  command: "npm test"
  timeout: 300000

Chain commands:
  command: "cargo fmt && cargo clippy"

## SAFETY NOTES

- Dangerous commands are blocked (rm -rf, sudo, etc.)
- Commands run in the workspace directory
- Output is truncated if too large (30KB limit)
- Long-running commands will timeout

## COMMON MISTAKES TO AVOID

1. Do NOT use bash for file reading (use 'read' tool)
2. Do NOT use bash for file editing (use 'edit' tool)
3. Do NOT use bash for file search (use 'glob' or 'grep' tools)
4. Do NOT pass JSON objects as command - use plain strings
5. Do NOT run destructive commands without explicit user permission
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "bash", command = %args.command))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Decode HTML entities before validation and execution
        let decoded_command = decode_html_entities_in_command(&args.command);

        // Validate the decoded command
        self.validate_command(&decoded_command)?;

        let timeout_ms = args
            .timeout
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(MAX_TIMEOUT_MS);
        let timeout_duration = Duration::from_millis(timeout_ms);

        // Execute in workspace directory using the decoded command
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(&decoded_command)
            .current_dir(self.context.workspace_directory())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(ToolError::Io)?;

        // Wait with timeout
        let result = timeout(timeout_duration, async {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();

            if let Some(mut stdout_pipe) = child.stdout.take() {
                stdout_pipe.read_to_end(&mut stdout).await.ok();
            }
            if let Some(mut stderr_pipe) = child.stderr.take() {
                stderr_pipe.read_to_end(&mut stderr).await.ok();
            }

            let status = child.wait().await?;
            Ok::<_, std::io::Error>((stdout, stderr, status))
        })
        .await;

        match result {
            Ok(Ok((stdout, stderr, status))) => {
                let exit_code = status.code().unwrap_or(-1);

                // Truncate output if too large
                let stdout_str = truncate_output(&stdout);
                let stderr_str = truncate_output(&stderr);

                Ok(BashOutput {
                    command: args.command,
                    stdout: stdout_str,
                    stderr: stderr_str,
                    exit_code,
                    success: exit_code == 0,
                    timeout_ms,
                })
            }
            Ok(Err(e)) => Err(ToolError::Io(e)),
            Err(_) => {
                // Timeout - try to kill the process
                warn!("Command timed out after {}ms, killing process", timeout_ms);
                let _ = child.kill().await;
                Err(ToolError::Timeout(timeout_ms))
            }
        }
    }
}

fn truncate_output(data: &[u8]) -> String {
    let s = String::from_utf8_lossy(data);
    if s.len() > MAX_OUTPUT_SIZE {
        let mut end = MAX_OUTPUT_SIZE;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!(
            "{}...\n[Output truncated, {} bytes total]",
            &s[..end],
            s.len()
        )
    } else {
        s.to_string()
    }
}
