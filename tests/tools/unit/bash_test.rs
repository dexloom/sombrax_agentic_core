//! Unit tests for BashTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use sombrax_agentic_core::tools::shell::{BashArgs, BashTool};
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_bash_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "bash");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_bash_tool_simple_command() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "echo 'Hello, World!'".to_string(),
            timeout: None,
            description: Some("Echo test".to_string()),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
    assert!(output.stdout.contains("Hello, World!"));
}

#[tokio::test]
async fn test_bash_tool_with_exit_code() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "exit 0".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
    assert_eq!(output.exit_code, 0);
}

#[tokio::test]
async fn test_bash_tool_failed_command() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "exit 1".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(!output.success);
    assert_eq!(output.exit_code, 1);
}

#[tokio::test]
async fn test_bash_tool_stderr() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "echo 'error' >&2".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.stderr.contains("error"));
}

#[tokio::test]
async fn test_bash_tool_working_directory() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "pwd".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    // The working directory should be the workspace
    assert!(!output.stdout.trim().is_empty());
}

#[tokio::test]
async fn test_bash_tool_with_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "echo 'quick'".to_string(),
            timeout: Some(5000), // 5 second timeout
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
}

#[tokio::test]
async fn test_bash_tool_dangerous_command_rejected() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // rm -rf / should be rejected
    let result = tool
        .call(BashArgs {
            command: "rm -rf /".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_bash_tool_multiline_output() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "echo 'line1'; echo 'line2'; echo 'line3'".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.stdout.contains("line1"));
    assert!(output.stdout.contains("line2"));
    assert!(output.stdout.contains("line3"));
}

#[tokio::test]
async fn test_bash_tool_html_entity_decoding() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Test command with HTML entities (e.g., &amp;&amp; for &&)
    let result = tool
        .call(BashArgs {
            command: "echo 'hello' &amp;&amp; echo 'world'".to_string(),
            timeout: None,
            description: Some("Test HTML entity decoding".to_string()),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
    assert!(output.stdout.contains("hello"));
    assert!(output.stdout.contains("world"));
}

#[tokio::test]
async fn test_bash_tool_html_entity_quotes() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Test command with HTML entity quotes
    let result = tool
        .call(BashArgs {
            command: "echo &quot;quoted text&quot;".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
    assert!(output.stdout.contains("quoted text"));
}

#[tokio::test]
async fn test_bash_tool_dangerous_command_with_html_entities_rejected() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Dangerous command with HTML entities should still be rejected after decoding
    let result = tool
        .call(BashArgs {
            command: "rm -rf / &amp;&amp; echo done".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_err());
}
