//! Unit tests for GrepTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{GrepArgs, GrepTool};
use sombrax_agentic_core::tools::registry::Tool;
use std::fs;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_grep_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = GrepTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "grep");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_grep_tool_simple_pattern() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(
        temp_dir.path().join("test.txt"),
        "Hello World\nFoo Bar\nHello Again",
    )
    .unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "Hello".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_matches, 2);
    assert_eq!(output.files_matched, 1);
}

#[tokio::test]
async fn test_grep_tool_case_insensitive() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("test.txt"), "Hello\nhello\nHELLO").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "hello".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: true,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_matches, 3);
}

#[tokio::test]
async fn test_grep_tool_with_glob_filter() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("file.txt"), "pattern here").unwrap();
    fs::write(temp_dir.path().join("file.rs"), "pattern here").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "pattern".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: Some("*.txt".to_string()),
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.files_matched, 1);
}

#[tokio::test]
async fn test_grep_tool_files_only() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(
        temp_dir.path().join("test1.txt"),
        "pattern\npattern\npattern",
    )
    .unwrap();
    fs::write(temp_dir.path().join("test2.txt"), "pattern").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "pattern".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: true,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.files_matched, 2);
}

#[tokio::test]
async fn test_grep_tool_no_matches() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("test.txt"), "Hello World").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "NotFound".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_matches, 0);
}

#[tokio::test]
async fn test_grep_tool_regex_pattern() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("test.txt"), "foo123\nbar456\nfoo789").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: r"foo\d+".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_matches, 2);
}

#[tokio::test]
async fn test_grep_tool_skips_binary_files() {
    let (temp_dir, ctx) = create_test_context();

    // Create a text file with the pattern
    fs::write(
        temp_dir.path().join("text.txt"),
        "Hello World\npattern here",
    )
    .unwrap();

    // Create a binary file with null bytes (simulates binary content)
    let binary_content = b"pattern here\x00binary data\x00more binary";
    fs::write(temp_dir.path().join("binary.bin"), binary_content).unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "pattern".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Should find match in text file only
    assert_eq!(output.total_matches, 1);
    assert_eq!(output.files_matched, 1);

    // Binary file should be in skipped_files
    assert_eq!(output.skipped_files.len(), 1);
    assert!(output.skipped_files[0].contains("binary.bin"));
}

#[tokio::test]
async fn test_grep_tool_skips_binary_with_null_at_start() {
    let (temp_dir, ctx) = create_test_context();

    // Create a binary file with null byte at the start
    let binary_content = b"\x00pattern at start".to_vec();
    fs::write(temp_dir.path().join("binary_start.bin"), &binary_content).unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "pattern".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // No matches since file is binary
    assert_eq!(output.total_matches, 0);
    assert_eq!(output.files_matched, 0);

    // Binary file should be reported as skipped
    assert_eq!(output.skipped_files.len(), 1);
}

#[tokio::test]
async fn test_grep_tool_reports_multiple_binary_files() {
    let (temp_dir, ctx) = create_test_context();

    // Create multiple binary files
    fs::write(temp_dir.path().join("bin1.dat"), b"\x00binary1").unwrap();
    fs::write(temp_dir.path().join("bin2.dat"), b"text\x00binary2").unwrap();

    // Create a text file
    fs::write(temp_dir.path().join("text.txt"), "pattern match").unwrap();

    let tool = GrepTool::new(ctx);
    let result = tool
        .call(GrepArgs {
            pattern: "pattern|binary".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            glob: None,
            case_insensitive: false,
            context_before: None,
            context_after: None,
            files_only: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Only text file should have matches
    assert_eq!(output.total_matches, 1);
    assert_eq!(output.files_matched, 1);

    // Both binary files should be skipped
    assert_eq!(output.skipped_files.len(), 2);
}
