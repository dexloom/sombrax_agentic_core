//! Unit tests for WriteTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{WriteArgs, WriteTool};
use sombrax_agentic_core::tools::registry::Tool;
use std::fs;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_write_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = WriteTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "write");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_write_tool_new_file() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("new_file.txt");

    let tool = WriteTool::new(ctx);
    let result = tool
        .call(WriteArgs {
            file_path: file_path.to_string_lossy().to_string(),
            content: "Hello, World!".to_string(),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.created);
    assert_eq!(output.bytes_written, 13);

    // Verify file contents
    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "Hello, World!");
}

#[tokio::test]
async fn test_write_tool_overwrite_file() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("existing.txt");
    fs::write(&file_path, "Original content").unwrap();

    let tool = WriteTool::new(ctx);
    let result = tool
        .call(WriteArgs {
            file_path: file_path.to_string_lossy().to_string(),
            content: "New content".to_string(),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(!output.created); // File already existed

    // Verify file was overwritten
    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "New content");
}

#[tokio::test]
async fn test_write_tool_create_in_subdirectory() {
    let (temp_dir, ctx) = create_test_context();
    let subdir = temp_dir.path().join("subdir");
    fs::create_dir(&subdir).unwrap();
    let file_path = subdir.join("file.txt");

    let tool = WriteTool::new(ctx);
    let result = tool
        .call(WriteArgs {
            file_path: file_path.to_string_lossy().to_string(),
            content: "Content in subdir".to_string(),
        })
        .await;

    assert!(result.is_ok());

    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "Content in subdir");
}

#[tokio::test]
async fn test_write_tool_multiline_content() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("multiline.txt");

    let tool = WriteTool::new(ctx);
    let content = "Line 1\nLine 2\nLine 3";
    let result = tool
        .call(WriteArgs {
            file_path: file_path.to_string_lossy().to_string(),
            content: content.to_string(),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.lines_written, 3);

    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, content);
}
