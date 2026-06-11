//! Unit tests for EditTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{EditArgs, EditTool};
use sombrax_agentic_core::tools::registry::Tool;
use std::fs;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_edit_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = EditTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "edit");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_edit_tool_simple_replacement() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "Hello, World!").unwrap();

    let tool = EditTool::new(ctx);
    let result = tool
        .call(EditArgs {
            file_path: file_path.to_string_lossy().to_string(),
            old_string: "World".to_string(),
            new_string: "Rust".to_string(),
            replace_all: false,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.replacements, 1);

    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "Hello, Rust!");
}

#[tokio::test]
async fn test_edit_tool_replace_all() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "foo bar foo baz foo").unwrap();

    let tool = EditTool::new(ctx);
    let result = tool
        .call(EditArgs {
            file_path: file_path.to_string_lossy().to_string(),
            old_string: "foo".to_string(),
            new_string: "qux".to_string(),
            replace_all: true,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.replacements, 3);

    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "qux bar qux baz qux");
}

#[tokio::test]
async fn test_edit_tool_no_match() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "Hello, World!").unwrap();

    let tool = EditTool::new(ctx);
    let result = tool
        .call(EditArgs {
            file_path: file_path.to_string_lossy().to_string(),
            old_string: "NotFound".to_string(),
            new_string: "Replacement".to_string(),
            replace_all: false,
        })
        .await;

    // Edit tool returns error when string is not found
    assert!(result.is_err());
}

#[tokio::test]
async fn test_edit_tool_multiline_replacement() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "Line 1\nLine 2\nLine 3").unwrap();

    let tool = EditTool::new(ctx);
    let result = tool
        .call(EditArgs {
            file_path: file_path.to_string_lossy().to_string(),
            old_string: "Line 2".to_string(),
            new_string: "Modified Line".to_string(),
            replace_all: false,
        })
        .await;

    assert!(result.is_ok());

    let contents = fs::read_to_string(&file_path).unwrap();
    assert_eq!(contents, "Line 1\nModified Line\nLine 3");
}

#[tokio::test]
async fn test_edit_tool_file_not_found() {
    let (temp_dir, ctx) = create_test_context();
    let tool = EditTool::new(ctx);

    let result = tool
        .call(EditArgs {
            file_path: temp_dir
                .path()
                .join("nonexistent.txt")
                .to_string_lossy()
                .to_string(),
            old_string: "old".to_string(),
            new_string: "new".to_string(),
            replace_all: false,
        })
        .await;

    assert!(result.is_err());
}
