//! Integration tests for workspace boundary enforcement (T051)
//!
//! Tests that tools properly enforce workspace boundaries across different
//! scenarios and tool interactions.

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{
    EditArgs, EditTool, GlobArgs, GlobTool, ReadArgs, ReadTool, WriteArgs, WriteTool,
};
use sombrax_agentic_core::tools::registry::Tool;
use tempfile::TempDir;

fn create_workspace() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("boundary-test".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

/// Test that ReadTool rejects paths outside workspace
#[tokio::test]
async fn test_read_outside_workspace() {
    let (_temp_dir, ctx) = create_workspace();
    let tool = ReadTool::new(ctx);

    let result = tool
        .call(ReadArgs {
            file_path: "/etc/passwd".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("outside") || err.contains("workspace") || err.contains("boundary"));
}

/// Test that WriteTool rejects paths outside workspace
#[tokio::test]
async fn test_write_outside_workspace() {
    let (_temp_dir, ctx) = create_workspace();
    let tool = WriteTool::new(ctx);

    let result = tool
        .call(WriteArgs {
            file_path: "/tmp/malicious_file.txt".to_string(),
            content: "should not be written".to_string(),
        })
        .await;

    assert!(result.is_err());
}

/// Test that EditTool rejects paths outside workspace
#[tokio::test]
async fn test_edit_outside_workspace() {
    let (_temp_dir, ctx) = create_workspace();
    let tool = EditTool::new(ctx);

    let result = tool
        .call(EditArgs {
            file_path: "/etc/hosts".to_string(),
            old_string: "localhost".to_string(),
            new_string: "malicious".to_string(),
            replace_all: false,
        })
        .await;

    assert!(result.is_err());
}

/// Test that GlobTool respects workspace boundaries
#[tokio::test]
async fn test_glob_workspace_boundary() {
    let (temp_dir, ctx) = create_workspace();

    // Create some files in workspace
    std::fs::write(temp_dir.path().join("file1.txt"), "content1").unwrap();
    std::fs::write(temp_dir.path().join("file2.txt"), "content2").unwrap();

    let tool = GlobTool::new(ctx);

    let result = tool
        .call(GlobArgs {
            pattern: "*.txt".to_string(),
            description: None,
            path: None,
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.matches.len(), 2);
}

/// Test that path traversal attempts are blocked
#[tokio::test]
async fn test_path_traversal_blocked() {
    let (temp_dir, ctx) = create_workspace();

    // Create a file in workspace
    std::fs::write(temp_dir.path().join("safe.txt"), "safe content").unwrap();

    let tool = ReadTool::new(ctx);

    // Try path traversal
    let result = tool
        .call(ReadArgs {
            file_path: "../../../etc/passwd".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_err());
}

/// Test that symlinks outside workspace are blocked
#[tokio::test]
#[cfg(unix)]
async fn test_symlink_outside_workspace_blocked() {
    let (temp_dir, ctx) = create_workspace();

    // Create a symlink pointing outside workspace
    let symlink_path = temp_dir.path().join("external_link");
    std::os::unix::fs::symlink("/etc/passwd", &symlink_path).unwrap();

    let tool = ReadTool::new(ctx);

    let result = tool
        .call(ReadArgs {
            file_path: symlink_path.to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    // Should either reject the symlink or fail to resolve it
    // The important thing is that /etc/passwd content is not returned
    if let Ok(output) = result {
        assert!(!output.content.contains("root:"));
    }
}

/// Test nested workspace operations
#[tokio::test]
async fn test_nested_directory_operations() {
    let (temp_dir, ctx) = create_workspace();

    // Create nested structure
    let nested_dir = temp_dir.path().join("level1/level2/level3");
    std::fs::create_dir_all(&nested_dir).unwrap();
    std::fs::write(nested_dir.join("deep.txt"), "deep content").unwrap();

    let read_tool = ReadTool::new(ctx.clone());

    // Read from nested path (absolute)
    let result = read_tool
        .call(ReadArgs {
            file_path: nested_dir.join("deep.txt").to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.content.contains("deep content"));
}

/// Test workspace boundary with relative paths
#[tokio::test]
async fn test_relative_path_within_workspace() {
    let (temp_dir, ctx) = create_workspace();

    // Create a file
    std::fs::write(
        temp_dir.path().join("relative_test.txt"),
        "relative content",
    )
    .unwrap();

    let tool = ReadTool::new(ctx);

    let result = tool
        .call(ReadArgs {
            file_path: "relative_test.txt".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.content.contains("relative content"));
}

/// Test that child contexts inherit workspace boundary
#[tokio::test]
async fn test_child_context_inherits_workspace() {
    let (_temp_dir, ctx) = create_workspace();
    let workspace_path = ctx.workspace_directory().clone();

    let child = ctx.child_context("child-session".to_string()).unwrap();

    assert_eq!(child.workspace_directory(), &workspace_path);

    // Child should also reject outside paths
    let result = child.validate_path("/etc/passwd");
    assert!(result.is_err());
}

/// Test workspace boundary with multiple workspaces
#[tokio::test]
async fn test_isolated_workspaces() {
    let temp1 = TempDir::new().unwrap();
    let temp2 = TempDir::new().unwrap();

    // Create files in both workspaces
    std::fs::write(temp1.path().join("file1.txt"), "workspace1").unwrap();
    std::fs::write(temp2.path().join("file2.txt"), "workspace2").unwrap();

    let ctx1 = ToolContext::new("session1".to_string(), temp1.path().to_path_buf());
    let ctx2 = ToolContext::new("session2".to_string(), temp2.path().to_path_buf());

    let tool1 = ReadTool::new(ctx1);
    let tool2 = ReadTool::new(ctx2);

    // Tool1 can read from workspace1
    let result1 = tool1
        .call(ReadArgs {
            file_path: temp1.path().join("file1.txt").to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;
    assert!(result1.is_ok());

    // Tool1 cannot read from workspace2
    let result1_cross = tool1
        .call(ReadArgs {
            file_path: temp2.path().join("file2.txt").to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;
    assert!(result1_cross.is_err());

    // Tool2 can read from workspace2
    let result2 = tool2
        .call(ReadArgs {
            file_path: temp2.path().join("file2.txt").to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;
    assert!(result2.is_ok());
}
