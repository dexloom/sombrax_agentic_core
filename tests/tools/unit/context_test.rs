//! Unit tests for ToolContext

use sombrax_agentic_core::tools::agent::TodoItem;
use sombrax_agentic_core::tools::context::ToolContext;
use std::path::PathBuf;

#[test]
fn test_context_creation() {
    let ctx = ToolContext::new("test-session".to_string(), PathBuf::from("/workspace"));
    assert_eq!(ctx.workspace_directory(), &PathBuf::from("/workspace"));
}

#[test]
fn test_context_session_id() {
    let ctx = ToolContext::new("my-session-123".to_string(), PathBuf::from("/workspace"));
    let session_id = ctx.session_id();
    assert_eq!(session_id, "my-session-123");
}

#[test]
fn test_context_validate_path_within_workspace() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let file_path = temp_dir.path().join("file.txt");
    std::fs::write(&file_path, "test").unwrap();

    let ctx = ToolContext::new("test".to_string(), temp_dir.path().to_path_buf());
    let result = ctx.validate_path(&file_path.to_string_lossy());
    assert!(result.is_ok());
}

#[test]
fn test_context_validate_path_outside_workspace() {
    let ctx = ToolContext::new("test".to_string(), PathBuf::from("/workspace"));
    let result = ctx.validate_path("/other/file.txt");
    assert!(result.is_err());
}

#[test]
fn test_context_validate_path_relative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let subdir = temp_dir.path().join("subdir");
    std::fs::create_dir(&subdir).unwrap();
    let file_path = subdir.join("file.txt");
    std::fs::write(&file_path, "test").unwrap();

    let ctx = ToolContext::new("test".to_string(), temp_dir.path().to_path_buf());
    let result = ctx.validate_path("subdir/file.txt");
    assert!(result.is_ok());
}

#[test]
fn test_context_child_context() {
    let ctx = ToolContext::new("test".to_string(), PathBuf::from("/workspace"));
    let child = ctx.child_context("child-session".to_string()).unwrap();
    assert_eq!(child.workspace_directory(), ctx.workspace_directory());
    assert_eq!(child.current_depth(), ctx.current_depth() + 1);
}

#[test]
fn test_context_recursion_depth() {
    let ctx = ToolContext::new("test".to_string(), PathBuf::from("/workspace"));
    assert_eq!(ctx.current_depth(), 0);

    let child = ctx.child_context("child-1".to_string()).unwrap();
    assert_eq!(child.current_depth(), 1);

    let grandchild = child.child_context("child-2".to_string()).unwrap();
    assert_eq!(grandchild.current_depth(), 2);
}

#[test]
fn test_context_todo_operations() {
    let ctx = ToolContext::new("test".to_string(), PathBuf::from("/workspace"));

    // Initially empty
    let todos = ctx.get_todos();
    assert!(todos.is_empty());

    // Set todos
    let new_todos = vec![TodoItem {
        id: "1".to_string(),
        content: "Task 1".to_string(),
        status: "pending".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        completed_at: None,
    }];
    ctx.set_todos(new_todos);

    // Verify set
    let todos = ctx.get_todos();
    assert_eq!(todos.len(), 1);
    assert_eq!(todos[0].content, "Task 1");
}
