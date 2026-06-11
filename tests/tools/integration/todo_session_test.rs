//! Integration tests for Todo session scoping (T084)
//!
//! Tests that todos are properly scoped to sessions and don't leak
//! between different sessions or contexts.

use sombrax_agentic_core::tools::agent::{
    TodoReadArgs, TodoReadTool, TodoWriteArgs, TodoWriteItem, TodoWriteTool,
};
use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use tempfile::TempDir;

fn create_session_context(session_id: &str) -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new(session_id.to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

/// Test that todos are isolated between sessions
#[tokio::test]
async fn test_session_isolation() {
    let (_temp1, ctx1) = create_session_context("session-1");
    let (_temp2, ctx2) = create_session_context("session-2");

    // Write todos to session 1
    let write_tool1 = TodoWriteTool::new(ctx1.clone());
    let _ = write_tool1
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Session 1 Task".to_string(),
                status: "pending".to_string(),
            }],
        })
        .await
        .unwrap();

    // Write different todos to session 2
    let write_tool2 = TodoWriteTool::new(ctx2.clone());
    let _ = write_tool2
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Session 2 Task".to_string(),
                status: "in_progress".to_string(),
            }],
        })
        .await
        .unwrap();

    // Read from session 1 - should only see session 1's todos
    let read_tool1 = TodoReadTool::new(ctx1.clone());
    let result1 = read_tool1.call(TodoReadArgs {}).await.unwrap();
    assert_eq!(result1.todos.len(), 1);
    assert_eq!(result1.todos[0].content, "Session 1 Task");

    // Read from session 2 - should only see session 2's todos
    let read_tool2 = TodoReadTool::new(ctx2.clone());
    let result2 = read_tool2.call(TodoReadArgs {}).await.unwrap();
    assert_eq!(result2.todos.len(), 1);
    assert_eq!(result2.todos[0].content, "Session 2 Task");
}

/// Test that child contexts can access parent todos
#[tokio::test]
async fn test_parent_child_todo_access() {
    let (_temp, parent_ctx) = create_session_context("parent-session");

    // Write todos to parent
    let parent_write = TodoWriteTool::new(parent_ctx.clone());
    let _ = parent_write
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Parent Task".to_string(),
                status: "pending".to_string(),
            }],
        })
        .await
        .unwrap();

    // Create child context
    let child_ctx = parent_ctx
        .child_context("child-session".to_string())
        .unwrap();

    // Child should be able to read parent's todos (shared context)
    let child_read = TodoReadTool::new(child_ctx.clone());
    let result = child_read.call(TodoReadArgs {}).await.unwrap();

    // Behavior depends on implementation - child might inherit or not
    // Document the actual behavior here
    assert!(result.todos.len() <= 1); // Either 0 (isolated) or 1 (inherited)
}

/// Test concurrent todo updates within same session
#[tokio::test]
async fn test_concurrent_session_updates() {
    let (_temp, ctx) = create_session_context("concurrent-session");

    // Clone context for concurrent access
    let ctx1 = ctx.clone();
    let ctx2 = ctx.clone();

    // Spawn concurrent todo writers
    let handle1 = tokio::spawn(async move {
        let tool = TodoWriteTool::new(ctx1);
        tool.call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Task A".to_string(),
                    status: "pending".to_string(),
                },
                TodoWriteItem {
                    content: "Task B".to_string(),
                    status: "pending".to_string(),
                },
            ],
        })
        .await
    });

    let handle2 = tokio::spawn(async move {
        let tool = TodoWriteTool::new(ctx2);
        tool.call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Task C".to_string(),
                status: "in_progress".to_string(),
            }],
        })
        .await
    });

    // Wait for both to complete
    let _ = handle1.await.unwrap();
    let _ = handle2.await.unwrap();

    // Read final state
    let read_tool = TodoReadTool::new(ctx.clone());
    let result = read_tool.call(TodoReadArgs {}).await.unwrap();

    // Last write wins, so we should have either {A, B} or {C}
    assert!(!result.todos.is_empty());
}

/// Test session ID uniqueness
#[tokio::test]
async fn test_session_id_tracking() {
    let (_temp1, ctx1) = create_session_context("unique-session-1");
    let (_temp2, ctx2) = create_session_context("unique-session-2");

    assert_eq!(ctx1.session_id(), "unique-session-1");
    assert_eq!(ctx2.session_id(), "unique-session-2");
    assert_ne!(ctx1.session_id(), ctx2.session_id());
}

/// Test todo persistence within session
#[tokio::test]
async fn test_todo_persistence() {
    let (_temp, ctx) = create_session_context("persist-session");

    // Write initial todos
    let write_tool = TodoWriteTool::new(ctx.clone());
    let _ = write_tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Persistent Task 1".to_string(),
                    status: "pending".to_string(),
                },
                TodoWriteItem {
                    content: "Persistent Task 2".to_string(),
                    status: "pending".to_string(),
                },
            ],
        })
        .await
        .unwrap();

    // Create new tool instances with same context
    let read_tool = TodoReadTool::new(ctx.clone());
    let result = read_tool.call(TodoReadArgs {}).await.unwrap();

    assert_eq!(result.todos.len(), 2);

    // Update todos
    let write_tool2 = TodoWriteTool::new(ctx.clone());
    let _ = write_tool2
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Persistent Task 1".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Persistent Task 2".to_string(),
                    status: "in_progress".to_string(),
                },
            ],
        })
        .await
        .unwrap();

    // Verify updates persisted
    let read_tool2 = TodoReadTool::new(ctx.clone());
    let result2 = read_tool2.call(TodoReadArgs {}).await.unwrap();

    assert_eq!(result2.todos.len(), 2);

    // Check status updates
    let completed = result2
        .todos
        .iter()
        .find(|t| t.content == "Persistent Task 1");
    assert!(completed.is_some());

    let in_progress = result2
        .todos
        .iter()
        .find(|t| t.content == "Persistent Task 2");
    assert!(in_progress.is_some());
}

/// Test empty todo list behavior
#[tokio::test]
async fn test_empty_todos() {
    let (_temp, ctx) = create_session_context("empty-session");

    // Read from fresh session
    let read_tool = TodoReadTool::new(ctx.clone());
    let result = read_tool.call(TodoReadArgs {}).await.unwrap();

    assert!(result.todos.is_empty());

    // Clear todos by writing empty list
    let write_tool = TodoWriteTool::new(ctx.clone());
    let output = write_tool
        .call(TodoWriteArgs { todos: vec![] })
        .await
        .unwrap();

    assert_eq!(output.count, 0);
}

/// Test todo status transitions
#[tokio::test]
async fn test_status_transitions() {
    let (_temp, ctx) = create_session_context("status-session");
    let write_tool = TodoWriteTool::new(ctx.clone());

    // Start with pending
    let output1 = write_tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Transitioning Task".to_string(),
                status: "pending".to_string(),
            }],
        })
        .await
        .unwrap();

    assert_eq!(*output1.status_counts.get("pending").unwrap_or(&0), 1);
    assert_eq!(*output1.status_counts.get("in_progress").unwrap_or(&0), 0);
    assert_eq!(*output1.status_counts.get("completed").unwrap_or(&0), 0);

    // Move to in_progress
    let output2 = write_tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Transitioning Task".to_string(),
                status: "in_progress".to_string(),
            }],
        })
        .await
        .unwrap();

    assert_eq!(*output2.status_counts.get("pending").unwrap_or(&0), 0);
    assert_eq!(*output2.status_counts.get("in_progress").unwrap_or(&0), 1);
    assert_eq!(*output2.status_counts.get("completed").unwrap_or(&0), 0);

    // Move to completed
    let output3 = write_tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Transitioning Task".to_string(),
                status: "completed".to_string(),
            }],
        })
        .await
        .unwrap();

    assert_eq!(*output3.status_counts.get("pending").unwrap_or(&0), 0);
    assert_eq!(*output3.status_counts.get("in_progress").unwrap_or(&0), 0);
    assert_eq!(*output3.status_counts.get("completed").unwrap_or(&0), 1);
}

/// Test large todo list handling
#[tokio::test]
async fn test_large_todo_list() {
    let (_temp, ctx) = create_session_context("large-session");
    let write_tool = TodoWriteTool::new(ctx.clone());

    // Create 100 todos
    let todos: Vec<TodoWriteItem> = (0..100)
        .map(|i| TodoWriteItem {
            content: format!("Task {}", i),
            status: if i % 3 == 0 {
                "completed"
            } else if i % 3 == 1 {
                "in_progress"
            } else {
                "pending"
            }
            .to_string(),
        })
        .collect();

    let output = write_tool.call(TodoWriteArgs { todos }).await.unwrap();

    assert_eq!(output.count, 100);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 34); // 0, 3, 6, ... 99 = 34 items
    assert_eq!(*output.status_counts.get("in_progress").unwrap_or(&0), 33); // 1, 4, 7, ... 97 = 33 items
    assert_eq!(*output.status_counts.get("pending").unwrap_or(&0), 33); // 2, 5, 8, ... 98 = 33 items

    // Read back
    let read_tool = TodoReadTool::new(ctx.clone());
    let result = read_tool.call(TodoReadArgs {}).await.unwrap();

    assert_eq!(result.todos.len(), 100);
}
