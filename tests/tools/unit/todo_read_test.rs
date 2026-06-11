//! Unit tests for TodoReadTool

use sombrax_agentic_core::tools::agent::{
    TodoReadArgs, TodoReadTool, TodoWriteArgs, TodoWriteItem, TodoWriteTool,
};
use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_todo_read_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoReadTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "todo_read");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_todo_read_empty() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoReadTool::new(ctx);

    let result = tool.call(TodoReadArgs {}).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.todos.is_empty());
    assert_eq!(output.total_count, 0);
}

#[tokio::test]
async fn test_todo_read_after_write() {
    let (_temp_dir, ctx) = create_test_context();

    // First write some todos
    let write_tool = TodoWriteTool::new(ctx.clone());
    let _ = write_tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Task 1".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Task 2".to_string(),
                    status: "pending".to_string(),
                },
            ],
        })
        .await;

    // Then read
    let read_tool = TodoReadTool::new(ctx);
    let result = read_tool.call(TodoReadArgs {}).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.todos.len(), 2);
    assert_eq!(output.total_count, 2);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 1);
    assert_eq!(*output.status_counts.get("pending").unwrap_or(&0), 1);
}

#[tokio::test]
async fn test_todo_read_progress_calculation() {
    let (_temp_dir, ctx) = create_test_context();

    let write_tool = TodoWriteTool::new(ctx.clone());
    let _ = write_tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Task 1".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Task 2".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Task 3".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Task 4".to_string(),
                    status: "pending".to_string(),
                },
            ],
        })
        .await;

    let read_tool = TodoReadTool::new(ctx);
    let result = read_tool.call(TodoReadArgs {}).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_count, 4);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 3);
    assert_eq!(*output.status_counts.get("pending").unwrap_or(&0), 1);
}

#[tokio::test]
async fn test_todo_read_all_completed() {
    let (_temp_dir, ctx) = create_test_context();

    let write_tool = TodoWriteTool::new(ctx.clone());
    let _ = write_tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Task 1".to_string(),
                status: "completed".to_string(),
            }],
        })
        .await;

    let read_tool = TodoReadTool::new(ctx);
    let result = read_tool.call(TodoReadArgs {}).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_count, 1);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 1);
    assert!(output.todos[0].completed_at.is_some());
}

#[tokio::test]
async fn test_todo_read_includes_in_progress() {
    let (_temp_dir, ctx) = create_test_context();

    let write_tool = TodoWriteTool::new(ctx.clone());
    let _ = write_tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Active task".to_string(),
                status: "in_progress".to_string(),
            }],
        })
        .await;

    let read_tool = TodoReadTool::new(ctx);
    let result = read_tool.call(TodoReadArgs {}).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(*output.status_counts.get("in_progress").unwrap_or(&0), 1);
    assert_eq!(output.todos[0].status, "in_progress");
}
