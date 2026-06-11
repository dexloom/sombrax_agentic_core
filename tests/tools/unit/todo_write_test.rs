//! Unit tests for TodoWriteTool

use sombrax_agentic_core::tools::agent::{TodoWriteArgs, TodoWriteItem, TodoWriteTool};
use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_todo_write_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "todo_write");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_todo_write_single_item() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    let result = tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Task 1".to_string(),
                status: "pending".to_string(),
            }],
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(*output.status_counts.get("pending").unwrap_or(&0), 1);
    assert_eq!(*output.status_counts.get("in_progress").unwrap_or(&0), 0);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 0);

    // Verify stored in context
    let todos = ctx.get_todos();
    assert_eq!(todos.len(), 1);
    assert_eq!(todos[0].content, "Task 1");
}

#[tokio::test]
async fn test_todo_write_multiple_items() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    let result = tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Task 1".to_string(),
                    status: "completed".to_string(),
                },
                TodoWriteItem {
                    content: "Task 2".to_string(),
                    status: "in_progress".to_string(),
                },
                TodoWriteItem {
                    content: "Task 3".to_string(),
                    status: "pending".to_string(),
                },
            ],
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.count, 3);
    assert_eq!(*output.status_counts.get("pending").unwrap_or(&0), 1);
    assert_eq!(*output.status_counts.get("in_progress").unwrap_or(&0), 1);
    assert_eq!(*output.status_counts.get("completed").unwrap_or(&0), 1);
}

#[tokio::test]
async fn test_todo_write_replaces_existing() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    // First write
    let _ = tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "Old task".to_string(),
                status: "pending".to_string(),
            }],
        })
        .await;

    // Second write should replace
    let result = tool
        .call(TodoWriteArgs {
            todos: vec![TodoWriteItem {
                content: "New task".to_string(),
                status: "in_progress".to_string(),
            }],
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.count, 1);

    let todos = ctx.get_todos();
    assert_eq!(todos.len(), 1);
    assert_eq!(todos[0].content, "New task");
}

#[tokio::test]
async fn test_todo_write_empty_list() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    let result = tool.call(TodoWriteArgs { todos: vec![] }).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.count, 0);
}

#[tokio::test]
async fn test_todo_write_freeform_status() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    let result = tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "Task 1".to_string(),
                    status: "blocked".to_string(),
                },
                TodoWriteItem {
                    content: "Task 2".to_string(),
                    status: "waiting_for_review".to_string(),
                },
                TodoWriteItem {
                    content: "Task 3".to_string(),
                    status: "custom_status".to_string(),
                },
            ],
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.count, 3);

    let todos = ctx.get_todos();
    assert_eq!(todos.len(), 3);

    // Verify freeform statuses are preserved (order may vary)
    let statuses: Vec<&str> = todos.iter().map(|t| t.status.as_str()).collect();
    assert!(statuses.contains(&"blocked"));
    assert!(statuses.contains(&"waiting_for_review"));
    assert!(statuses.contains(&"custom_status"));
}

#[tokio::test]
async fn test_todo_write_sequential_ids() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx.clone());

    let result = tool
        .call(TodoWriteArgs {
            todos: vec![
                TodoWriteItem {
                    content: "First task".to_string(),
                    status: "pending".to_string(),
                },
                TodoWriteItem {
                    content: "Second task".to_string(),
                    status: "in_progress".to_string(),
                },
                TodoWriteItem {
                    content: "Third task".to_string(),
                    status: "completed".to_string(),
                },
            ],
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Verify IDs are sequential (TASK-01, TASK-02, TASK-03)
    assert_eq!(output.todo_ids.len(), 3);
    assert_eq!(output.todo_ids[0], "TASK-01");
    assert_eq!(output.todo_ids[1], "TASK-02");
    assert_eq!(output.todo_ids[2], "TASK-03");

    // Verify todos in context have correct IDs (sorted by ID)
    let todos = ctx.get_todos();
    assert_eq!(todos.len(), 3);
    assert_eq!(todos[0].id, "TASK-01");
    assert_eq!(todos[0].content, "First task");
    assert_eq!(todos[1].id, "TASK-02");
    assert_eq!(todos[1].content, "Second task");
    assert_eq!(todos[2].id, "TASK-03");
    assert_eq!(todos[2].content, "Third task");
}
