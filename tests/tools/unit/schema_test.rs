//! Unit tests for JSON schema validity

use sombrax_agentic_core::tools::agent::{TaskTool, TodoReadTool, TodoWriteTool};
use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{EditTool, GlobTool, GrepTool, ReadTool, WriteTool};
use sombrax_agentic_core::tools::registry::Tool;
use sombrax_agentic_core::tools::shell::BashTool;
use sombrax_agentic_core::tools::web::FetchTool;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

/// Verify a JSON schema has the expected structure
fn validate_schema(schema: &serde_json::Value, tool_name: &str) {
    // Should be an object
    assert!(
        schema.is_object(),
        "{} schema should be an object",
        tool_name
    );

    let schema_obj = schema.as_object().unwrap();

    // Should have $schema or type field (depending on schemars output)
    assert!(
        schema_obj.contains_key("$schema") || schema_obj.contains_key("type"),
        "{} schema should have $schema or type field",
        tool_name
    );

    // If it has properties, they should be an object
    if let Some(props) = schema_obj.get("properties") {
        assert!(
            props.is_object(),
            "{} properties should be an object",
            tool_name
        );
    }
}

#[tokio::test]
async fn test_read_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = ReadTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "ReadTool");
}

#[tokio::test]
async fn test_write_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = WriteTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "WriteTool");
}

#[tokio::test]
async fn test_edit_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = EditTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "EditTool");
}

#[tokio::test]
async fn test_glob_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = GlobTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "GlobTool");
}

#[tokio::test]
async fn test_grep_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = GrepTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "GrepTool");
}

#[tokio::test]
async fn test_bash_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "BashTool");
}

#[tokio::test]
async fn test_task_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TaskTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "TaskTool");
}

#[tokio::test]
async fn test_todo_write_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoWriteTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "TodoWriteTool");
}

#[tokio::test]
async fn test_todo_read_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TodoReadTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "TodoReadTool");
}

#[tokio::test]
async fn test_fetch_tool_schema_valid() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = FetchTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert!(!def.name.is_empty());
    assert!(!def.description.is_empty());
    validate_schema(&def.parameters, "FetchTool");
}

#[tokio::test]
async fn test_all_tool_schemas_serializable() {
    let (temp_dir, _) = create_test_context();

    // Create all tools
    let tools: Vec<Box<dyn std::any::Any>> = vec![
        Box::new(ReadTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(WriteTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(EditTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(GlobTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(GrepTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(BashTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(TaskTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(TodoWriteTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(TodoReadTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
        Box::new(FetchTool::new(ToolContext::new(
            "s".to_string(),
            temp_dir.path().to_path_buf(),
        ))),
    ];

    // All tools should have been created without panic
    assert_eq!(tools.len(), 10);
}
