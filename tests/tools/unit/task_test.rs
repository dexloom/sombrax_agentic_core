//! Unit tests for TaskTool

use sombrax_agentic_core::tools::agent::{
    AgentRuntime, SubAgentRequest, SubAgentResponse, TaskArgs, TaskTool,
};
use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use std::sync::Arc;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

/// Mock runtime for testing
struct MockRuntime {
    response: String,
}

#[async_trait::async_trait]
impl AgentRuntime for MockRuntime {
    async fn spawn_subagent(&self, request: SubAgentRequest) -> Result<SubAgentResponse, String> {
        Ok(SubAgentResponse {
            child_session_id: format!("{}-child-mock", request.parent_session_id),
            response: self.response.clone(),
            conversation_turns: Some(1),
        })
    }
}

fn create_mock_runtime() -> Arc<dyn AgentRuntime> {
    Arc::new(MockRuntime {
        response: "Mock task completed".to_string(),
    })
}

#[tokio::test]
async fn test_task_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TaskTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "task");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_task_tool_without_runtime_fails() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = TaskTool::new(ctx);

    // Without runtime, the tool should fail
    let result = tool
        .call(TaskArgs {
            prompt: "Test task".to_string(),
            description: "Test".to_string(),
            subagent_type: "unknown-type".to_string(),
            model: None,
        })
        .await;

    // Tool fails because no runtime is configured
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("No AgentRuntime is configured"));
}

#[tokio::test]
async fn test_task_tool_unknown_subagent_type_warns_but_succeeds() {
    let (_temp_dir, ctx) = create_test_context();
    let runtime = create_mock_runtime();
    let tool = TaskTool::new(ctx).with_runtime(runtime);

    // Unknown subagent_type warns but doesn't error when runtime is configured
    let result = tool
        .call(TaskArgs {
            prompt: "Test task".to_string(),
            description: "Test".to_string(),
            subagent_type: "unknown-type".to_string(),
            model: None,
        })
        .await;

    // Tool succeeds but logs a warning for unknown types
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_task_tool_valid_subagent_types() {
    let valid_types = ["explore", "plan", "general-purpose", "code-reviewer"];

    for subagent_type in valid_types {
        let temp_dir = TempDir::new().unwrap();
        let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
        let runtime = create_mock_runtime();
        let tool = TaskTool::new(ctx).with_runtime(runtime);

        let result = tool
            .call(TaskArgs {
                prompt: "Test task".to_string(),
                description: "Test".to_string(),
                subagent_type: subagent_type.to_string(),
                model: None,
            })
            .await;

        assert!(
            result.is_ok(),
            "Valid subagent type {} should succeed",
            subagent_type
        );
    }
}

#[tokio::test]
async fn test_task_tool_recursion_depth_at_limit() {
    let (_temp_dir, ctx) = create_test_context();

    // Create a context at max depth
    let mut deep_ctx = ctx;
    for i in 0..5 {
        deep_ctx = deep_ctx.child_context(format!("child-{}", i)).unwrap();
    }
    assert_eq!(deep_ctx.current_depth(), 5);

    let runtime = create_mock_runtime();
    let tool = TaskTool::new(deep_ctx).with_runtime(runtime);

    // Should reject due to max depth
    let result = tool
        .call(TaskArgs {
            prompt: "Test task".to_string(),
            description: "Test".to_string(),
            subagent_type: "general".to_string(),
            model: None,
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_task_tool_accepts_valid_args() {
    let (_temp_dir, ctx) = create_test_context();
    let runtime = create_mock_runtime();
    let tool = TaskTool::new(ctx).with_runtime(runtime);

    // Valid args with runtime configured
    let result = tool
        .call(TaskArgs {
            prompt: "Search for files".to_string(),
            description: "Find files".to_string(),
            subagent_type: "explore".to_string(),
            model: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(!output.child_session_id.is_empty());
    assert_eq!(output.response, "Mock task completed");
}

#[tokio::test]
async fn test_task_tool_with_model_override() {
    let (_temp_dir, ctx) = create_test_context();
    let runtime = create_mock_runtime();
    let tool = TaskTool::new(ctx).with_runtime(runtime);

    let result = tool
        .call(TaskArgs {
            prompt: "Background task".to_string(),
            description: "Test".to_string(),
            subagent_type: "general".to_string(),
            model: Some("haiku".to_string()),
        })
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_task_tool_has_runtime() {
    let (_temp_dir, ctx) = create_test_context();

    // Without runtime
    let tool = TaskTool::new(ctx.clone());
    assert!(!tool.has_runtime());

    // With runtime
    let runtime = create_mock_runtime();
    let tool_with_runtime = TaskTool::new(ctx).with_runtime(runtime);
    assert!(tool_with_runtime.has_runtime());
}

#[tokio::test]
async fn test_task_tool_validates_empty_description() {
    let (_temp_dir, ctx) = create_test_context();
    let runtime = create_mock_runtime();
    let tool = TaskTool::new(ctx).with_runtime(runtime);

    let result = tool
        .call(TaskArgs {
            prompt: "Test task".to_string(),
            description: "".to_string(),
            subagent_type: "general".to_string(),
            model: None,
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("description"));
}

#[tokio::test]
async fn test_task_tool_validates_empty_prompt() {
    let (_temp_dir, ctx) = create_test_context();
    let runtime = create_mock_runtime();
    let tool = TaskTool::new(ctx).with_runtime(runtime);

    let result = tool
        .call(TaskArgs {
            prompt: "".to_string(),
            description: "Test".to_string(),
            subagent_type: "general".to_string(),
            model: None,
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("prompt"));
}
