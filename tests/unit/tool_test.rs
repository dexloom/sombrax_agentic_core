//! Unit tests for Tool and ToolDefinition (T032, T033)

use serde::{Deserialize, Serialize};
use sombrax_agentic_core::error::ToolError;
use sombrax_agentic_core::tool::{into_dyn, Tool, ToolDefinition, ToolDyn};

#[test]
fn test_tool_definition_creation() {
    let def = ToolDefinition::new(
        "get_weather",
        "Get weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    );

    assert_eq!(def.name, "get_weather");
    assert_eq!(def.description, "Get weather for a location");
    assert!(def.parameters.is_object());
}

#[test]
fn test_tool_definition_serialization() {
    let def = ToolDefinition::new(
        "calculator",
        "Perform calculations",
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            }
        }),
    );

    let json = serde_json::to_string(&def).unwrap();
    assert!(json.contains("\"name\":\"calculator\""));
    assert!(json.contains("\"description\":\"Perform calculations\""));

    let parsed: ToolDefinition = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "calculator");
    assert_eq!(parsed.description, "Perform calculations");
}

#[test]
fn test_tool_definition_equality() {
    let def1 = ToolDefinition::new("test", "desc", serde_json::json!({}));
    let def2 = ToolDefinition::new("test", "desc", serde_json::json!({}));
    let def3 = ToolDefinition::new("other", "desc", serde_json::json!({}));

    assert_eq!(def1, def2);
    assert_ne!(def1, def3);
}

/// A simple test tool implementation
#[derive(Clone)]
struct AddTool;

#[derive(Deserialize)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Serialize, Deserialize)]
struct AddResult {
    sum: i32,
}

impl Tool for AddTool {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = AddResult;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition::new(
            "add",
            "Add two numbers",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                },
                "required": ["a", "b"]
            }),
        )
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    }
}

#[tokio::test]
async fn test_tool_trait_implementation() {
    let tool = AddTool;

    // Get definition
    let def = tool.definition("test prompt".to_string()).await;
    assert_eq!(def.name, "add");
    assert_eq!(def.description, "Add two numbers");

    // Call tool
    let args = AddArgs { a: 2, b: 3 };
    let result = tool.call(args).await.unwrap();
    assert_eq!(result.sum, 5);
}

#[tokio::test]
async fn test_tool_dyn_conversion() {
    let tool = AddTool;
    let dyn_tool: Box<dyn ToolDyn> = into_dyn(tool);

    assert_eq!(dyn_tool.name(), "add");

    let def = dyn_tool.definition("test".to_string()).await;
    assert_eq!(def.name, "add");
}

#[tokio::test]
async fn test_tool_dyn_call() {
    let tool = AddTool;
    let dyn_tool: Box<dyn ToolDyn> = into_dyn(tool);

    let args = r#"{"a": 10, "b": 20}"#;
    let result = dyn_tool.call(args.to_string()).await.unwrap();

    // Parse result
    let parsed: AddResult = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed.sum, 30);
}

#[tokio::test]
async fn test_tool_dyn_invalid_args() {
    let tool = AddTool;
    let dyn_tool: Box<dyn ToolDyn> = into_dyn(tool);

    let invalid_args = r#"{"invalid": true}"#;
    let result = dyn_tool.call(invalid_args.to_string()).await;

    assert!(result.is_err());
}

/// A tool that can fail
#[derive(Clone)]
struct FailingTool;

#[derive(Deserialize)]
struct FailArgs {
    should_fail: bool,
}

impl Tool for FailingTool {
    const NAME: &'static str = "failing_tool";
    type Args = FailArgs;
    type Output = String;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition::new(
            "failing_tool",
            "A tool that can fail",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "should_fail": { "type": "boolean" }
                }
            }),
        )
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.should_fail {
            Err(ToolError::ExecutionFailed(
                "Intentional failure".to_string(),
            ))
        } else {
            Ok("Success".to_string())
        }
    }
}

#[tokio::test]
async fn test_tool_error_handling() {
    let tool = FailingTool;

    // Success case
    let args = FailArgs { should_fail: false };
    let result = tool.call(args).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Success");

    // Failure case
    let args = FailArgs { should_fail: true };
    let result = tool.call(args).await;
    assert!(result.is_err());
    if let Err(ToolError::ExecutionFailed(msg)) = result {
        assert_eq!(msg, "Intentional failure");
    }
}

#[test]
fn test_tool_definition_with_complex_schema() {
    let def = ToolDefinition::new(
        "search",
        "Search for items",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "filters": {
                    "type": "object",
                    "properties": {
                        "category": { "type": "string" },
                        "min_price": { "type": "number" },
                        "max_price": { "type": "number" }
                    }
                },
                "limit": { "type": "integer", "default": 10 }
            },
            "required": ["query"]
        }),
    );

    assert!(def.parameters["properties"]["filters"]["type"] == "object");
    assert!(def.parameters["required"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("query")));
}
