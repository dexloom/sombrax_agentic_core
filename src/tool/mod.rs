//! Tool trait and definitions
//!
//! Provides the Tool trait for defining tools that can be invoked by the LLM.

mod mcp;
pub mod mcp_stdio;
mod server;

pub use mcp::{McpTool, McpToolSource};
pub use mcp_stdio::StdioMcpClient;
pub use server::ToolServer;

use crate::error::ToolError;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// Tool definition for LLM context
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,

    /// Human-readable description for the model
    pub description: String,

    /// JSON Schema for the tool's parameters
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Create a tool definition with no parameters
    pub fn no_params(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }
}

/// Static tool trait with compile-time type safety
///
/// # Example
///
/// ```ignore
/// #[derive(Clone)]
/// struct Calculator;
///
/// #[derive(Deserialize)]
/// struct AddArgs { x: f64, y: f64 }
///
/// impl Tool for Calculator {
///     const NAME: &'static str = "calculator_add";
///     type Args = AddArgs;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///
///     async fn definition(&self, _prompt: String) -> ToolDefinition {
///         ToolDefinition {
///             name: Self::NAME.into(),
///             description: "Add two numbers".into(),
///             parameters: serde_json::json!({
///                 "type": "object",
///                 "properties": {
///                     "x": { "type": "number" },
///                     "y": { "type": "number" }
///                 },
///                 "required": ["x", "y"]
///             }),
///         }
///     }
///
///     async fn call(&self, args: AddArgs) -> Result<f64, Self::Error> {
///         Ok(args.x + args.y)
///     }
/// }
/// ```
pub trait Tool: Clone + Send + Sync + 'static {
    /// Unique name for this tool
    const NAME: &'static str;

    /// Arguments type (deserialized from JSON)
    type Args: DeserializeOwned + Send;

    /// Output type (serialized to JSON)
    type Output: Serialize;

    /// Error type for tool execution
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the tool name
    fn name(&self) -> &str {
        Self::NAME
    }

    /// Returns the tool definition for the LLM
    ///
    /// The prompt parameter can be used to tailor the definition to context.
    fn definition(&self, prompt: String) -> impl Future<Output = ToolDefinition> + Send;

    /// Execute the tool with the given arguments
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// Dynamic dispatch trait for tools (type-erased)
pub trait ToolDyn: Send + Sync {
    /// Returns the tool name
    fn name(&self) -> &str;

    /// Returns the tool definition
    fn definition<'a>(
        &'a self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + 'a>>;

    /// Execute the tool with JSON string arguments, returning JSON string result
    fn call<'a>(
        &'a self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;
}

/// Wrapper to implement ToolDyn for any Tool
struct ToolWrapper<T: Tool>(T);

impl<T: Tool> ToolDyn for ToolWrapper<T> {
    fn name(&self) -> &str {
        <T as Tool>::name(&self.0)
    }

    fn definition<'a>(
        &'a self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + 'a>> {
        Box::pin(<T as Tool>::definition(&self.0, prompt))
    }

    fn call<'a>(
        &'a self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let parsed_args: T::Args = serde_json::from_str(&args)?;
            let result = <T as Tool>::call(&self.0, parsed_args)
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
            serde_json::to_string(&result).map_err(Into::into)
        })
    }
}

/// Convert a Tool into a ToolDyn
pub fn into_dyn<T: Tool>(tool: T) -> Box<dyn ToolDyn> {
    Box::new(ToolWrapper(tool))
}

/// Convert a Tool into an `Arc<dyn ToolDyn>` for use with agents
pub fn into_arc_dyn<T: Tool>(tool: T) -> std::sync::Arc<dyn ToolDyn> {
    std::sync::Arc::new(ToolWrapper(tool))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition() {
        let def = ToolDefinition::new(
            "test_tool",
            "A test tool",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }),
        );

        assert_eq!(def.name, "test_tool");
        assert_eq!(def.description, "A test tool");
    }

    #[test]
    fn test_tool_definition_no_params() {
        let def = ToolDefinition::no_params("simple_tool", "A simple tool");

        assert_eq!(def.name, "simple_tool");
        assert!(def.parameters["type"].as_str() == Some("object"));
    }

    #[derive(Clone)]
    struct MockTool;

    #[derive(Deserialize)]
    struct MockArgs {
        value: i32,
    }

    impl Tool for MockTool {
        const NAME: &'static str = "mock_tool";
        type Args = MockArgs;
        type Output = i32;
        type Error = std::convert::Infallible;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition::new(
                Self::NAME,
                "A mock tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"}
                    }
                }),
            )
        }

        async fn call(&self, args: MockArgs) -> Result<i32, Self::Error> {
            Ok(args.value * 2)
        }
    }

    #[tokio::test]
    async fn test_tool_implementation() {
        let tool = MockTool;

        assert_eq!(tool.name(), "mock_tool");

        let def = tool.definition("test".to_string()).await;
        assert_eq!(def.name, "mock_tool");

        let result = tool.call(MockArgs { value: 21 }).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_tool_dyn() {
        let tool = MockTool;
        let dyn_tool = into_dyn(tool);

        assert_eq!(dyn_tool.name(), "mock_tool");

        let result = dyn_tool.call(r#"{"value": 21}"#.to_string()).await.unwrap();
        assert_eq!(result, "42");
    }
}
