//! Tool registry for dynamic tool management

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::error::ToolError;

/// Tool definition for LLM function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name exposed to the model.
    pub name: String,
    /// Tool description for the model.
    pub description: String,
    /// JSON schema for tool parameters.
    pub parameters: serde_json::Value,
}

/// Trait that all tools must implement
pub trait Tool: Clone + Send + Sync + 'static {
    /// Unique name for this tool
    const NAME: &'static str;

    /// Arguments type (deserialized from JSON)
    type Args: for<'de> Deserialize<'de> + Send + JsonSchema;

    /// Output type (serialized to JSON)
    type Output: Serialize;

    /// Error type for tool execution
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the tool name
    fn name(&self) -> &str {
        Self::NAME
    }

    /// Returns the tool definition for the LLM
    fn definition(&self, prompt: String) -> impl Future<Output = ToolDefinition> + Send;

    /// Execute the tool with the given arguments
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// Type-erased tool trait for dynamic dispatch
pub trait ToolDyn: Send + Sync {
    /// Returns the tool name.
    fn name(&self) -> &str;
    /// Returns the tool definition for the model.
    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + '_>>;
    /// Executes the tool with JSON arguments.
    fn call_json(
        &self,
        args: &str,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send + '_>>;
}

/// Wrapper to implement ToolDyn for any Tool
struct ToolWrapper<T: Tool> {
    tool: T,
}

impl<T: Tool<Error = ToolError>> ToolDyn for ToolWrapper<T>
where
    T::Output: 'static,
{
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + '_>> {
        Box::pin(async move { self.tool.definition(prompt).await })
    }

    fn call_json(
        &self,
        args: &str,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send + '_>> {
        let args_str = args.to_string();
        Box::pin(async move {
            let parsed_args: T::Args = serde_json::from_str(&args_str).map_err(ToolError::Json)?;
            let result = self
                .tool
                .call(parsed_args)
                .await
                .map_err(|e| ToolError::Validation(e.to_string()))?;
            serde_json::to_value(result).map_err(ToolError::Json)
        })
    }
}

/// Dynamic registry for tool management
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolDyn>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<T: Tool<Error = ToolError>>(&mut self, tool: T)
    where
        T::Output: 'static,
    {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(ToolWrapper { tool }));
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn ToolDyn>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tool names
    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Check if a tool exists
    pub fn exists(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get tool definition by name
    pub async fn definition(&self, name: &str, prompt: &str) -> Option<ToolDefinition> {
        if let Some(tool) = self.tools.get(name) {
            Some(tool.definition(prompt.to_string()).await)
        } else {
            None
        }
    }

    /// Get all tool definitions
    pub async fn all_definitions(&self, prompt: &str) -> Vec<ToolDefinition> {
        let mut definitions = Vec::new();
        for tool in self.tools.values() {
            definitions.push(tool.definition(prompt.to_string()).await);
        }
        definitions
    }

    /// Execute a tool by name with JSON arguments
    pub async fn execute(
        &self,
        name: &str,
        args_json: &str,
    ) -> Result<serde_json::Value, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::Validation(format!("Tool not found: {}", name)))?;

        tool.call_json(args_json).await
    }
}
