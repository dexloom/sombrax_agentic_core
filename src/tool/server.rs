//! Tool server for managing tools
//!
//! Provides a server component for managing and executing tools.

use crate::error::ToolError;
use crate::tool::{ToolDefinition, ToolDyn};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tool server for managing a collection of tools
///
/// Acts as a registry for tools and handles tool execution.
pub struct ToolServer {
    /// Registered tools by name
    tools: RwLock<HashMap<String, Arc<dyn ToolDyn>>>,
}

impl ToolServer {
    /// Create a new empty tool server
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool
    pub async fn register(&self, tool: Arc<dyn ToolDyn>) -> Result<(), ToolError> {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().await;

        if tools.contains_key(&name) {
            return Err(ToolError::ExecutionFailed(format!(
                "Tool '{}' already registered",
                name
            )));
        }

        tools.insert(name, tool);
        Ok(())
    }

    /// Unregister a tool
    pub async fn unregister(&self, name: &str) -> Option<Arc<dyn ToolDyn>> {
        self.tools.write().await.remove(name)
    }

    /// Get a tool by name
    pub async fn get(&self, name: &str) -> Option<Arc<dyn ToolDyn>> {
        self.tools.read().await.get(name).cloned()
    }

    /// List all tool names
    pub async fn list(&self) -> Vec<String> {
        self.tools.read().await.keys().cloned().collect()
    }

    /// Get definitions for all registered tools
    pub async fn definitions(&self, prompt: String) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        let mut defs = Vec::with_capacity(tools.len());

        for tool in tools.values() {
            defs.push(tool.definition(prompt.clone()).await);
        }

        defs
    }

    /// Call a tool by name
    pub async fn call(&self, name: &str, args: String) -> Result<String, ToolError> {
        let tool = self
            .get(name)
            .await
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        tool.call(args).await
    }

    /// Check if a tool is registered
    pub async fn has(&self, name: &str) -> bool {
        self.tools.read().await.contains_key(name)
    }

    /// Get the number of registered tools
    pub async fn len(&self) -> usize {
        self.tools.read().await.len()
    }

    /// Check if the server has no tools
    pub async fn is_empty(&self) -> bool {
        self.tools.read().await.is_empty()
    }
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolDefinition;
    use std::future::Future;
    use std::pin::Pin;

    struct MockTool {
        name: String,
    }

    impl ToolDyn for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn definition<'a>(
            &'a self,
            _prompt: String,
        ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + 'a>> {
            Box::pin(async { ToolDefinition::new(&self.name, "Mock tool", serde_json::json!({})) })
        }

        fn call<'a>(
            &'a self,
            args: String,
        ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>> {
            Box::pin(async move { Ok(format!("Called {} with {}", self.name, args)) })
        }
    }

    #[tokio::test]
    async fn test_tool_server() {
        let server = ToolServer::new();
        assert!(server.is_empty().await);

        let tool = Arc::new(MockTool {
            name: "test_tool".to_string(),
        });

        server.register(tool).await.unwrap();
        assert!(!server.is_empty().await);
        assert_eq!(server.len().await, 1);
        assert!(server.has("test_tool").await);

        let result = server.call("test_tool", "{}".to_string()).await.unwrap();
        assert!(result.contains("test_tool"));

        server.unregister("test_tool").await;
        assert!(server.is_empty().await);
    }

    #[tokio::test]
    async fn test_tool_not_found() {
        let server = ToolServer::new();

        let result = server.call("nonexistent", "{}".to_string()).await;
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }
}
