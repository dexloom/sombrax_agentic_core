//! MCP tool integration
//!
//! Provides integration with Model Context Protocol (MCP) servers for tool discovery
//! and execution using the rmcp crate.

use crate::error::ToolError;
use crate::tool::{ToolDefinition, ToolDyn};
use rmcp::{
    model::{CallToolRequestParams, Content, Tool},
    service::RunningService,
    transport::streamable_http_client::{
        StreamableHttpClientTransportConfig, StreamableHttpClientWorker,
    },
    RoleClient, ServiceExt,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

/// MCP client type alias - using () as the handler (basic client with no custom handling)
type McpClient = RunningService<RoleClient, ()>;

/// MCP tool source for connecting to MCP servers
///
/// Discovers and executes tools from an MCP server using the MCP protocol.
#[derive(Clone, Debug)]
pub struct McpToolSource {
    /// Server endpoint
    endpoint: String,
    /// Discovered tools
    tools: Arc<RwLock<Vec<ToolDefinition>>>,
    /// MCP client connection
    client: Arc<RwLock<Option<McpClient>>>,
}

impl McpToolSource {
    /// Connect to an MCP server at the given endpoint
    ///
    /// Establishes a connection using the Streamable HTTP transport and
    /// initializes the MCP protocol handshake.
    pub async fn connect(endpoint: &str) -> Result<Self, ToolError> {
        let config = StreamableHttpClientTransportConfig::with_uri(endpoint);
        let transport = StreamableHttpClientWorker::new(reqwest::Client::default(), config);

        // Use () as the handler - this is the simplest client with no custom handling
        let client: McpClient = ().serve(transport).await.map_err(|e| {
            ToolError::McpError(format!(
                "Failed to connect to MCP server at {}: {}",
                endpoint, e
            ))
        })?;

        let source = Self {
            endpoint: endpoint.to_string(),
            tools: Arc::new(RwLock::new(Vec::new())),
            client: Arc::new(RwLock::new(Some(client))),
        };

        Ok(source)
    }

    /// Check if connected to the MCP server
    pub async fn is_connected(&self) -> bool {
        self.client.read().await.is_some()
    }

    /// Discover tools from the MCP server
    ///
    /// Queries the connected MCP server for available tools and caches them.
    pub async fn discover(&self) -> Result<Vec<ToolDefinition>, ToolError> {
        let client_guard = self.client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| ToolError::McpError("Not connected to MCP server".into()))?;

        let response = client
            .list_tools(Default::default())
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to list tools: {}", e)))?;

        let definitions: Vec<ToolDefinition> = response
            .tools
            .into_iter()
            .map(|tool| convert_mcp_tool_to_definition(&tool))
            .collect();

        // Cache the discovered tools
        drop(client_guard);
        *self.tools.write().await = definitions.clone();

        Ok(definitions)
    }

    /// Get all discovered tools
    pub async fn tools(&self) -> Vec<ToolDefinition> {
        self.tools.read().await.clone()
    }

    /// Call a tool on the MCP server
    ///
    /// Executes the specified tool with the given arguments via the MCP protocol.
    pub async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<String, ToolError> {
        let client_guard = self.client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| ToolError::McpError("Not connected to MCP server".into()))?;

        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(obj) = args.as_object().cloned() {
            params = params.with_arguments(obj);
        }

        let result = client
            .call_tool(params)
            .await
            .map_err(|e| ToolError::McpError(format!("Tool call failed: {}", e)))?;

        // Check if the tool call returned an error
        if result.is_error.unwrap_or(false) {
            let error_text = extract_text_from_content(&result.content);
            return Err(ToolError::ExecutionFailed(error_text));
        }

        // Extract text content from the result
        let output = extract_text_from_content(&result.content);

        Ok(output)
    }

    /// Get the server endpoint
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Create McpTool wrappers for all discovered tools
    pub async fn as_tools(&self) -> Vec<McpTool> {
        let tools = self.tools.read().await;
        tools
            .iter()
            .map(|def| McpTool {
                definition: def.clone(),
                source: self.clone(),
            })
            .collect()
    }

    /// Disconnect from the MCP server
    pub async fn disconnect(&self) {
        let mut client_guard = self.client.write().await;
        *client_guard = None;
    }
}

/// Extract text content from MCP content items
fn extract_text_from_content(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| {
            c.as_text()
                .map(|text_content| text_content.text.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Convert an MCP Tool to our ToolDefinition
fn convert_mcp_tool_to_definition(tool: &Tool) -> ToolDefinition {
    // input_schema is Arc<Map<String, Value>> which is the JSON schema
    let parameters = serde_json::to_value(tool.input_schema.as_ref()).unwrap_or_else(|_| {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    });

    ToolDefinition {
        name: tool.name.to_string(),
        description: tool
            .description
            .clone()
            .map(|s| s.to_string())
            .unwrap_or_default(),
        parameters,
    }
}

/// An MCP tool wrapper implementing ToolDyn
#[derive(Clone)]
pub struct McpTool {
    /// Tool definition from MCP server
    pub definition: ToolDefinition,
    /// Source for calling the tool
    source: McpToolSource,
}

impl McpTool {
    /// Create a new MCP tool
    pub fn new(definition: ToolDefinition, source: McpToolSource) -> Self {
        Self { definition, source }
    }
}

impl ToolDyn for McpTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn definition<'a>(
        &'a self,
        _prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + 'a>> {
        Box::pin(async { self.definition.clone() })
    }

    fn call<'a>(
        &'a self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: serde_json::Value = serde_json::from_str(&args)?;
            self.source.call_tool(&self.definition.name, parsed).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_mcp_tool_to_definition() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        });
        let mcp_tool = Tool::new(
            "test_tool",
            "A test tool",
            Arc::new(schema.as_object().unwrap().clone()),
        );

        let def = convert_mcp_tool_to_definition(&mcp_tool);

        assert_eq!(def.name, "test_tool");
        assert_eq!(def.description, "A test tool");
        assert!(def.parameters["properties"]["input"]["type"]
            .as_str()
            .is_some());
    }

    #[test]
    fn test_convert_mcp_tool_no_description() {
        let mcp_tool = Tool::new_with_raw("simple_tool", None, Arc::new(serde_json::Map::new()));

        let def = convert_mcp_tool_to_definition(&mcp_tool);

        assert_eq!(def.name, "simple_tool");
        assert_eq!(def.description, "");
    }

    #[test]
    fn test_mcp_tool_definition() {
        let def = ToolDefinition::new(
            "mcp_test",
            "MCP test tool",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        );

        // We can't test connect without a real server, but we can test the wrapper creation
        assert_eq!(def.name, "mcp_test");
        assert_eq!(def.description, "MCP test tool");
    }

    #[test]
    fn test_extract_text_from_content() {
        let content = vec![Content::text("Hello"), Content::text("World")];
        let result = extract_text_from_content(&content);
        assert_eq!(result, "Hello\nWorld");
    }

    #[test]
    fn test_extract_text_empty() {
        let content: Vec<Content> = vec![];
        let result = extract_text_from_content(&content);
        assert_eq!(result, "");
    }
}
