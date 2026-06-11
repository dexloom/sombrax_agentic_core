//! Stdio-based MCP client for spawning MCP servers as child processes.
//!
//! Complement to the HTTP-based `McpToolSource` in `mcp.rs`. Spawns an MCP server
//! binary as a subprocess and communicates via newline-delimited JSON-RPC over
//! stdin/stdout (the same protocol used by alloy_mcp, sombra_mcp, etherscan_mcp).

use crate::error::ToolError;
use crate::tool::{ToolDefinition, ToolDyn};
use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{Mutex, RwLock};

/// MCP client that spawns a server binary as a child process.
/// Communicates via newline-delimited JSON-RPC over stdin/stdout.
#[derive(Clone)]
pub struct StdioMcpClient {
    inner: Arc<Mutex<StdioMcpClientInner>>,
    tools: Arc<RwLock<Vec<ToolDefinition>>>,
    alive: Arc<AtomicBool>,
    request_id: Arc<AtomicI64>,
}

struct StdioMcpClientInner {
    _child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl StdioMcpClient {
    /// Spawn an MCP server binary as a child process.
    /// Performs `initialize` + `notifications/initialized` handshake.
    pub async fn spawn(command: &str, args: &[&str]) -> Result<Self, ToolError> {
        let mut child = tokio::process::Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ToolError::McpError(format!("Failed to spawn MCP server '{command}': {e}"))
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ToolError::McpError("Failed to capture MCP server stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolError::McpError("Failed to capture MCP server stdout".into()))?;

        // Drain stderr in background to prevent blocking
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break,
                        Ok(_) => {
                            tracing::debug!(target: "mcp_stdio::stderr", "{}", line.trim_end());
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        let client = Self {
            inner: Arc::new(Mutex::new(StdioMcpClientInner {
                _child: child,
                stdin: BufWriter::new(stdin),
                stdout: BufReader::new(stdout),
            })),
            tools: Arc::new(RwLock::new(Vec::new())),
            alive: Arc::new(AtomicBool::new(true)),
            request_id: Arc::new(AtomicI64::new(1)),
        };

        // MCP handshake
        client
            .send_request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "sac-stdio-client",
                        "version": "0.1.0"
                    }
                }),
            )
            .await?;

        client
            .send_notification("notifications/initialized", serde_json::json!({}))
            .await?;

        Ok(client)
    }

    /// Discover available tools via `tools/list`.
    pub async fn discover(&self) -> Result<Vec<ToolDefinition>, ToolError> {
        let result = self
            .send_request("tools/list", serde_json::json!({}))
            .await?;

        let tools_array = result
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let definitions: Vec<ToolDefinition> = tools_array
            .into_iter()
            .filter_map(|tool| {
                let name = tool.get("name")?.as_str()?.to_string();
                let description = tool
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let parameters = tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));
                Some(ToolDefinition {
                    name,
                    description,
                    parameters,
                })
            })
            .collect();

        *self.tools.write().await = definitions.clone();
        Ok(definitions)
    }

    /// Call a tool via `tools/call`.
    pub async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<String, ToolError> {
        let result = self
            .send_request(
                "tools/call",
                serde_json::json!({
                    "name": name,
                    "arguments": args,
                }),
            )
            .await?;

        // Check for isError
        if result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let error_text = result
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown error");
            return Err(ToolError::ExecutionFailed(error_text.to_string()));
        }

        // Extract text from content array
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| serde_json::to_string(&result).unwrap_or_default());

        Ok(text)
    }

    /// Get discovered tools as ToolDyn implementations.
    pub async fn as_tool_dyns(&self) -> Vec<Arc<dyn ToolDyn>> {
        let tools = self.tools.read().await;
        tools
            .iter()
            .map(|def| {
                Arc::new(StdioMcpTool {
                    definition: def.clone(),
                    client: self.clone(),
                }) as Arc<dyn ToolDyn>
            })
            .collect()
    }

    /// Check if the child process is alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Graceful shutdown.
    pub async fn shutdown(&self) {
        self.alive.store(false, Ordering::SeqCst);
        // Try to send a clean exit but don't fail if it errors
        let _ = self.send_notification("exit", serde_json::json!({})).await;
    }

    /// Send a JSON-RPC request and read the response.
    async fn send_request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut inner = self.inner.lock().await;

        // Write request as a single line
        let line = serde_json::to_string(&request)
            .map_err(|e| ToolError::McpError(format!("Failed to serialize request: {e}")))?;
        inner
            .stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to write to MCP server: {e}")))?;
        inner
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to write newline: {e}")))?;
        inner
            .stdin
            .flush()
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to flush: {e}")))?;

        // Read response line
        let mut response_line = String::new();
        let bytes = inner
            .stdout
            .read_line(&mut response_line)
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to read from MCP server: {e}")))?;

        if bytes == 0 {
            self.alive.store(false, Ordering::SeqCst);
            return Err(ToolError::McpError(
                "MCP server closed stdout (process died)".into(),
            ));
        }

        let response: serde_json::Value = serde_json::from_str(response_line.trim())
            .map_err(|e| ToolError::McpError(format!("Invalid JSON from MCP server: {e}")))?;

        // Check for JSON-RPC error
        if let Some(error) = response.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return Err(ToolError::McpError(format!("MCP error: {msg}")));
        }

        Ok(response
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null))
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn send_notification(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<(), ToolError> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        let mut inner = self.inner.lock().await;
        let line = serde_json::to_string(&notification)
            .map_err(|e| ToolError::McpError(format!("Failed to serialize notification: {e}")))?;
        inner
            .stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to write notification: {e}")))?;
        inner
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to write newline: {e}")))?;
        inner
            .stdin
            .flush()
            .await
            .map_err(|e| ToolError::McpError(format!("Failed to flush: {e}")))?;
        Ok(())
    }
}

/// An MCP tool backed by a stdio MCP client.
pub struct StdioMcpTool {
    definition: ToolDefinition,
    client: StdioMcpClient,
}

impl ToolDyn for StdioMcpTool {
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
            self.client.call_tool(&self.definition.name, parsed).await
        })
    }
}
