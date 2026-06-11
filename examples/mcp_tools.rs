//! Example — using tools from an MCP server.
//!
//! Spawns an MCP (Model Context Protocol) server as a child process, speaks
//! newline-delimited JSON-RPC over its stdio, discovers the tools it exposes, and hands
//! those tools to an agent. The agent can then call them inside its normal execution loop.
//!
//! This example needs an external MCP server command, supplied via env vars so the example
//! stays provider- and server-agnostic:
//!
//!   `MCP_SERVER_CMD`   — the server executable (required to actually run)
//!   `MCP_SERVER_ARGS`  — optional, space-separated arguments
//!
//! Example using the reference filesystem server (Node):
//!
//!   MCP_SERVER_CMD=npx \
//!   MCP_SERVER_ARGS="-y @modelcontextprotocol/server-filesystem ." \
//!     cargo run --example mcp_tools
//!
//! Use a tool-calling-capable model (set `SAC_PROVIDER`/`SAC_MODEL` accordingly). If
//! `MCP_SERVER_CMD` is unset the example prints usage and exits cleanly, so it always
//! builds in CI.

mod common;

use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::tool::StdioMcpClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner(
        "MCP tools — discover tools from an MCP server and use them",
        &cfg,
    );

    let server_cmd = std::env::var("MCP_SERVER_CMD")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let Some(server_cmd) = server_cmd else {
        eprintln!(
            "Set MCP_SERVER_CMD to an MCP server command to run this example, e.g.:\n  \
             MCP_SERVER_CMD=npx \\\n  \
             MCP_SERVER_ARGS=\"-y @modelcontextprotocol/server-filesystem .\" \\\n  \
             cargo run --example mcp_tools"
        );
        return Ok(());
    };

    let args_str = std::env::var("MCP_SERVER_ARGS").unwrap_or_default();
    let args: Vec<&str> = args_str.split_whitespace().collect();

    println!("Spawning MCP server: {server_cmd} {}\n", args.join(" "));
    let client = StdioMcpClient::spawn(&server_cmd, &args).await?;

    // Ask the server which tools it offers (MCP `tools/list`).
    let defs = client.discover().await?;
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    println!("Discovered {} tool(s): {}\n", defs.len(), names.join(", "));

    // Wrap the discovered tools as dynamic tools the agent can invoke.
    let tools = client.as_tool_dyns().await;
    let system = "You are a helpful assistant. Use the available tools to answer the user; \
                  prefer calling a tool over guessing.";
    let agent = build_agent(&cfg, system, 800, tools).await?;

    let question = std::env::var("SAC_QUESTION").unwrap_or_else(|_| {
        "Use a tool to list what's available to you, then briefly summarize it.".to_string()
    });
    println!("Prompt: {question}\n");

    let (answer, stats) = agent.execute(&question, &[]).await?;
    println!("Answer:\n  {}\n", answer.trim().replace('\n', "\n  "));
    println!(
        "tool calls: {} · completions: {} · tokens: {}",
        stats.tool_calls,
        stats.message_count,
        stats.total_tokens(),
    );

    client.shutdown().await;
    Ok(())
}
