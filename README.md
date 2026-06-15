# SombraX Agentic Core (`sombrax_agentic_core`)

[![crates.io](https://img.shields.io/crates/v/sombrax_agentic_core.svg)](https://crates.io/crates/sombrax_agentic_core)
[![docs.rs](https://docs.rs/sombrax_agentic_core/badge.svg)](https://docs.rs/sombrax_agentic_core)
[![license](https://img.shields.io/crates/l/sombrax_agentic_core.svg)](#license)

`sombrax_agentic_core` (SAC) is a Rust library for building tool-using LLM agents with hookable execution, resilient completion loops, and strong context management.

It is designed for coding/automation workflows where context quality and local model stability matter as much as raw model capability.
It was initially inspired by the RIG library and evolved toward stronger agent orchestration, context control, and local-model workflows.

## What Problems It Solves

| Problem | How `sombrax_agentic_core` addresses it |
|---|---|
| Agent logic becomes hard to control | `AgentBuilder` + hook chain let you intercept and modify messages, tool calls, and responses. |
| Tool loops fail on malformed/truncated model output | Built-in retry/backoff, validation retries, and rollback behavior around tool-call failures. |
| Long sessions bloat context windows | Pluggable context optimizers (`Recency`, `Priority`, `Truncation`) + configurable token budgets. |
| Repeated turns re-bill the same prompt prefix | Provider-independent prompt-cache hints keep the system+history prefix cache-stable; per-turn cache telemetry surfaces the hit ratio. |
| File-edit workflows keep stale snapshots | File-history context classification tracks `read`/`write`/`edit` and computes keep/drop/move decisions. |
| Local models need provider-specific handling | First-class local providers: `ollama`, `lmstudio`, `mlx/mlxlm`, with anti-loop controls and template handling. |
| Tool execution needs guardrails | Workspace-bounded file tools and shell safety filters for dangerous command patterns. |

## Core Capabilities

- Content-modifying hooks (`pre_completion`, `post_completion`, `pre_tool_call`, `post_tool_call`, `filter_tools`)
- Built-in tools for file/shell/web/task workflows
- MCP tool integration (`McpToolSource`)
- Cross-agent registry and shared context
- Provider-agnostic factory (`build_agent`) and provider-specific builders
- Provider-independent prompt caching with per-turn cache telemetry
- OpenTelemetry-friendly metrics/tracing integration

## Context Management (Main Focus)

### 1) Token-Budget Optimization

Use any optimizer implementing `ContextOptimizer`:
- `RecencyOptimizer`
- `PriorityOptimizer`
- `TruncationOptimizer`

```rust
use sombrax_agentic_core::context::{OptimizationConfig, RecencyOptimizer};
use sombrax_agentic_core::providers::{OpenAIClientBuilder, OpenAIClientExt};
use sombrax_agentic_core::AgentBuilder;

let client = OpenAIClientBuilder::new("api-key").build();
let model = client.completion_model_adapter("gpt-4o-mini");

let agent = AgentBuilder::new(model)
    .context_optimizer(RecencyOptimizer::new())
    .optimization_config(
        OptimizationConfig::with_budget(8_192)
            .preserve_recent(12)
    )
    .build();
```

### 2) File-History Context Classification

For coding agents, `FileContextHook` tracks file operations and computes context decisions to avoid stale file history dominating prompts.

```rust
use sombrax_agentic_core::context::classification::FileContextHook;

let file_hook = FileContextHook::new();
// Attach file_hook to your AgentBuilder via .hook(file_hook.clone())

// Later, inspect optimization signals:
let optimization = file_hook.compute_optimization();
let files_needing_read = file_hook.files_needing_read();

// optimization.keep / optimization.drop / optimization.move_to_end
```

This is especially useful when an agent reads/edits the same file repeatedly in one session.

### 3) Request and Shared State

- `HookContext`: per-request mutable context across hooks
- `SharedContext`: session-scoped shared state for multi-agent flows

### 4) Prompt Caching

Long agent loops resend a large, mostly-static prefix (system preamble + tool
definitions + prior turns) on every request. `sombrax_agentic_core` expresses
caching as a provider-independent intent — `provider::CacheHints` on each
`CompletionRequest` — so core SAC owns the semantic and each provider translates
or ignores it:

- Providers with an explicit cache protocol (Anthropic, MiniMax) translate the
  hints into `cache_control` ephemeral markers on the system block and the moving
  conversation tail.
- Implicit-prefix-cache providers (OpenAI, ZAI, OpenRouter, Cerebras, and the
  local runtimes) need no wire changes — they simply benefit from a stable,
  append-only message prefix. With caching off, the request body is byte-identical
  to before.

The agent loop computes the hints for you (cache the system+tools prefix and the
already-sent history high-water mark) and keeps that prefix byte-stable between
deliberate compaction points. It is on by default; toggle it per agent, or per
config for the factory path:

```rust
use sombrax_agentic_core::AgentBuilder;

// Default is on. Disable explicitly when you want raw, uncached requests:
let agent = AgentBuilder::new(model)
    .prompt_caching(false)
    .build();
```

`LlmConfigLike::prompt_caching() -> Option<bool>` overrides the per-provider
default in the `build_agent` path (Anthropic defaults on, MiniMax off until its
compat endpoint is probed). Cache effectiveness is observable: `ExecutionStats`
carries per-turn `Usage` (`turn_usages`, including `cache_read`/`cache_creation`
tokens) and each completion emits a `sac::cache` hit-ratio tracing line.

## Local Models (Main Focus)

`sombrax_agentic_core` supports local inference without forcing cloud APIs.

### Ollama (OpenAI-compatible endpoint)

```rust
use sombrax_agentic_core::providers::{OpenAIClientBuilder, OpenAIClientExt};
use sombrax_agentic_core::AgentBuilder;

let client = OpenAIClientBuilder::new("none")
    .base_url("http://localhost:11434/v1")
    .build();

let model = client.completion_model_adapter("qwen2.5-coder:7b");
let agent = AgentBuilder::new(model)
    .preamble("You are a precise Rust assistant.")
    .build();
```

### LM Studio (anti-repetition controls)

```rust
use sombrax_agentic_core::providers::{LmStudioClientBuilder, LmStudioClientExt};
use sombrax_agentic_core::AgentBuilder;

let client = LmStudioClientBuilder::new()
    .base_url("http://localhost:1234/v1")
    .with_anti_loop_config()
    .with_anti_repetition_stops()
    .build();

let model = client.completion_model_adapter("qwen2.5-coder-14b-instruct");
let agent = AgentBuilder::new(model).build();
```

### MLX-LM (Apple Silicon, chat-template aware)

```rust
use sombrax_agentic_core::providers::{MlxLmClientBuilder, MlxLmClientExt};
use sombrax_agentic_core::AgentBuilder;

let model_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit";

let client = MlxLmClientBuilder::new()
    .base_url("http://localhost:8080/v1")
    .auto_chat_template(model_id)
    .with_anti_loop_config()
    .with_chatml_stop_sequences()
    .with_anti_repetition_stops()
    .build();

let model = client.completion_model_adapter(model_id);
let agent = AgentBuilder::new(model).build();
```

## Provider-Agnostic Agent Construction

Use `LlmConfigLike` + `build_agent` when you want one config path for both local and cloud models.

```rust
use sombrax_agentic_core::providers::{build_agent, LlmConfigLike};

struct AppLlmConfig {
    provider: String,
    url: String,
    model: String,
    api_key: Option<String>,
    temperature: Option<f64>,
}

impl LlmConfigLike for AppLlmConfig {
    fn provider(&self) -> &str { &self.provider }
    fn url(&self) -> &str { &self.url }
    fn model(&self) -> &str { &self.model }
    fn api_key(&self) -> Option<&str> { self.api_key.as_deref() }
    fn temperature(&self) -> Option<f64> { self.temperature }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let cfg = AppLlmConfig {
    provider: "lmstudio".into(),
    url: "http://localhost:1234/v1".into(),
    model: "qwen2.5-coder-14b-instruct".into(),
    api_key: None,
    temperature: Some(0.3),
};

let agent = build_agent(&cfg, "You are a Rust assistant.", 4096, vec![]).await?;
let (content, stats) = agent.execute("Refactor this function.", &[]).await?;
println!("{}\n(total tokens: {})", content, stats.total_tokens());
# Ok(())
# }
```

Supported provider IDs include: `openrouter`, `openai`, `anthropic`, `minimax`, `cerebras`, `zai`, `ollama`, `mlx`, `lmstudio`.

## Tools and Safety

`sombrax_agentic_core::tools` includes:
- File tools: `ReadTool`, `WriteTool`, `EditTool`, `GlobTool`, `GrepTool`
- Shell tool: `BashTool` (dangerous pattern rejection)
- Web tool: `FetchTool`
- Agent tools: `TaskTool`, `TodoReadTool`, `TodoWriteTool`

Tools run inside a `ToolContext` with workspace boundary checks. `GlobTool`
additionally rejects patterns that escape the workspace (absolute, leading `~`,
or `..` components) and runs every walk under a hard 20s wall-clock timeout, so a
pathological pattern can never hang an agent.

## MCP Integration

```rust
use sombrax_agentic_core::tool::McpToolSource;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let source = McpToolSource::connect("http://localhost:3000/mcp").await?;
let _tools = source.discover().await?;
# Ok(())
# }
```

You can attach MCP tools to an agent via `AgentBuilder::mcp_tools(source).await`.

## Installation

```toml
[dependencies]
sombrax_agentic_core = "0.2"
tokio = { version = "1", features = ["full"] }
```

Optional features: `openai`, `anthropic`, and `runs` (a pluggable pipeline/job
runtime); `full` enables all three. Default build pulls in none of them.

## Examples

Runnable, provider-agnostic examples live in [`examples/`](examples/) (they
default to a local Ollama, no API key required):

```bash
cargo run --example basic_agent       # one prompt, no tools
cargo run --example react_tool_agent  # custom Tool + agentic loop
cargo run --example guardrail_hook    # content-modifying hooks
cargo run --example panel_discussion  # multiple concurrent agents
```

See [`examples/README.md`](examples/README.md) for the full list and the
`SAC_*` environment variables that point them at any supported provider.

## Development

```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo fmt --check
```

## Project Layout

- `src/agent` - agent runtime, loop execution, retries, wrapper
- `src/context` - hook/shared context + optimizers + file-history classification
- `src/hook` - hook trait, hook chain, built-in hooks
- `src/providers` - provider clients/adapters/builders (cloud + local)
- `src/tools` and `src/tool` - built-in tools and tool abstractions/MCP support

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
