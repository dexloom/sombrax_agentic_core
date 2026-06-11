# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-10

Initial public release of SombraX Agentic Core (`sombrax_agentic_core`).

### Added

- **Agent runtime** — `Agent`/`AgentBuilder`/`AgentWrapper` with a tool-execution
  loop, retry/backoff, response validation, and `ExecutionStats`.
- **Content-modifying hooks** — the `Hook` trait and `HookChain`
  (`pre_completion`, `post_completion`, `pre_tool_call`, `post_tool_call`,
  `filter_tools`), plus built-in hooks (logging, prefix/suffix, validation,
  workspace boundary).
- **Providers** — a provider-agnostic `build_agent` factory plus first-class
  clients for OpenAI, Anthropic, MiniMax, ZAI, Cerebras, OpenRouter, and the
  local runtimes Ollama (native), LM Studio, and MLX-LM, with reasoning/thinking
  support and SSE streaming.
- **Tools** — file (`Read`/`Write`/`Edit`/`Glob`/`Grep`), shell (`Bash` with
  dangerous-pattern rejection), web (`Fetch`), and agent tools (`Task`,
  `TodoRead`/`TodoWrite`), all workspace-bounded.
- **MCP integration** — `McpToolSource` (HTTP) and `StdioMcpClient` (subprocess)
  for discovering and calling tools exposed by MCP servers.
- **Context management** — pluggable optimizers (recency/priority/truncation),
  token budgets, file-history classification, and per-request/shared context.
- **Skills & prompts** — on-disk skill and system-prompt asset discovery with a
  name-based resolution ladder.
- **`runs` runtime** (optional feature) — a pluggable pipeline/bundle/job runtime.
- **Observability** — OpenTelemetry metrics and tracing.
- **Examples** — nine runnable, provider-agnostic showcases under `examples/`.

[Unreleased]: https://github.com/dexloom/sombrax_agentic_core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dexloom/sombrax_agentic_core/releases/tag/v0.1.0
