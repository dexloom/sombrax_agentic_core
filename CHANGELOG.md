# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-15

### Added

- **Provider-independent prompt caching.** A new `provider::CacheHints`
  (`cache_system` + message `breakpoints`) on every `CompletionRequest` expresses
  cache intent without committing to a provider wire format. The Anthropic and
  MiniMax clients translate hints into `cache_control` ephemeral markers (system
  block + moving conversation tail); implicit-prefix-cache providers (OpenAI,
  ZAI, OpenRouter, Cerebras, Ollama, LM Studio, MLX-LM) ignore them and simply
  benefit from a stable, append-only prefix. With caching off the request body is
  byte-identical to before.
- **Caching controls.** `AgentBuilder::prompt_caching(bool)` (default on) and
  `LlmConfigLike::prompt_caching() -> Option<bool>` (Anthropic defaults on,
  MiniMax off) toggle it. The agent loop computes hints per request and plumbs
  the already-sent high-water mark into `OptimizationConfig::last_sent_len` so
  cache-aware optimizers keep the sent prefix byte-stable.
- **Per-turn cache telemetry.** `ExecutionStats` now carries
  `turn_usages: Vec<Usage>` (including `cache_read`/`cache_creation` tokens), and
  each completion emits a `sac::cache` hit-ratio tracing line.

### Changed

- The Anthropic and MiniMax request `system` field is now an enum
  (`Text(String)` or a cache-markable block list) instead of a plain string, to
  carry optional `cache_control`. Serialization is untagged, so an uncached
  request still serializes `system` as a plain string. (Breaking for code that
  constructed these request types directly.)

### Fixed

- **`GlobTool` workspace clamp + timeout.** Glob patterns that escape the
  workspace (absolute, leading `~`, or `..` components) are now rejected with
  `PathOutsideWorkspace` and remediation guidance, and every walk runs under a
  hard 20s wall-clock timeout — an absolute pattern (e.g. `/Users/**/Foo.sol`)
  can no longer traverse the whole filesystem or hang an agent.
- Restored `cache_read`/`cache_creation` token accounting in the provider
  adapter (previously zeroed at `Usage::new` call sites; now `Usage::with_cache`).

## [0.1.1] - 2026-06-11

### Fixed

- Resolved all rustdoc intra-doc-link and unclosed-HTML-tag warnings in the
  public API docs. `cargo doc` is now warning-free, and CI enforces it with
  `-D warnings`.

### Changed

- First release shipped through the automated GitHub Actions release pipeline
  (tag `vX.Y.Z` → publish to crates.io).

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

[Unreleased]: https://github.com/dexloom/sombrax_agentic_core/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dexloom/sombrax_agentic_core/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/dexloom/sombrax_agentic_core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dexloom/sombrax_agentic_core/releases/tag/v0.1.0
