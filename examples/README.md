# sombrax_agentic_core examples

Runnable showcases of the `sombrax_agentic_core` (SombraX Agentic Core) library. Every example builds its
agents through the provider-agnostic [`build_agent`](../src/providers/builder.rs) factory, so a
single example runs against **any** supported provider just by changing environment variables —
no code edits.

## Configuration

All examples read their model config from the environment. Nothing is required: the defaults
target a local Ollama (`http://localhost:11434`, model `llama3.2`). See
[`common/mod.rs`](common/mod.rs) for the shared config helper.

| Variable          | Default                  | Meaning                              |
|-------------------|--------------------------|--------------------------------------|
| `SAC_PROVIDER`    | `ollama`                 | `ollama`, `openai`, `anthropic`/`claude`, `minimax`, `cerebras`, `openrouter`, `zai`, `mlx`, `lmstudio` |
| `SAC_URL`         | `http://localhost:11434` | base URL of the API                  |
| `SAC_MODEL`       | `llama3.2`               | model id                             |
| `SAC_API_KEY`     | *(unset)*                | API key, if the provider needs one   |
| `SAC_TEMPERATURE` | *(unset)*                | sampling temperature (float)         |

Examples that use a **second** model (supervisor / judge) also read an optional `SAC_JUDGE_*`
set of the same variables, falling back to `SAC_*` per-field. That lets one model do the work
while another (cheaper, or from a different provider) reviews or referees.

```bash
# default: local Ollama
cargo run --example panel_discussion

# OpenAI for the work, a cheap model as judge
SAC_PROVIDER=openai SAC_MODEL=gpt-4o SAC_API_KEY=sk-... \
SAC_JUDGE_MODEL=gpt-4o-mini \
  cargo run --example debate_judge
```

## The examples

| Example | What it demonstrates |
|---------|----------------------|
| [`basic_agent`](basic_agent.rs) | **Start here.** The minimal happy path: build one agent, send one prompt, print the answer and `ExecutionStats`. No tools, no hooks. |
| [`cross_agent`](cross_agent.rs) | **Agent handoff.** A drafter agent's output feeds an editor agent; the two can run on different providers via `SAC_JUDGE_*`. |
| [`mcp_tools`](mcp_tools.rs) | **MCP tools.** Spawns an MCP server (`StdioMcpClient`), discovers its tools, and lets the agent call them. Set `MCP_SERVER_CMD` to run. |
| [`panel_discussion`](panel_discussion.rs) | **4 agents, concurrent.** Distinct personas discuss a topic over rounds; within a round all four think at once via `tokio::spawn`, sharing each agent across tasks with `Arc`. |
| [`supervised_loop`](supervised_loop.rs) | **Worker + supervisor.** A worker advances a task step by step; a second model reviews each step using only a *minimal, model-summarized* slice of prior context. Roles can be different models. |
| [`guardrail_hook`](guardrail_hook.rs) | **Content-modifying hooks** (SAC's signature feature). One hook redacts secrets from prompts before they reach the model (`pre_completion`) and blocks dangerous tool calls (`pre_tool_call`). |
| [`react_tool_agent`](react_tool_agent.rs) | **Custom `Tool` + agentic loop.** Defines a typed `Calculator` tool, registers it with `into_arc_dyn`, and lets the agent chain tool calls to solve a multi-step problem. Needs a tool-calling model. |
| [`debate_judge`](debate_judge.rs) | **Cross-provider orchestration.** Two debaters argue concurrently each round; a judge on a possibly different provider declares a winner. |
| [`research_pipeline`](research_pipeline.rs) | **Map-reduce.** A planner splits a question into sub-questions, workers answer them concurrently (fan-out), a synthesizer merges the findings into one cited report (fan-in). |

Each prints `ExecutionStats` (completions, tokens, tool calls) so you can see what the run cost.

## Ideas for more examples

- **Self-refine loop** — a generator and a critic iterate on one artifact until the critic stops finding issues (a convergence variant of `supervised_loop`).
- **Provider race / fallback** — send the same prompt to several providers, take the first to finish (or fall back on error).
- **Context-optimizer demo** — a long conversation with a `RecencyOptimizer` + `OptimizationConfig` budget, showing token use stay flat as history grows.
- **Tournament** — bracket of agents on a task, pairwise judged, winners advance.
