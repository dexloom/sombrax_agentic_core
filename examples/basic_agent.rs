//! Example — the simplest possible agent.
//!
//! Builds one agent through the provider-agnostic [`build_agent`] factory and sends it a
//! single prompt. No tools, no hooks — just the minimal happy path. Like every example
//! here it reads its model config from `SAC_*` env vars (see `common/mod.rs`) and defaults
//! to a local Ollama, so `cargo run --example basic_agent` works out of the box if Ollama
//! is running.
//!
//! Run:
//!   cargo run --example basic_agent
//!   SAC_PROVIDER=openai SAC_MODEL=gpt-4o SAC_API_KEY=sk-... cargo run --example basic_agent
//!
//! [`build_agent`]: sombrax_agentic_core::providers::build_agent

mod common;

use sombrax_agentic_core::providers::build_agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner("Basic agent — a single prompt, no tools", &cfg);

    // `build_agent` picks the right provider client from `cfg.provider`, applies the
    // system prompt and token budget, and returns a ready-to-use agent.
    let system = "You are a concise, helpful assistant. Answer in plain prose.";
    let agent = build_agent(&cfg, system, 512, vec![]).await?;

    let question = std::env::var("SAC_QUESTION")
        .unwrap_or_else(|_| "In two sentences, what is an LLM agent?".to_string());
    println!("Prompt: {question}\n");

    // `execute` runs the full agent loop (here just one completion, since there are no
    // tools) and returns the final answer plus `ExecutionStats`.
    let (answer, stats) = agent.execute(&question, &[]).await?;
    println!("Answer:\n  {}\n", answer.trim().replace('\n', "\n  "));
    println!(
        "completions: {} · tokens: {} (in {} / out {})",
        stats.message_count,
        stats.total_tokens(),
        stats.input_tokens,
        stats.output_tokens,
    );
    Ok(())
}
