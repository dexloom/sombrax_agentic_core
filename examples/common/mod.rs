//! Shared helpers for the `sac` examples.
//!
//! Every example builds its agents through [`build_agent`](sombrax_agentic_core::providers::build_agent),
//! which is provider-agnostic: it picks the right client from the `provider` string
//! of an [`LlmConfigLike`]. That means a single example runs against OpenAI, Anthropic,
//! Ollama, Minimax, ZAI, Cerebras, OpenRouter, MLX-LM or LM Studio without code changes —
//! you just point the environment variables somewhere else.
//!
//! ## Configuration
//!
//! All examples read their model config from the environment. Nothing is required:
//! the defaults target a local Ollama at `http://localhost:11434` with `llama3.2`.
//!
//! | Variable          | Default                   | Meaning                          |
//! |-------------------|---------------------------|----------------------------------|
//! | `SAC_PROVIDER`    | `ollama`                  | provider id (see table below)    |
//! | `SAC_URL`         | `http://localhost:11434`  | base URL of the API              |
//! | `SAC_MODEL`       | `llama3.2`                | model id                         |
//! | `SAC_API_KEY`     | *(unset)*                 | API key, if the provider needs one |
//! | `SAC_TEMPERATURE` | *(unset)*                 | sampling temperature (float)     |
//!
//! Provider ids: `ollama`, `openai`, `anthropic`/`claude`, `minimax`, `cerebras`,
//! `openrouter`, `zai`, `mlx`/`mlxlm`, `lmstudio`.
//!
//! Examples that use a *second* model (a supervisor, a judge) read an optional
//! `SAC_JUDGE_*` set of the same variables and fall back to the primary config when
//! a given `SAC_JUDGE_*` var is unset — so by default the same model plays both roles,
//! and you can split them across providers when you want to.
//!
//! ```text
//! # run everything against a local Ollama (default — no env needed)
//! cargo run --example panel_discussion
//!
//! # run against OpenAI, and judge with a cheaper model
//! SAC_PROVIDER=openai SAC_MODEL=gpt-4o SAC_API_KEY=sk-... \
//! SAC_JUDGE_MODEL=gpt-4o-mini \
//!   cargo run --example debate_judge
//! ```
#![allow(dead_code)]

use sombrax_agentic_core::providers::LlmConfigLike;

/// A minimal, cloneable configuration that satisfies [`LlmConfigLike`].
///
/// This is all `build_agent` needs to construct a fully-wired agent for any
/// supported provider.
#[derive(Clone, Debug)]
pub struct ExampleConfig {
    pub provider: String,
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: Option<f64>,
}

impl LlmConfigLike for ExampleConfig {
    fn provider(&self) -> &str {
        &self.provider
    }
    fn url(&self) -> &str {
        &self.url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
    fn temperature(&self) -> Option<f64> {
        self.temperature
    }
}

/// Read a non-empty environment variable, if present.
fn env_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.trim().is_empty())
}

/// Build the primary model config from `SAC_*` env vars, defaulting to local Ollama.
pub fn config_from_env() -> ExampleConfig {
    ExampleConfig {
        provider: env_opt("SAC_PROVIDER").unwrap_or_else(|| "ollama".to_string()),
        url: env_opt("SAC_URL").unwrap_or_else(|| "http://localhost:11434".to_string()),
        model: env_opt("SAC_MODEL").unwrap_or_else(|| "llama3.2".to_string()),
        api_key: env_opt("SAC_API_KEY"),
        temperature: env_opt("SAC_TEMPERATURE").and_then(|s| s.parse().ok()),
    }
}

/// Build the secondary ("judge"/"supervisor") config from `SAC_JUDGE_*` env vars,
/// falling back field-by-field to the primary config when a var is unset.
pub fn judge_config_from_env() -> ExampleConfig {
    let base = config_from_env();
    ExampleConfig {
        provider: env_opt("SAC_JUDGE_PROVIDER").unwrap_or(base.provider),
        url: env_opt("SAC_JUDGE_URL").unwrap_or(base.url),
        model: env_opt("SAC_JUDGE_MODEL").unwrap_or(base.model),
        api_key: env_opt("SAC_JUDGE_API_KEY").or(base.api_key),
        temperature: env_opt("SAC_JUDGE_TEMPERATURE")
            .and_then(|s| s.parse().ok())
            .or(base.temperature),
    }
}

/// Print a banner describing which model an example is about to use.
pub fn banner(title: &str, cfg: &ExampleConfig) {
    println!("\n{:=<70}", "");
    println!("  {title}");
    println!(
        "  primary model: provider={} model={} url={}",
        cfg.provider, cfg.model, cfg.url
    );
    println!("{:=<70}\n", "");
}

/// Initialise `tracing` so SAC's internal logs surface when `RUST_LOG` is set.
///
/// Safe to call from every example's `main`; it ignores the "already initialised"
/// error so two examples sharing a process never panic.
pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
}
