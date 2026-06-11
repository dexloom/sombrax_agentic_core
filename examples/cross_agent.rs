//! Example — cross-agent handoff (drafter ➜ editor).
//!
//! Composes two agents into a tiny pipeline: a fast "drafter" produces a first attempt,
//! then an "editor" rewrites it. The output of the first agent becomes the input of the
//! second — the simplest form of multi-agent orchestration.
//!
//! The two agents can run on **different providers/models**: the drafter reads the primary
//! `SAC_*` config and the editor reads the optional `SAC_JUDGE_*` config (falling back to
//! the primary per-field). See `common/mod.rs`. For richer multi-agent patterns see
//! `debate_judge` (cross-provider refereeing) and `panel_discussion` (concurrent agents).
//!
//! Run:
//!   cargo run --example cross_agent
//!   # drafter on Ollama, editor on OpenAI:
//!   SAC_JUDGE_PROVIDER=openai SAC_JUDGE_MODEL=gpt-4o SAC_JUDGE_API_KEY=sk-... \
//!     cargo run --example cross_agent

mod common;

use sombrax_agentic_core::providers::build_agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let drafter_cfg = common::config_from_env();
    let editor_cfg = common::judge_config_from_env();
    common::banner("Cross-agent handoff — drafter ➜ editor", &drafter_cfg);

    let topic = std::env::var("SAC_QUESTION")
        .unwrap_or_else(|_| "the benefits of writing tests".to_string());

    // Agent 1: produce a quick, unpolished draft.
    let drafter = build_agent(
        &drafter_cfg,
        "You are a fast first-draft writer. Write a single short paragraph. Do not self-edit.",
        400,
        vec![],
    )
    .await?;
    let (draft, dstats) = drafter
        .execute(&format!("Write a short paragraph about {topic}."), &[])
        .await?;
    println!(
        "--- draft (provider={}, model={}) ---\n{}\n",
        drafter_cfg.provider,
        drafter_cfg.model,
        draft.trim()
    );

    // Agent 2: take the draft as input and improve it.
    let editor = build_agent(
        &editor_cfg,
        "You are a sharp copy editor. Tighten the prose, fix any errors, and return ONLY the \
         improved paragraph — no commentary.",
        400,
        vec![],
    )
    .await?;
    let (edited, estats) = editor
        .execute(&format!("Improve this paragraph:\n\n{draft}"), &[])
        .await?;
    println!(
        "--- edited (provider={}, model={}) ---\n{}\n",
        editor_cfg.provider,
        editor_cfg.model,
        edited.trim()
    );

    println!(
        "drafter tokens: {} · editor tokens: {}",
        dstats.total_tokens(),
        estats.total_tokens(),
    );
    Ok(())
}
