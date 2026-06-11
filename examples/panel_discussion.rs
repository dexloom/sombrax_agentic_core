//! Example 1 — Asynchronous panel discussion (4 agents, concurrent).
//!
//! Four agents, each with a distinct persona, discuss a topic over several rounds.
//! Within a round every agent reacts to the *same* shared transcript, and they all
//! think **at the same time**: each `execute` runs in its own `tokio` task, so a
//! round costs roughly one model round-trip instead of four sequential ones.
//!
//! This showcases:
//!   * `build_agent` building four independent agents from one config,
//!   * sharing each agent across tasks via `Arc` (agents are `Send + Sync`),
//!   * fanning N concurrent completions out with `tokio::spawn` and joining them,
//!   * accumulating `ExecutionStats` across many calls.
//!
//! Run (defaults to local Ollama):
//!   cargo run --example panel_discussion
//!   SAC_TOPIC="Should AGI research be open source?" cargo run --example panel_discussion

mod common;

use std::sync::Arc;
use std::time::Instant;

use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::{AgentWrapper, ExecutionStats};

/// One panelist: a display name and the system prompt that defines its voice.
struct Persona {
    name: &'static str,
    system: &'static str,
}

const PANEL: &[Persona] = &[
    Persona {
        name: "Optimist",
        system: "You are the Optimist on a panel. You highlight opportunities and upside. \
                 Be vivid but concise: at most 3 sentences. React to what others just said.",
    },
    Persona {
        name: "Skeptic",
        system: "You are the Skeptic on a panel. You probe risks, hidden costs and weak \
                 assumptions. Be sharp but fair: at most 3 sentences. React to the others.",
    },
    Persona {
        name: "Pragmatist",
        system: "You are the Pragmatist on a panel. You focus on what is actually doable, \
                 trade-offs and next steps. At most 3 sentences. React to the others.",
    },
    Persona {
        name: "Historian",
        system: "You are the Historian on a panel. You ground the discussion in precedent \
                 and how similar things played out before. At most 3 sentences. React to the others.",
    },
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner("Panel discussion — 4 agents, concurrent", &cfg);

    let topic = std::env::var("SAC_TOPIC")
        .unwrap_or_else(|_| "Will small local models out-compete giant cloud models?".to_string());
    let rounds: usize = std::env::var("SAC_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("Topic: {topic}\nRounds: {rounds}\n");

    // Build one agent per persona and wrap each in an Arc so it can be shared
    // across concurrently-spawned tasks. `AgentWrapper` is Send + Sync, so the
    // Arc is all we need — no locking, because `execute` only takes `&self`.
    let mut agents: Vec<Arc<AgentWrapper>> = Vec::with_capacity(PANEL.len());
    for p in PANEL {
        let agent = build_agent(&cfg, p.system, 512, vec![]).await?;
        agents.push(Arc::new(agent));
    }

    let mut transcript: Vec<(String, String)> = Vec::new(); // (speaker, text)
    let mut totals = ExecutionStats::default();
    let started = Instant::now();

    for round in 1..=rounds {
        println!("───────────────── Round {round} ─────────────────");

        // Build the prompt each panelist sees this round: the topic plus the
        // discussion so far. Everyone in the round sees the same snapshot.
        let prompt = render_prompt(&topic, &transcript);

        // Fan out: spawn every panelist's turn concurrently, then join in order
        // so the printed transcript stays stable.
        let mut handles = Vec::with_capacity(agents.len());
        for (idx, agent) in agents.iter().enumerate() {
            let agent = Arc::clone(agent);
            let prompt = prompt.clone();
            handles.push(tokio::spawn(async move {
                (idx, agent.execute(&prompt, &[]).await)
            }));
        }

        let mut round_results: Vec<(usize, String, ExecutionStats)> =
            Vec::with_capacity(handles.len());
        for handle in handles {
            let (idx, result) = handle.await?; // JoinError -> propagate
            match result {
                Ok((content, stats)) => round_results.push((idx, content, stats)),
                Err(e) => {
                    eprintln!("  [{}] failed: {e}", PANEL[idx].name);
                    return Err(e.into());
                }
            }
        }
        round_results.sort_by_key(|(idx, _, _)| *idx);

        for (idx, content, stats) in round_results {
            let name = PANEL[idx].name;
            println!("\n  {name}:\n    {}", indent(&content));
            totals.accumulate(&stats);
            transcript.push((name.to_string(), content));
        }
        println!();
    }

    println!("═════════════════════════════════════════════════════");
    println!(
        "Done in {:.1}s · {} completions · {} tokens ({} in / {} out)",
        started.elapsed().as_secs_f64(),
        totals.message_count,
        totals.total_tokens(),
        totals.input_tokens,
        totals.output_tokens,
    );
    Ok(())
}

/// Render the shared context a panelist reacts to this round.
fn render_prompt(topic: &str, transcript: &[(String, String)]) -> String {
    let mut s = format!("Panel topic: {topic}\n\n");
    if transcript.is_empty() {
        s.push_str("You are opening the discussion. Give your perspective.");
    } else {
        s.push_str("Discussion so far:\n");
        for (speaker, text) in transcript {
            s.push_str(&format!("- {speaker}: {text}\n"));
        }
        s.push_str("\nAdd your perspective, reacting to the points above.");
    }
    s
}

/// Indent multi-line model output so it nests under the speaker label.
fn indent(text: &str) -> String {
    text.trim().replace('\n', "\n    ")
}
