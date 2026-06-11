//! Example 5 — Debate with a cross-provider judge.
//!
//! Two debaters argue opposite sides of a motion across several rounds; within each round
//! both speak *concurrently* (each reacting to the prior round). A third agent — the judge —
//! then reads the whole transcript and declares a winner with reasoning.
//!
//! The twist: the judge can run on a *different model/provider* than the debaters. Debaters
//! use the `SAC_*` config; the judge uses `SAC_JUDGE_*` (falling back to `SAC_*`). Because
//! every provider funnels through the same `AgentWrapper`/`execute` interface, mixing them is
//! free — a strong model debates while a neutral one referees.
//!
//! Run:
//!   cargo run --example debate_judge
//!   SAC_MOTION="This house would ban autonomous weapons" \
//!   SAC_JUDGE_PROVIDER=openai SAC_JUDGE_MODEL=gpt-4o-mini SAC_JUDGE_API_KEY=sk-... \
//!     cargo run --example debate_judge

mod common;

use std::sync::Arc;

use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::ExecutionStats;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let debater_cfg = common::config_from_env();
    let judge_cfg = common::judge_config_from_env();
    common::banner("Debate + cross-provider judge", &debater_cfg);
    println!(
        "  judge model: provider={} model={}\n",
        judge_cfg.provider, judge_cfg.model
    );

    let motion = std::env::var("SAC_MOTION").unwrap_or_else(|_| {
        "This house believes remote work is better for software teams".to_string()
    });
    let rounds: usize = std::env::var("SAC_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("Motion: {motion}\nRounds: {rounds}\n");

    let pro = Arc::new(
        build_agent(
            &debater_cfg,
            "You are the PROPOSITION in a formal debate. Argue FOR the motion persuasively. \
             At most 4 sentences per turn. Rebut the opposition's latest points.",
            512,
            vec![],
        )
        .await?,
    );
    let con = Arc::new(
        build_agent(
            &debater_cfg,
            "You are the OPPOSITION in a formal debate. Argue AGAINST the motion persuasively. \
             At most 4 sentences per turn. Rebut the proposition's latest points.",
            512,
            vec![],
        )
        .await?,
    );

    let mut transcript: Vec<(&'static str, String)> = Vec::new();
    let mut totals = ExecutionStats::default();

    for round in 1..=rounds {
        println!("───────────────── Round {round} ─────────────────");
        let prompt = render_debate(&motion, &transcript);

        // Both debaters speak at once, each reacting to the same prior transcript.
        let (pro_h, con_h) = {
            let (p, c) = (Arc::clone(&pro), Arc::clone(&con));
            let (pp, cp) = (prompt.clone(), prompt.clone());
            (
                tokio::spawn(async move { p.execute(&pp, &[]).await }),
                tokio::spawn(async move { c.execute(&cp, &[]).await }),
            )
        };
        let (pro_out, pro_stats) = pro_h.await??;
        let (con_out, con_stats) = con_h.await??;
        totals.accumulate(&pro_stats);
        totals.accumulate(&con_stats);

        println!("\n  PRO:\n    {}", indent(&pro_out));
        println!("\n  CON:\n    {}\n", indent(&con_out));
        transcript.push(("PRO", pro_out));
        transcript.push(("CON", con_out));
    }

    // The judge reads the full transcript and rules. Possibly a different provider.
    let judge = build_agent(
        &judge_cfg,
        "You are an impartial debate judge. Read the transcript and decide which side argued \
         better ON THE MERITS OF ARGUMENTATION, not your own opinion of the motion. \
         Reply with: WINNER: PRO|CON, then 2-3 sentences of justification.",
        400,
        vec![],
    )
    .await?;

    let verdict_prompt = format!(
        "Motion: {motion}\n\nTranscript:\n{}\n\nWho argued better?",
        transcript
            .iter()
            .map(|(side, text)| format!("{side}: {text}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    let (verdict, jstats) = judge.execute(&verdict_prompt, &[]).await?;
    totals.accumulate(&jstats);

    println!("═════════════════════════════════════════════════════");
    println!("Judge ({}):\n  {}\n", judge_cfg.model, indent(&verdict));
    println!(
        "{} completions · {} tokens ({} in / {} out)",
        totals.message_count,
        totals.total_tokens(),
        totals.input_tokens,
        totals.output_tokens,
    );
    Ok(())
}

fn render_debate(motion: &str, transcript: &[(&'static str, String)]) -> String {
    let mut s = format!("Debate motion: {motion}\n\n");
    if transcript.is_empty() {
        s.push_str("Opening statements. Make your case.");
    } else {
        s.push_str("Transcript so far:\n");
        for (side, text) in transcript {
            s.push_str(&format!("{side}: {text}\n"));
        }
        s.push_str("\nYour turn: rebut and advance your side.");
    }
    s
}

fn indent(text: &str) -> String {
    text.trim().replace('\n', "\n    ")
}
