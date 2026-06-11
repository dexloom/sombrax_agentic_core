//! Example 6 — Map-reduce research pipeline.
//!
//! Three stages, classic fan-out / fan-in:
//!
//! 1. PLAN — a planner agent breaks a question into N independent sub-questions.
//! 2. MAP — one worker agent answers every sub-question *concurrently* (the same
//!    agent, shared via `Arc`, handling many prompts at once).
//! 3. REDUCE — a synthesizer agent merges the partial answers into one cited report.
//!
//! This showcases driving a real pipeline with SAC: concurrent fan-out over a shared agent,
//! then a reduce step that consumes the collected outputs. (For tasks where the reduce step
//! should *continue* a worker's conversation rather than start fresh, `AgentWrapper` also
//! exposes `execute_with_messages`, which returns the full message history to thread forward.)
//!
//! Run:
//!   cargo run --example research_pipeline
//!   SAC_QUESTION="What makes a good incident postmortem?" cargo run --example research_pipeline

mod common;

use std::sync::Arc;

use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::ExecutionStats;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner("Map-reduce research pipeline", &cfg);

    let question = std::env::var("SAC_QUESTION").unwrap_or_else(|_| {
        "What should a small team consider before adopting a local LLM instead of a cloud API?"
            .to_string()
    });
    let max_subqs: usize = std::env::var("SAC_SUBQUESTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    println!("Question: {question}\n");

    let mut totals = ExecutionStats::default();

    // ── Stage 1: PLAN ──
    let planner = build_agent(
        &cfg,
        "You are a research planner. Given a question, output a numbered list of distinct, \
         non-overlapping sub-questions that together cover it. Output ONLY the list, one \
         sub-question per line, each starting with a number and a period.",
        400,
        vec![],
    )
    .await?;
    let (plan_raw, pstats) = planner
        .execute(
            &format!("Question: {question}\nGive up to {max_subqs} sub-questions."),
            &[],
        )
        .await?;
    totals.accumulate(&pstats);

    let subquestions = parse_list(&plan_raw, max_subqs);
    if subquestions.is_empty() {
        return Err("planner returned no sub-questions".into());
    }
    println!("Plan ({} sub-questions):", subquestions.len());
    for (i, q) in subquestions.iter().enumerate() {
        println!("  {}. {q}", i + 1);
    }
    println!();

    // ── Stage 2: MAP (concurrent) ──
    // One worker agent, shared across tasks; each sub-question answered in parallel.
    let worker = Arc::new(
        build_agent(
            &cfg,
            "You are a research assistant. Answer the single sub-question concisely and \
             concretely in 2-4 sentences. If you are uncertain, say so.",
            400,
            vec![],
        )
        .await?,
    );

    let mut handles = Vec::with_capacity(subquestions.len());
    for (idx, q) in subquestions.iter().cloned().enumerate() {
        let worker = Arc::clone(&worker);
        handles.push(tokio::spawn(async move {
            let out = worker.execute(&q, &[]).await;
            (idx, q, out)
        }));
    }

    let mut findings: Vec<(usize, String, String)> = Vec::with_capacity(handles.len());
    for h in handles {
        let (idx, q, out) = h.await?;
        let (answer, wstats) = out?;
        totals.accumulate(&wstats);
        findings.push((idx, q, answer));
    }
    findings.sort_by_key(|(idx, _, _)| *idx);

    println!("Findings:");
    for (idx, q, a) in &findings {
        println!(
            "  [{}] {q}\n      {}\n",
            idx + 1,
            a.trim().replace('\n', "\n      ")
        );
    }

    // ── Stage 3: REDUCE ──
    let synthesizer = build_agent(
        &cfg,
        "You are a synthesis writer. Combine the provided sub-question findings into a single \
         coherent answer to the ORIGINAL question. Cite findings inline as [1], [2], etc. \
         matching their numbers. Be well-organized and avoid repetition.",
        900,
        vec![],
    )
    .await?;

    let reduce_prompt = build_reduce_prompt(&question, &findings);
    let (report, sstats) = synthesizer.execute(&reduce_prompt, &[]).await?;
    totals.accumulate(&sstats);

    println!("═════════════════════════════════════════════════════");
    println!("Synthesized report:\n\n{}\n", report.trim());
    println!("═════════════════════════════════════════════════════");
    println!(
        "Pipeline: 1 plan + {} concurrent workers + 1 synthesis · {} completions · {} tokens",
        findings.len(),
        totals.message_count,
        totals.total_tokens(),
    );
    Ok(())
}

/// Parse a numbered/bulleted list from model output into clean lines.
fn parse_list(raw: &str, max: usize) -> Vec<String> {
    raw.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(strip_leader)
        .filter(|l| !l.is_empty())
        .take(max)
        .map(|s| s.to_string())
        .collect()
}

/// Strip a leading "1.", "1)", "-", "*", "•" marker from a list item.
fn strip_leader(line: &str) -> &str {
    let t = line.trim_start();
    let t = t
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .trim_start_matches(['.', ')', ':', '-', '*', '•', ' ']);
    t.trim()
}

fn build_reduce_prompt(question: &str, findings: &[(usize, String, String)]) -> String {
    let mut s = format!("Original question: {question}\n\nFindings:\n");
    for (idx, q, a) in findings {
        s.push_str(&format!("[{}] Q: {q}\n    A: {a}\n", idx + 1));
    }
    s.push_str("\nWrite the synthesized answer with inline [n] citations.");
    s
}
