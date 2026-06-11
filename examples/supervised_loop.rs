//! Example 2 — Assisted agentic loop: a worker does the job, a supervisor reviews
//! each step using only a *minimal, summarized* slice of prior context.
//!
//! The worker advances a task one step at a time. After each step a second model —
//! the supervisor — judges that step. Crucially, the supervisor never sees the full
//! history: it sees the goal, a short running summary of approved progress, and the
//! single step under review. That keeps the supervisor's context small and cheap even
//! as the task grows, and it's the supervisor itself that produces each summary line.
//!
//! Flow per step:
//!   worker (goal + summary + last feedback) → proposes the next step
//!   supervisor (goal + summary + this step only) → APPROVE + 1-line summary, or REVISE + feedback
//!   on REVISE the worker retries the same step with the feedback; on APPROVE we move on.
//!
//! The two roles can run on different models/providers: the worker uses `SAC_*`, the
//! supervisor uses `SAC_JUDGE_*` (falling back to `SAC_*`). A common pattern is a strong
//! worker and a cheap, fast supervisor.
//!
//! Run:
//!   cargo run --example supervised_loop
//!   SAC_GOAL="Design a CLI to rename files by EXIF date" cargo run --example supervised_loop

mod common;

use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::ExecutionStats;

const WORKER_SYSTEM: &str = "You are a focused worker executing a task step by step. \
    Produce ONLY the next single step of work — concrete and self-contained — not the whole \
    solution at once. Keep each step short (a few sentences or a small code/config block). \
    When the task is fully complete, reply with exactly the word DONE and nothing else. \
    If the reviewer gave feedback, address it in your next attempt.";

const SUPERVISOR_SYSTEM: &str = "You are a strict but constructive reviewer. You see the goal, \
    a short summary of approved progress, and ONE proposed step. Judge only that step. \
    Reply in EXACTLY this format, no extra prose:\n\
    VERDICT: APPROVE or REVISE\n\
    SUMMARY: <one short line capturing what this step accomplished, for the running log>\n\
    FEEDBACK: <if REVISE, what to fix; if APPROVE, write n/a>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let worker_cfg = common::config_from_env();
    let supervisor_cfg = common::judge_config_from_env();
    common::banner("Supervised agentic loop (worker + reviewer)", &worker_cfg);
    println!(
        "  supervisor model: provider={} model={}\n",
        supervisor_cfg.provider, supervisor_cfg.model
    );

    let goal = std::env::var("SAC_GOAL").unwrap_or_else(|_| {
        "Write a short Rust function `fn slugify(s: &str) -> String` that lowercases, \
         trims, and replaces runs of non-alphanumeric characters with single hyphens, \
         then explain and add two unit tests."
            .to_string()
    });
    let max_steps: usize = env_usize("SAC_MAX_STEPS", 5);
    let max_attempts: usize = env_usize("SAC_MAX_ATTEMPTS", 2);

    println!("Goal: {goal}\n");

    let worker = build_agent(&worker_cfg, WORKER_SYSTEM, 800, vec![]).await?;
    let supervisor = build_agent(&supervisor_cfg, SUPERVISOR_SYSTEM, 300, vec![]).await?;

    // The "minimal summarized previous context": one short line per approved step.
    let mut summary: Vec<String> = Vec::new();
    let mut totals = ExecutionStats::default();

    'steps: for step in 1..=max_steps {
        println!("───────────────── Step {step} ─────────────────");
        let mut feedback: Option<String> = None;

        for attempt in 1..=max_attempts {
            // --- Worker turn: full goal + compact summary + last feedback ---
            let worker_prompt = build_worker_prompt(&goal, &summary, feedback.as_deref());
            let (proposal, wstats) = worker.execute(&worker_prompt, &[]).await?;
            totals.accumulate(&wstats);

            if proposal.trim().eq_ignore_ascii_case("DONE") {
                println!("  Worker signalled DONE — task complete.\n");
                break 'steps;
            }

            println!(
                "  Worker (attempt {attempt}/{max_attempts}):\n    {}\n",
                indent(&proposal)
            );

            // --- Supervisor turn: minimal context, just this step ---
            let review_prompt = build_review_prompt(&goal, &summary, &proposal);
            let (review, sstats) = supervisor.execute(&review_prompt, &[]).await?;
            totals.accumulate(&sstats);

            let verdict = Review::parse(&review);
            println!(
                "  Supervisor: {} — {}",
                if verdict.approved {
                    "APPROVE"
                } else {
                    "REVISE"
                },
                verdict.feedback.as_deref().unwrap_or("n/a")
            );

            if verdict.approved {
                let line = verdict
                    .summary
                    .unwrap_or_else(|| first_line(&proposal).to_string());
                println!("  ↳ logged: {line}\n");
                summary.push(format!("Step {step}: {line}"));
                continue 'steps;
            }

            // Not approved: carry feedback into the next attempt of THIS step.
            feedback = verdict.feedback.or(Some(review));
            if attempt == max_attempts {
                println!("  ⚠ step not approved after {max_attempts} attempts; moving on.\n");
            }
        }
    }

    println!("═════════════════════════════════════════════════════");
    println!("Approved progress log ({} steps):", summary.len());
    for line in &summary {
        println!("  • {line}");
    }
    println!(
        "\n{} completions · {} tokens ({} in / {} out)",
        totals.message_count,
        totals.total_tokens(),
        totals.input_tokens,
        totals.output_tokens,
    );
    Ok(())
}

/// Parsed supervisor verdict.
struct Review {
    approved: bool,
    summary: Option<String>,
    feedback: Option<String>,
}

impl Review {
    /// Tolerant line-based parse of the supervisor's structured reply.
    fn parse(raw: &str) -> Review {
        let mut approved = false;
        let mut summary = None;
        let mut feedback = None;
        for line in raw.lines() {
            let line = line.trim();
            if let Some(rest) = strip_ci(line, "VERDICT:") {
                approved = rest.trim().to_ascii_uppercase().starts_with("APPROVE");
            } else if let Some(rest) = strip_ci(line, "SUMMARY:") {
                let v = rest.trim();
                if !v.is_empty() {
                    summary = Some(v.to_string());
                }
            } else if let Some(rest) = strip_ci(line, "FEEDBACK:") {
                let v = rest.trim();
                if !v.is_empty() && !v.eq_ignore_ascii_case("n/a") {
                    feedback = Some(v.to_string());
                }
            }
        }
        Review {
            approved,
            summary,
            feedback,
        }
    }
}

fn build_worker_prompt(goal: &str, summary: &[String], feedback: Option<&str>) -> String {
    let mut s = format!("Goal:\n{goal}\n\n");
    if summary.is_empty() {
        s.push_str("No steps approved yet. Produce the FIRST step.\n");
    } else {
        s.push_str("Approved progress so far:\n");
        for line in summary {
            s.push_str(&format!("- {line}\n"));
        }
        s.push_str("\nProduce the NEXT step (or DONE if the goal is fully met).\n");
    }
    if let Some(fb) = feedback {
        s.push_str(&format!(
            "\nThe reviewer rejected your previous attempt: {fb}\nFix it.\n"
        ));
    }
    s
}

/// Note how little the supervisor receives: goal + summary + the one step.
fn build_review_prompt(goal: &str, summary: &[String], proposal: &str) -> String {
    let progress = if summary.is_empty() {
        "(nothing approved yet)".to_string()
    } else {
        summary
            .iter()
            .map(|l| format!("- {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Goal:\n{goal}\n\nApproved progress summary:\n{progress}\n\nProposed next step to review:\n{proposal}"
    )
}

fn strip_ci<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if line.len() >= prefix.len() && line[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&line[prefix.len()..])
    } else {
        None
    }
}

fn first_line(s: &str) -> &str {
    s.trim().lines().next().unwrap_or("").trim()
}

fn indent(text: &str) -> String {
    text.trim().replace('\n', "\n    ")
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
