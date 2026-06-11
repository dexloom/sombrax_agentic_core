//! Example 3 — Guardrail hook: content-modifying interception.
//!
//! Hooks are SAC's signature feature: they sit in the agent lifecycle and can *rewrite*
//! content, not just observe it. This example implements one `GuardrailHook` that:
//!
//! * `pre_completion` — redacts secrets (API-key-looking tokens, emails) out of the
//!   user message *before* it ever reaches the model;
//! * `pre_tool_call` — blocks dangerous tool invocations (e.g. `rm -rf`) by returning
//!   `ToolCallDecision::Block`, so the tool never runs;
//! * `on_assistant_message` — observes each assistant turn for an audit trail.
//!
//! It's wired into a real agent via `build_agent_with_options`, so the redaction is
//! demonstrated end-to-end (the model literally cannot see the secret). The tool-blocking
//! decision is also exercised directly against the hook so you can see both a blocked and
//! an allowed call deterministically, without depending on the model choosing to call a tool.
//!
//! Run:
//!   cargo run --example guardrail_hook

mod common;

use sombrax_agentic_core::context::HookContext;
use sombrax_agentic_core::error::HookResult;
use sombrax_agentic_core::hook::{Hook, ToolCallDecision};
use sombrax_agentic_core::providers::{build_agent_with_options, AgentBuildOptions};
use sombrax_agentic_core::Message;

use regex::Regex;

/// A hook that redacts secrets from outgoing prompts and blocks dangerous tool calls.
#[derive(Clone)]
struct GuardrailHook {
    api_key_re: Regex,
    email_re: Regex,
    // Substrings that, if present in a tool call's arguments, get the call blocked.
    danger_substrings: Vec<&'static str>,
}

impl GuardrailHook {
    fn new() -> Self {
        Self {
            // e.g. "sk-XXXX...", "ghp_XXXX", generic long key-like tokens
            api_key_re: Regex::new(r"(?i)\b(?:sk|ghp|xoxb|api)[-_][A-Za-z0-9]{8,}\b").unwrap(),
            email_re: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(),
            danger_substrings: vec!["rm -rf", "mkfs", "dd if=", ":(){", "> /dev/sda"],
        }
    }

    /// Apply redaction to a piece of text, returning (redacted, count).
    fn redact(&self, text: &str) -> (String, usize) {
        let mut count = 0;
        let step1 = self.api_key_re.replace_all(text, |_: &regex::Captures| {
            count += 1;
            "[REDACTED_SECRET]".to_string()
        });
        let step2 = self.email_re.replace_all(&step1, |_: &regex::Captures| {
            count += 1;
            "[REDACTED_EMAIL]".to_string()
        });
        (step2.into_owned(), count)
    }
}

impl Hook for GuardrailHook {
    /// Rewrite the user's message before it reaches the LLM.
    async fn pre_completion(
        &self,
        message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> HookResult<Message> {
        let original = message.text();
        let (clean, n) = self.redact(&original);
        if n == 0 {
            return Ok(message);
        }
        tracing::warn!("guardrail: redacted {n} secret(s) from outgoing prompt");
        // Preserve the role/id by rebuilding a user message with scrubbed text.
        Ok(Message::user(clean))
    }

    /// Decide whether a tool call may proceed.
    async fn pre_tool_call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        _ctx: &mut HookContext,
    ) -> HookResult<ToolCallDecision> {
        let blob = args.to_string();
        if let Some(bad) = self
            .danger_substrings
            .iter()
            .find(|needle| blob.contains(**needle))
        {
            tracing::warn!("guardrail: blocking '{tool_name}' (matched {bad:?})");
            return Ok(ToolCallDecision::Block(format!(
                "Blocked by guardrail: argument contains forbidden pattern {bad:?}."
            )));
        }
        Ok(ToolCallDecision::Proceed(args))
    }

    /// Observe assistant turns (audit trail).
    async fn on_assistant_message(
        &self,
        message: &Message,
        _ctx: &mut HookContext,
    ) -> HookResult<()> {
        tracing::info!("guardrail: assistant turn ({} chars)", message.text().len());
        Ok(())
    }

    fn name(&self) -> &str {
        "guardrail"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner("Guardrail hook — redaction + tool blocking", &cfg);

    let hook = GuardrailHook::new();

    // ── Part A: tool-call gating, demonstrated directly against the hook ──
    // Deterministic: we don't rely on the model deciding to call a tool.
    println!("Part A — pre_tool_call decisions:");
    let mut ctx = HookContext::new_with_uuid();
    let safe = serde_json::json!({ "command": "ls -la /tmp" });
    let danger = serde_json::json!({ "command": "rm -rf / --no-preserve-root" });
    for (label, args) in [("safe", safe), ("dangerous", danger)] {
        let decision = hook.pre_tool_call("shell", args, &mut ctx).await?;
        match decision {
            ToolCallDecision::Proceed(_) => println!("  {label:>9}: PROCEED"),
            ToolCallDecision::Block(reason) => println!("  {label:>9}: BLOCKED — {reason}"),
        }
    }

    // ── Part B: prompt redaction, demonstrated end-to-end through an agent ──
    println!("\nPart B — pre_completion redaction (end-to-end):");
    let options = AgentBuildOptions {
        hook: Some(hook.clone()),
        ..Default::default()
    };
    let agent = build_agent_with_options(
        &cfg,
        "You are a careful assistant. If asked to repeat the user's message, repeat it verbatim.",
        300,
        vec![],
        options,
    )
    .await?;

    let leaky = "Here is my key sk-ABCD1234EFGH5678 and email alice@example.com — \
                 please repeat my message back to me exactly.";
    println!("  user (raw)     : {leaky}");
    let (redacted_preview, n) = hook.redact(leaky);
    println!("  hook will send : {redacted_preview}  ({n} redactions)");

    match agent.execute(leaky, &[]).await {
        Ok((reply, stats)) => {
            println!("  model reply    : {}", reply.trim());
            println!(
                "  (the model never saw the real secret — {} tokens)",
                stats.total_tokens()
            );
        }
        Err(e) => {
            // Network/model errors shouldn't mask the point of the example.
            println!("  model call failed ({e}).");
            println!("  Part A above already proves the guardrail logic offline.");
        }
    }

    Ok(())
}
