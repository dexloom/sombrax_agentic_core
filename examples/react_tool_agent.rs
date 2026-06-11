//! Example 4 — ReAct agent with a custom `Tool`.
//!
//! Defines a `Calculator` tool by implementing SAC's `Tool` trait (typed args, typed
//! output, async `call`), converts it to a dynamic tool with `into_arc_dyn`, and hands
//! it to an agent. The agent then runs the tool-execution loop: it reasons, calls the
//! tool (possibly several times), feeds results back to itself, and answers — all inside
//! a single `execute`. `ExecutionStats::tool_calls` shows how many times the tool fired.
//!
//! This showcases:
//!   * the `Tool` trait with `#[derive(Deserialize)]`/`Serialize` arg & output types,
//!   * `into_arc_dyn` to register a tool with an agent,
//!   * the built-in agentic loop driving multi-step tool use.
//!
//! NOTE: this needs a model that supports function/tool calling. Pick one accordingly
//! (e.g. an Ollama model with tool support, or a hosted provider).
//!
//! Run:
//!   cargo run --example react_tool_agent

mod common;

use serde::{Deserialize, Serialize};
use sombrax_agentic_core::providers::build_agent;
use sombrax_agentic_core::tool::{into_arc_dyn, Tool, ToolDefinition};

/// Typed arguments the model must produce to call the calculator.
#[derive(Debug, Deserialize)]
struct CalcArgs {
    /// One of: add, sub, mul, div.
    op: String,
    a: f64,
    b: f64,
}

/// Typed result the tool returns (serialized to JSON for the model).
#[derive(Debug, Serialize)]
struct CalcResult {
    result: f64,
}

/// A tool's error type only has to be a real `std::error::Error`.
#[derive(Debug)]
struct CalcError(String);

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CalcError {}

#[derive(Clone)]
struct Calculator;

impl Tool for Calculator {
    const NAME: &'static str = "calculator";
    type Args = CalcArgs;
    type Output = CalcResult;
    type Error = CalcError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition::new(
            Self::NAME,
            "Perform a single arithmetic operation on two numbers. \
             Call this once per operation; chain calls for multi-step math.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "sub", "mul", "div"],
                        "description": "The operation to perform"
                    },
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                },
                "required": ["op", "a", "b"]
            }),
        )
    }

    async fn call(&self, args: CalcArgs) -> Result<CalcResult, CalcError> {
        let result = match args.op.as_str() {
            "add" => args.a + args.b,
            "sub" => args.a - args.b,
            "mul" => args.a * args.b,
            "div" => {
                if args.b == 0.0 {
                    return Err(CalcError("division by zero".to_string()));
                }
                args.a / args.b
            }
            other => return Err(CalcError(format!("unknown op: {other}"))),
        };
        // A real tool might hit a DB or an API here; this one just computes.
        tracing::info!("calculator: {} {} {} = {result}", args.a, args.op, args.b);
        Ok(CalcResult { result })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let cfg = common::config_from_env();
    common::banner("ReAct agent with a custom Calculator tool", &cfg);

    let system = "You are a precise math assistant. You CANNOT do arithmetic yourself — \
                  you must use the `calculator` tool for every individual operation, one \
                  operation per call. After computing, state the final numeric answer clearly.";

    // Register the typed tool as a dynamic tool the agent can call.
    let tools = vec![into_arc_dyn(Calculator)];
    let agent = build_agent(&cfg, system, 600, tools).await?;

    let question = std::env::var("SAC_QUESTION").unwrap_or_else(|_| {
        "Compute (12.5 * 8) + 100, then divide that total by 3. Show the final number.".to_string()
    });
    println!("Question: {question}\n");

    let (answer, stats) = agent.execute(&question, &[]).await?;
    println!("Answer:\n  {}\n", answer.trim().replace('\n', "\n  "));
    println!(
        "Tool calls: {} · completions: {} · tokens: {} · tool errors: {}",
        stats.tool_calls,
        stats.message_count,
        stats.total_tokens(),
        stats.tool_error_count,
    );
    Ok(())
}
