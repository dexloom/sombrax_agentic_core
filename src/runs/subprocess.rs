//! `JobHandler` impl that spawns an external process — covers legacy CLI agents
//! (the `sombrax_rig audit` style) without bespoke executors.
//!
//! Inputs are interpolated into argv and env via [`ArgTpl::FromInput`] and
//! [`EnvTpl::FromInput`]. Stdout + stderr lines are streamed to the per-job log
//! via [`crate::runs::log::LogWriter`]. The `OUTPUT_RESULT:<path>` sentinel
//! (matching `sombrax_worker_core/src/types.rs:77-89`) is parsed from the last
//! matching stdout line and surfaced in the [`JobOutput`].
//!
//! Cancellation: when the cancel sweep calls [`SubprocessHandler::cancel`], the
//! handler sends `SIGTERM` followed by `SIGKILL` to the recorded pid (Unix only).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::runs::handler::{JobContext, JobError, JobHandler, JobOutput};
use crate::runs::model::JobId;

/// Output-parsing strategy for [`SubprocessHandler`].
#[derive(Clone, Debug, Default)]
pub enum OutputParser {
    /// Look for the last `OUTPUT_RESULT:<path>` line. The handler returns
    /// `{"output_file": "<path>"}` as its `JobOutput::value`. Default — matches
    /// the `sombrax_worker` agent contract.
    #[default]
    OutputResultSentinel,
    /// Parse the last non-empty stdout line as JSON; failure → `Failed`.
    LastJsonLine,
    /// No parsing; the handler returns `{}` as output.
    None,
}

/// One template fragment for an argv slot.
#[derive(Clone, Debug)]
pub enum ArgTpl {
    /// Verbatim literal.
    Literal(String),
    /// Dotted path into `JobContext::inputs`. Missing fields render as empty.
    FromInput(String),
}

impl ArgTpl {
    /// Convenience: short form for literal.
    pub fn lit<S: Into<String>>(s: S) -> Self {
        Self::Literal(s.into())
    }
    /// Convenience: short form for input.
    pub fn input<S: Into<String>>(path: S) -> Self {
        Self::FromInput(path.into())
    }
}

/// Env-var template.
#[derive(Clone, Debug)]
pub enum EnvTpl {
    /// Verbatim literal.
    Literal(String),
    /// Dotted path into `JobContext::inputs`.
    FromInput(String),
}

/// In-process handler that spawns a subprocess.
pub struct SubprocessHandler {
    kind: String,
    binary: PathBuf,
    args: Vec<ArgTpl>,
    env: HashMap<String, EnvTpl>,
    timeout: Option<Duration>,
    parser: OutputParser,
    /// Map of in-flight pids per job, populated when `run()` spawns and used by `cancel()`.
    /// Note: this is *also* persisted to the store via `Store::set_pid`, so the
    /// cancel sweep can find rows even across restarts.
    pids: Arc<Mutex<HashMap<JobId, u32>>>,
}

impl SubprocessHandler {
    /// Build a handler for `kind` that runs `binary` with the given argv template.
    pub fn new<S: Into<String>>(kind: S, binary: impl Into<PathBuf>) -> Self {
        Self {
            kind: kind.into(),
            binary: binary.into(),
            args: Vec::new(),
            env: HashMap::new(),
            timeout: None,
            parser: OutputParser::default(),
            pids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Append an argv slot.
    pub fn arg(mut self, tpl: ArgTpl) -> Self {
        self.args.push(tpl);
        self
    }

    /// Convenience: literal arg.
    pub fn arg_lit<S: Into<String>>(self, s: S) -> Self {
        self.arg(ArgTpl::lit(s))
    }

    /// Convenience: input arg.
    pub fn arg_input<S: Into<String>>(self, path: S) -> Self {
        self.arg(ArgTpl::input(path))
    }

    /// Set an env var.
    pub fn env<K: Into<String>>(mut self, key: K, tpl: EnvTpl) -> Self {
        self.env.insert(key.into(), tpl);
        self
    }

    /// Set the per-job timeout (also surfaced via `JobHandler::timeout`).
    pub fn timeout_dur(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Choose how stdout is parsed into `JobOutput::value`.
    pub fn parser(mut self, p: OutputParser) -> Self {
        self.parser = p;
        self
    }
}

const OUTPUT_RESULT_SENTINEL: &str = "OUTPUT_RESULT:";

/// Resolve a dotted path from a JSON value.
fn resolve_path(value: &Value, path: &str) -> String {
    let mut cur = value;
    for seg in path.split('.') {
        cur = cur.get(seg).unwrap_or(&Value::Null);
    }
    match cur {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn render_arg(tpl: &ArgTpl, inputs: &Value) -> String {
    match tpl {
        ArgTpl::Literal(s) => s.clone(),
        ArgTpl::FromInput(p) => resolve_path(inputs, p),
    }
}

fn render_env(tpl: &EnvTpl, inputs: &Value) -> String {
    match tpl {
        EnvTpl::Literal(s) => s.clone(),
        EnvTpl::FromInput(p) => resolve_path(inputs, p),
    }
}

#[async_trait]
impl JobHandler for SubprocessHandler {
    fn kind(&self) -> &str {
        &self.kind
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    async fn run(&self, ctx: JobContext) -> Result<JobOutput, JobError> {
        let argv: Vec<String> = self
            .args
            .iter()
            .map(|t| render_arg(t, &ctx.inputs))
            .collect();
        let env: Vec<(String, String)> = self
            .env
            .iter()
            .map(|(k, t)| (k.clone(), render_env(t, &ctx.inputs)))
            .collect();

        let mut cmd = Command::new(&self.binary);
        cmd.args(&argv);
        for (k, v) in &env {
            cmd.env(k, v);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(JobError::Io)?;
        if let Some(pid) = child.id() {
            self.pids.lock().await.insert(ctx.job_id, pid);
            // Best-effort persistence so the cancel sweep can target the row.
            if let Err(e) = ctx.store.set_pid(ctx.job_id, pid).await {
                ctx.log.warn(format!("failed to persist pid: {e}")).await;
            }
        }

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let log = ctx.log.clone();
        let parser = self.parser.clone();

        // Spawn a small task to stream both streams concurrently and capture
        // stdout for sentinel parsing.
        let stream_handle = tokio::spawn(async move {
            let mut combined_stdout = String::new();
            if let (Some(stdout), Some(stderr)) = (stdout, stderr) {
                let mut so = BufReader::new(stdout).lines();
                let mut se = BufReader::new(stderr).lines();
                loop {
                    tokio::select! {
                        line = so.next_line() => match line {
                            Ok(Some(l)) => {
                                combined_stdout.push_str(&l);
                                combined_stdout.push('\n');
                                log.info(l).await;
                            }
                            Ok(None) => break,
                            Err(e) => {
                                log.warn(format!("stdout read error: {e}")).await;
                                break;
                            }
                        },
                        line = se.next_line() => match line {
                            Ok(Some(l)) => log.warn(format!("[stderr] {l}")).await,
                            Ok(None) => {}
                            Err(e) => log.warn(format!("stderr read error: {e}")).await,
                        }
                    }
                }
                while let Ok(Some(l)) = se.next_line().await {
                    log.warn(format!("[stderr] {l}")).await;
                }
            }
            (combined_stdout, parser)
        });

        let exit_status = match child.wait().await {
            Ok(s) => s,
            Err(e) => {
                // Drop pid mapping before returning.
                self.pids.lock().await.remove(&ctx.job_id);
                return Err(JobError::Io(e));
            }
        };

        let (combined_stdout, parser) = stream_handle.await.unwrap_or_default();

        self.pids.lock().await.remove(&ctx.job_id);

        if !exit_status.success() {
            let code = exit_status.code();
            return Err(JobError::Failed(format!(
                "subprocess exited with status {:?}",
                code
            )));
        }

        let value: Value = match parser {
            OutputParser::OutputResultSentinel => {
                if let Some(path) = combined_stdout
                    .lines()
                    .rev()
                    .find_map(|l| l.strip_prefix(OUTPUT_RESULT_SENTINEL))
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                {
                    serde_json::json!({ "output_file": path })
                } else {
                    serde_json::json!({})
                }
            }
            OutputParser::LastJsonLine => {
                let line = combined_stdout
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .trim();
                if line.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str(line)
                        .map_err(|e| JobError::Failed(format!("LastJsonLine parse failed: {e}")))?
                }
            }
            OutputParser::None => serde_json::json!({}),
        };

        Ok(JobOutput::new(value))
    }

    async fn cancel(&self, job_id: JobId) -> Result<(), JobError> {
        let pid = self.pids.lock().await.get(&job_id).copied();
        if let Some(pid) = pid {
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;
                let target = Pid::from_raw(pid as i32);
                if let Err(e) = kill(target, Signal::SIGTERM) {
                    tracing::warn!(?e, %job_id, %pid, "SIGTERM failed; trying SIGKILL");
                    let _ = kill(target, Signal::SIGKILL);
                }
            }
            #[cfg(not(unix))]
            {
                tracing::warn!(%job_id, %pid, "SubprocessHandler::cancel is no-op on non-Unix");
            }
            self.pids.lock().await.remove(&job_id);
        }
        Ok(())
    }
}
