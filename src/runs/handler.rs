//! [`JobHandler`] trait + [`JobContext`] + [`JobOutput`] + [`JobError`].
//!
//! Handlers are the in-process execution path. A subprocess fallback is provided
//! by [`super::subprocess::SubprocessHandler`] (Phase 3).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

use crate::context::CancelSignal;

use super::log::LogWriter;
use super::model::{BundleId, JobId, RunId};
use super::store::Store;

/// What a handler does to fulfil one job.
#[async_trait]
pub trait JobHandler: Send + Sync + 'static {
    /// Unique kind string. Must match `JobTemplate::kind`.
    fn kind(&self) -> &str;

    /// Optional informational JSON schema for the inputs the handler accepts.
    fn input_schema(&self) -> Option<Value> {
        None
    }

    /// Per-kind concurrency cap. `None` means no cap.
    fn max_concurrent(&self) -> Option<usize> {
        None
    }

    /// Per-job timeout. `None` means no timeout (caller must rely on cancel).
    fn timeout(&self) -> Option<Duration> {
        None
    }

    /// Run the handler. The runtime invokes this exactly once per dispatch.
    async fn run(&self, ctx: JobContext) -> Result<JobOutput, JobError>;

    /// Best-effort cancellation hook. Default no-op; `SubprocessHandler` overrides
    /// this to send SIGTERM/SIGKILL.
    async fn cancel(&self, _job_id: JobId) -> Result<(), JobError> {
        Ok(())
    }
}

/// Per-invocation context the runtime hands to a handler.
pub struct JobContext {
    /// The job being executed.
    pub job_id: JobId,
    /// The run this job belongs to.
    pub run_id: RunId,
    /// Bundle id, when this job belongs to a bundle.
    pub bundle_id: Option<BundleId>,
    /// Effective inputs after edges resolved.
    pub inputs: Value,
    /// Cooperative cancellation. The runtime fires this when `cancel_run`
    /// targets the run.
    pub cancel: CancelSignal,
    /// Per-job log sink.
    pub log: LogWriter,
    /// Direct store access (advanced handlers can read sibling state).
    pub store: Arc<dyn Store>,
}

/// Successful handler output.
#[derive(Clone, Debug)]
pub struct JobOutput {
    /// The output value persisted into `Job::output`. Edges read from this.
    pub value: Value,
    /// Optional findings count for telemetry.
    pub findings_count: Option<usize>,
}

impl JobOutput {
    /// Helper: output with no findings count.
    pub fn new(value: Value) -> Self {
        Self {
            value,
            findings_count: None,
        }
    }
}

/// Failure mode of a [`JobHandler::run`] call.
#[derive(Debug, Error)]
pub enum JobError {
    /// Handler-specific error.
    #[error("{0}")]
    Failed(String),
    /// Handler observed cancel and decided to bail out cleanly.
    #[error("cancelled")]
    Cancelled,
    /// Handler hit its own timeout (the runtime also enforces a hard timeout
    /// outside the handler).
    #[error("timeout")]
    Timeout,
    /// Underlying I/O / RPC error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl JobError {
    /// Convenience constructor.
    pub fn failed(msg: impl Into<String>) -> Self {
        Self::Failed(msg.into())
    }
}
