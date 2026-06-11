//! Per-job log capture, surfacing to both the [`super::store::Store`] and
//! the `tracing` subscriber.

use std::sync::Arc;

use tracing::Level;

use super::model::{JobId, RunId};
use super::store::Store;

/// Cheap, cloneable log handle for a single job. Writes go to the store via
/// `append_log` (so `runtime.read_log()` can replay them) and are also re-emitted
/// to `tracing` with structured fields.
#[derive(Clone)]
pub struct LogWriter {
    job_id: JobId,
    run_id: RunId,
    kind: Arc<str>,
    store: Arc<dyn Store>,
}

impl LogWriter {
    /// Build a writer scoped to one job.
    pub fn new(job_id: JobId, run_id: RunId, kind: Arc<str>, store: Arc<dyn Store>) -> Self {
        Self {
            job_id,
            run_id,
            kind,
            store,
        }
    }

    /// The job this writer is scoped to.
    pub fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Append one log line. Writes to the store and emits a tracing event at INFO.
    pub async fn info(&self, line: impl Into<String>) {
        self.write(Level::INFO, line.into()).await;
    }

    /// Append at WARN.
    pub async fn warn(&self, line: impl Into<String>) {
        self.write(Level::WARN, line.into()).await;
    }

    /// Append at ERROR.
    pub async fn error(&self, line: impl Into<String>) {
        self.write(Level::ERROR, line.into()).await;
    }

    async fn write(&self, level: Level, line: String) {
        match level {
            Level::ERROR => tracing::error!(
                job_id = %self.job_id,
                run_id = %self.run_id,
                kind = %self.kind,
                "{}",
                line
            ),
            Level::WARN => tracing::warn!(
                job_id = %self.job_id,
                run_id = %self.run_id,
                kind = %self.kind,
                "{}",
                line
            ),
            _ => tracing::info!(
                job_id = %self.job_id,
                run_id = %self.run_id,
                kind = %self.kind,
                "{}",
                line
            ),
        }
        // Best-effort persistence — log failures must not propagate up to the
        // handler's run loop.
        let _ = self.store.append_log(self.job_id, &line).await;
    }
}
