//! Pluggable persistence for the runs runtime.
//!
//! The contract is intentionally coarse: there is exactly one method
//! ([`Store::complete_job_and_propagate`]) that finalises a job and walks its
//! outgoing edges. See the project plan for the full step-by-step semantics.

mod event;
mod memory;

/// Reusable conformance suite for `Store` impls.
///
/// Always public (test-shipping crate) so external `Store` impls in sibling
/// crates (e.g. `sac_runs_surreal`) can import the same invariants. Only
/// referenced by their own `#[cfg(test)]` modules, so it never lands in
/// release binaries that don't pull tests.
pub mod contract_tests;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

pub use contract_tests::store_conformance_suite;
pub use event::{StoreEvent, StoreEventStream};
pub use memory::InMemoryStore;

use super::error::StoreError;
use super::model::{
    DispatchTicket, Job, JobId, LogRange, PipelineId, PipelineSpec, Run, RunId, TerminalOutcome,
};

/// Pluggable persistence + atomic-orchestration backend.
///
/// Implementations MUST honour the per-method contracts below. The
/// reusable conformance suite at [`contract_tests::store_conformance_suite`] (test-only)
/// asserts the invariants that matter (idempotency, mixed fan-in, cancellation
/// fence). Run it against every `Store` impl.
#[async_trait]
pub trait Store: Send + Sync + 'static {
    // -------- pipeline registry --------

    /// Insert a pipeline spec. Idempotent: re-putting an identical spec is a no-op.
    /// Re-putting under the same id with a different spec returns
    /// [`StoreError::PipelineConflict`].
    async fn put_pipeline(&self, spec: PipelineSpec) -> Result<(), StoreError>;

    /// Look up a pipeline spec.
    async fn get_pipeline(&self, id: PipelineId) -> Result<Option<PipelineSpec>, StoreError>;

    // -------- run lifecycle --------

    /// Atomically materialise a Run + every Job/Edge/Bundle instance for the spec.
    /// Submitted `inputs` are deep-merged into each root job's `default_inputs`
    /// (root = job with zero incoming edges). Returns the new run id.
    async fn create_run(
        &self,
        pipeline: PipelineId,
        inputs: serde_json::Value,
    ) -> Result<RunId, StoreError>;

    // -------- dispatcher hot path --------

    /// Pick the oldest `Queued` job whose `kind` is in the supplied set.
    /// Does NOT transition status — the caller follows up with
    /// [`Store::try_dispatch`] which performs the atomic CAS.
    async fn next_dispatchable(&self, kinds: &[String]) -> Result<Option<Job>, StoreError>;

    /// Atomic Queued→Running CAS with per-kind concurrency cap. On success returns
    /// a [`DispatchTicket`] capturing the job/run cancel + completion generations
    /// observed at dispatch time; the dispatcher MUST hand this back to
    /// [`Store::complete_job_and_propagate`]. Returns `None` if the cap is full
    /// or the row's status changed under us.
    async fn try_dispatch(
        &self,
        job_id: JobId,
        max_concurrent_for_kind: usize,
    ) -> Result<Option<DispatchTicket>, StoreError>;

    /// Record a process id on a Running job (used by `SubprocessHandler` so the
    /// cancel sweep can SIGTERM/SIGKILL).
    async fn set_pid(&self, job_id: JobId, pid: u32) -> Result<(), StoreError>;

    /// SINGLE atomic finaliser. See plan §4 for the full step list. Returns
    /// `Ok(true)` if the call mutated state, `Ok(false)` if it was a no-op
    /// (stale generation, already terminal, or a `cancel_run` finalised the row).
    async fn complete_job_and_propagate(
        &self,
        ticket: DispatchTicket,
        terminal: TerminalOutcome,
    ) -> Result<bool, StoreError>;

    /// Cascade cancellation. Atomic: bumps `Run.cancel_generation`, marks every
    /// non-terminal job `Cancelled` (including Running rows — the dispatcher's
    /// cancel sweep takes care of killing in-flight handlers), resolves their
    /// outgoing edges as `Unsatisfied`, and recomputes downstream readiness.
    async fn cancel_run(&self, run_id: RunId) -> Result<(), StoreError>;

    // -------- observability --------

    /// Append a log line. Sequence numbers are assigned by the store.
    async fn append_log(&self, job_id: JobId, line: &str) -> Result<(), StoreError>;

    /// Read log lines in the requested range.
    async fn read_log(&self, job_id: JobId, range: LogRange) -> Result<Vec<String>, StoreError>;

    /// Snapshot a single job.
    async fn get_job(&self, job_id: JobId) -> Result<Option<Job>, StoreError>;

    /// Snapshot a single run.
    async fn get_run(&self, run_id: RunId) -> Result<Option<Run>, StoreError>;

    /// Every job that belongs to `run_id`, in stable id order.
    async fn list_run_jobs(&self, run_id: RunId) -> Result<Vec<Job>, StoreError>;

    /// Subscribe to lifecycle events. Used by `Runtime::wait` to avoid polling.
    async fn subscribe(&self) -> StoreEventStream;

    // -------- sweepers --------

    /// Jobs whose `started_at < deadline`. Returned tickets carry the witnesses
    /// `OrphanWorker` needs to call [`Store::complete_job_and_propagate`] safely.
    async fn find_running_past(
        &self,
        deadline: DateTime<Utc>,
    ) -> Result<Vec<DispatchTicket>, StoreError>;

    /// Jobs flagged by `cancel_run` for the SIGTERM/SIGKILL sweep.
    async fn find_cancelled_with_pid(&self) -> Result<Vec<(JobId, u32)>, StoreError>;

    /// Clear `cancel_kill_pending` and `pid` on a job after the sweep killed it.
    async fn clear_pid_and_kill_flag(&self, job_id: JobId) -> Result<(), StoreError>;
}
