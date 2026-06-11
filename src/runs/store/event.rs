//! Lifecycle events emitted by [`super::Store::subscribe`].

use std::pin::Pin;

use futures_util::Stream;

use crate::runs::model::{BundleId, BundleStatus, JobId, JobStatus, RunId, RunStatus};

/// One lifecycle event. Implementations should emit at least the four variants
/// below; new variants may be added in the future, so consumers must use
/// `#[non_exhaustive]` matching.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum StoreEvent {
    /// A job reached a terminal state.
    JobTerminal {
        /// The terminal job.
        job_id: JobId,
        /// The run.
        run_id: RunId,
        /// Final status (Completed / Failed / Timeout / Cancelled / Skipped).
        status: JobStatus,
    },
    /// A bundle changed status.
    BundleStatus {
        /// The bundle.
        bundle_id: BundleId,
        /// The run.
        run_id: RunId,
        /// New bundle status.
        status: BundleStatus,
    },
    /// The whole run reached a terminal state.
    RunTerminal {
        /// The run.
        run_id: RunId,
        /// Final run status.
        status: RunStatus,
    },
    /// `cancel_run` fired (emitted exactly once per cancellation).
    RunCancelled {
        /// The run.
        run_id: RunId,
    },
}

/// Boxed `Stream` of [`StoreEvent`]s. Rebroadcast: each subscriber sees events
/// emitted after `subscribe()` returned (no replay).
pub type StoreEventStream = Pin<Box<dyn Stream<Item = StoreEvent> + Send>>;
