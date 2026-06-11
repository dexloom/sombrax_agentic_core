//! Background workers that drive the runtime.
//!
//! - [`dispatcher::DispatchWorker`]: pulls Queued jobs and runs handlers.
//! - [`orphan::OrphanWorker`]: re-finalises Running jobs that exceeded their timeout.
//! - [`cancel_sweep::CancelSweep`]: invokes `handler.cancel` for `cancel_run`-marked rows.

pub mod cancel_sweep;
pub mod dispatcher;
pub mod orphan;

use std::time::Duration;

/// Tick interval / one-shot policy for a [`BackgroundWorker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerSchedule {
    /// Sleep `interval` between ticks.
    Periodic(Duration),
    /// Run once and exit.
    OneShot,
}

/// Common shape for the dispatcher / orphan / cancel-sweep tasks. They are not
/// strictly required to share a trait — the runtime spawns each one explicitly —
/// but giving them a uniform `tick` makes testing easier.
#[allow(dead_code)] // currently spawned ad-hoc; trait is reserved for testability.
#[async_trait::async_trait]
pub trait BackgroundWorker: Send + Sync + 'static {
    /// Human-readable name for tracing.
    fn name(&self) -> &'static str;
    /// Schedule.
    fn schedule(&self) -> WorkerSchedule;
    /// One iteration of work.
    async fn tick(&self);
}
