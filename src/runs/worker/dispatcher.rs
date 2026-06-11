//! Dispatch worker.
//!
//! Loop:
//! 1. `next_dispatchable(kinds)` → oldest Queued job whose kind is registered.
//! 2. `try_dispatch(job_id, max_concurrent_for_kind)` → atomic Queued→Running CAS.
//!    Returns a `DispatchTicket` capturing `(completion_gen, cancel_gen)` at dispatch time.
//! 3. Spawn a `tokio::task` that runs `handler.run(ctx)` with the handler's optional
//!    timeout, then calls `complete_job_and_propagate(ticket, terminal)` exactly once.
//!
//! Cancellation is handled by `super::cancel_sweep::CancelSweep` — that's a separate
//! task so the hot dispatch path doesn't pay the scan cost.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::time;
use tracing::{debug, error, instrument, warn};

use crate::context::CancelSignal;
use crate::runs::handler::{JobContext, JobError};
use crate::runs::log::LogWriter;
use crate::runs::model::{DispatchTicket, JobStatus, TerminalOutcome};
use crate::runs::registry::HandlerRegistry;
use crate::runs::store::Store;

/// Run the dispatcher loop until `cancel` fires.
///
/// `tick_interval` is the back-off between empty polls. When a dispatchable job
/// is found, the loop continues immediately so a burst drains without sleeping.
#[instrument(skip_all, name = "runs.dispatcher")]
pub async fn run_dispatcher(
    store: Arc<dyn Store>,
    registry: HandlerRegistry,
    tick_interval: Duration,
    cancel: CancelSignal,
) {
    let kinds = registry.kinds();
    if kinds.is_empty() {
        warn!("dispatcher started with empty registry; exiting");
        return;
    }

    loop {
        if cancel.is_cancelled() {
            debug!("dispatcher cancel signal observed; exiting");
            return;
        }

        let work = match store.next_dispatchable(&kinds).await {
            Ok(Some(job)) => Some(job),
            Ok(None) => None,
            Err(e) => {
                error!(error = %e, "next_dispatchable failed");
                None
            }
        };

        match work {
            Some(job) => {
                let max_conc = registry.max_concurrent(&job.kind);
                let handler = match registry.get(&job.kind) {
                    Some(h) => h,
                    None => {
                        // Shouldn't happen — pipeline validation rejects unknown kinds —
                        // but guard regardless.
                        error!(kind = %job.kind, "handler missing from registry; will not dispatch");
                        time::sleep(tick_interval).await;
                        continue;
                    }
                };

                let ticket = match store.try_dispatch(job.id, max_conc).await {
                    Ok(Some(t)) => t,
                    Ok(None) => {
                        // Cap saturated or status changed under us; back off briefly.
                        time::sleep(Duration::from_millis(5)).await;
                        continue;
                    }
                    Err(e) => {
                        error!(error = %e, job_id = %job.id, "try_dispatch failed");
                        time::sleep(tick_interval).await;
                        continue;
                    }
                };

                let store_for_task = store.clone();
                let handler_for_task = handler.clone();
                let kind: Arc<str> = Arc::from(job.kind.as_str());
                let inputs = job.inputs.clone();
                let bundle_id = job.bundle_id;
                let timeout = handler.timeout();
                tokio::spawn(async move {
                    execute_one(
                        store_for_task,
                        handler_for_task,
                        ticket,
                        kind,
                        inputs,
                        bundle_id,
                        timeout,
                    )
                    .await;
                });
            }
            None => {
                // Idle — poll again after the back-off. Use `select!` so cancel
                // wakes us promptly.
                let cancel_clone = cancel.clone();
                tokio::select! {
                    _ = time::sleep(tick_interval) => {}
                    _ = wait_until_cancelled(cancel_clone) => return,
                }
            }
        }
    }
}

async fn wait_until_cancelled(cancel: CancelSignal) {
    let mut interval = time::interval(Duration::from_millis(50));
    loop {
        interval.tick().await;
        if cancel.is_cancelled() {
            return;
        }
    }
}

#[instrument(skip_all, fields(kind = %kind, job_id = %ticket.job_id, run_id = %ticket.run_id))]
async fn execute_one(
    store: Arc<dyn Store>,
    handler: Arc<dyn crate::runs::handler::JobHandler>,
    ticket: DispatchTicket,
    kind: Arc<str>,
    inputs: Value,
    bundle_id: Option<crate::runs::model::BundleId>,
    timeout: Option<Duration>,
) {
    let cancel = CancelSignal::new();
    let log = LogWriter::new(ticket.job_id, ticket.run_id, kind.clone(), store.clone());
    let ctx = JobContext {
        job_id: ticket.job_id,
        run_id: ticket.run_id,
        bundle_id,
        inputs,
        cancel: cancel.clone(),
        log,
        store: store.clone(),
    };

    let run_fut = handler.run(ctx);
    let outcome = match timeout {
        Some(d) => match time::timeout(d, run_fut).await {
            Ok(r) => to_terminal(r),
            Err(_elapsed) => {
                // Signal cooperative cancel just in case the handler is still
                // holding state somewhere; the outcome is reported as Timeout.
                cancel.cancel();
                TerminalOutcome::Timeout
            }
        },
        None => to_terminal(run_fut.await),
    };

    let mutated = match store.complete_job_and_propagate(ticket, outcome).await {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "complete_job_and_propagate failed");
            return;
        }
    };
    if !mutated {
        // CAS missed — either cancel_run finalised the row, or this is a stale
        // dispatch (should be impossible given the dispatcher's invariants).
        debug!("complete_job_and_propagate was a no-op (cancel raced or stale generation)");
    }
}

fn to_terminal(res: Result<crate::runs::handler::JobOutput, JobError>) -> TerminalOutcome {
    match res {
        Ok(out) => TerminalOutcome::Success {
            output: out.value,
            findings_count: out.findings_count,
        },
        Err(JobError::Cancelled) => TerminalOutcome::Cancelled,
        Err(JobError::Timeout) => TerminalOutcome::Timeout,
        Err(JobError::Failed(msg)) => TerminalOutcome::Failure {
            error: msg,
            exit_code: None,
        },
        Err(JobError::Io(e)) => TerminalOutcome::Failure {
            error: format!("io: {e}"),
            exit_code: None,
        },
    }
}

// JobStatus is referenced indirectly via TerminalOutcome::status() in tests;
// keep the import alive for future tracing fields.
const _: fn() = || {
    let _ = JobStatus::Completed;
};
