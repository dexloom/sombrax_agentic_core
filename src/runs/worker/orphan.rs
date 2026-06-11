//! Orphan worker — re-finalises Running jobs that exceeded a global deadline.
//!
//! Most timeouts are handled by [`super::dispatcher`] via `tokio::time::timeout`
//! around the handler. This worker is a backstop: if the dispatcher task is lost
//! (panic, runtime drop, machine reboot), Running rows would otherwise sit
//! forever. The orphan worker scans `find_running_past(now - max_runtime)` and
//! calls `complete_job_and_propagate(ticket, Timeout)` with the witness ticket
//! the store returned, so it never races a healthy dispatch.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::time;
use tracing::{debug, error, instrument};

use crate::context::CancelSignal;
use crate::runs::model::TerminalOutcome;
use crate::runs::store::Store;

/// Run the orphan loop until `cancel` fires.
#[instrument(skip_all, name = "runs.orphan")]
pub async fn run_orphan(
    store: Arc<dyn Store>,
    max_runtime: Duration,
    poll_interval: Duration,
    cancel: CancelSignal,
) {
    loop {
        if cancel.is_cancelled() {
            debug!("orphan worker cancel signal observed; exiting");
            return;
        }
        let deadline = Utc::now() - chrono::Duration::from_std(max_runtime).unwrap_or_default();
        match store.find_running_past(deadline).await {
            Ok(tickets) => {
                for ticket in tickets {
                    match store
                        .complete_job_and_propagate(ticket, TerminalOutcome::Timeout)
                        .await
                    {
                        Ok(true) => {
                            tracing::warn!(
                                job_id = %ticket.job_id,
                                run_id = %ticket.run_id,
                                "orphan timeout finalised stale Running job"
                            );
                        }
                        Ok(false) => {
                            // Witness was stale (real dispatcher beat us, or cancel landed) — fine.
                        }
                        Err(e) => error!(error = %e, "orphan timeout propagation failed"),
                    }
                }
            }
            Err(e) => error!(error = %e, "find_running_past failed"),
        }

        let cancel_clone = cancel.clone();
        tokio::select! {
            _ = time::sleep(poll_interval) => {}
            _ = wait_until_cancelled(cancel_clone) => return,
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
