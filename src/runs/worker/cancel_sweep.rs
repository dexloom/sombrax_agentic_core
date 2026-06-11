//! Cancellation sweep — invokes `handler.cancel(job_id)` on rows that
//! `cancel_run` flagged with `cancel_kill_pending`. After the handler is asked
//! to stop, we clear the flag + pid via `Store::clear_pid_and_kill_flag`.

use std::sync::Arc;
use std::time::Duration;

use tokio::time;
use tracing::{debug, error, instrument, warn};

use crate::context::CancelSignal;
use crate::runs::registry::HandlerRegistry;
use crate::runs::store::Store;

/// Run the cancel-sweep loop until `cancel` fires.
#[instrument(skip_all, name = "runs.cancel_sweep")]
pub async fn run_cancel_sweep(
    store: Arc<dyn Store>,
    registry: HandlerRegistry,
    poll_interval: Duration,
    cancel: CancelSignal,
) {
    loop {
        if cancel.is_cancelled() {
            debug!("cancel_sweep observed shutdown; exiting");
            return;
        }
        match store.find_cancelled_with_pid().await {
            Ok(rows) => {
                for (job_id, _pid) in rows {
                    // Look up the job kind so we know which handler to ask.
                    let kind = match store.get_job(job_id).await {
                        Ok(Some(j)) => j.kind,
                        Ok(None) => continue,
                        Err(e) => {
                            error!(error = %e, %job_id, "get_job failed during cancel sweep");
                            continue;
                        }
                    };
                    if let Some(handler) = registry.get(&kind) {
                        if let Err(e) = handler.cancel(job_id).await {
                            warn!(error = %e, %job_id, "handler.cancel reported error");
                        }
                    }
                    if let Err(e) = store.clear_pid_and_kill_flag(job_id).await {
                        error!(error = %e, %job_id, "clear_pid_and_kill_flag failed");
                    }
                }
            }
            Err(e) => error!(error = %e, "find_cancelled_with_pid failed"),
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
