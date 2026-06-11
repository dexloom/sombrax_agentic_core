//! Public façade — wires a [`Store`], a [`HandlerRegistry`], and the background
//! workers into a single object.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::context::CancelSignal;

use super::error::StoreError;
use super::model::{
    Job, JobId, LogRange, PipelineId, PipelineSpec, Run, RunId, RunOutcome, RunStatus,
};
use super::registry::HandlerRegistry;
use super::store::{Store, StoreEvent};
use super::worker::{cancel_sweep, dispatcher, orphan};

/// Runtime tunables.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Idle back-off when no jobs are dispatchable.
    pub dispatch_tick: Duration,
    /// Cancel-sweep poll interval.
    pub cancel_sweep_tick: Duration,
    /// Orphan worker poll interval.
    pub orphan_tick: Duration,
    /// Backstop maximum job runtime; jobs Running past this for any reason
    /// (panic, stalled handler) are finalised as `Timeout` by the orphan worker.
    /// Set to `None` to disable the orphan worker entirely.
    pub orphan_max_runtime: Option<Duration>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            dispatch_tick: Duration::from_millis(25),
            cancel_sweep_tick: Duration::from_millis(50),
            orphan_tick: Duration::from_secs(5),
            orphan_max_runtime: Some(Duration::from_secs(300)),
        }
    }
}

/// The thing consumers use.
///
/// Cheap to clone; internal state is `Arc`-shared.
#[derive(Clone)]
pub struct Runtime {
    store: Arc<dyn Store>,
    registry: HandlerRegistry,
    config: RuntimeConfig,
    cancel: CancelSignal,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl Runtime {
    /// Construct a new runtime. Workers are NOT started — call [`Runtime::start`].
    pub fn new(store: Arc<dyn Store>, registry: HandlerRegistry, config: RuntimeConfig) -> Self {
        Self {
            store,
            registry,
            config,
            cancel: CancelSignal::new(),
            handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Convenience: build with defaults.
    pub fn with_defaults(store: Arc<dyn Store>, registry: HandlerRegistry) -> Self {
        Self::new(store, registry, RuntimeConfig::default())
    }

    /// Access the underlying store (advanced consumers / tests).
    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    /// Access the registry.
    pub fn registry(&self) -> &HandlerRegistry {
        &self.registry
    }

    /// Register a pipeline spec. Idempotent.
    pub async fn put_pipeline(&self, spec: PipelineSpec) -> Result<PipelineId, StoreError> {
        let id = spec.id;
        self.store.put_pipeline(spec).await?;
        Ok(id)
    }

    /// Fetch a pipeline spec.
    pub async fn get_pipeline(&self, id: PipelineId) -> Result<Option<PipelineSpec>, StoreError> {
        self.store.get_pipeline(id).await
    }

    /// Submit a new run.
    pub async fn submit(&self, pipeline: PipelineId, inputs: Value) -> Result<RunId, StoreError> {
        self.store.create_run(pipeline, inputs).await
    }

    /// Block until the run reaches a terminal status, then return a snapshot.
    /// Uses [`Store::subscribe`] so there is no polling.
    pub async fn wait(&self, run_id: RunId) -> Result<RunOutcome, StoreError> {
        // Fast-path: if already terminal, return immediately.
        if let Some(run) = self.store.get_run(run_id).await? {
            if matches!(
                run.status,
                RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
            ) {
                return Ok(RunOutcome {
                    status: run.status,
                    jobs: self.store.list_run_jobs(run_id).await?,
                });
            }
        }

        // Subscribe before re-checking (avoids missing the event between fast-path and subscribe).
        let mut events = self.store.subscribe().await;
        if let Some(run) = self.store.get_run(run_id).await? {
            if matches!(
                run.status,
                RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
            ) {
                return Ok(RunOutcome {
                    status: run.status,
                    jobs: self.store.list_run_jobs(run_id).await?,
                });
            }
        }

        while let Some(ev) = events.next().await {
            match ev {
                StoreEvent::RunTerminal { run_id: r, status }
                    if r == run_id
                        && matches!(
                            status,
                            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
                        ) =>
                {
                    return Ok(RunOutcome {
                        status,
                        jobs: self.store.list_run_jobs(run_id).await?,
                    });
                }
                _ => {}
            }
        }
        Err(StoreError::Backend("event stream closed".into()))
    }

    /// Cancel a run.
    pub async fn cancel(&self, run_id: RunId) -> Result<(), StoreError> {
        self.store.cancel_run(run_id).await
    }

    /// Read a job's logs.
    pub async fn read_log(
        &self,
        job_id: JobId,
        range: LogRange,
    ) -> Result<Vec<String>, StoreError> {
        self.store.read_log(job_id, range).await
    }

    /// Snapshot a job.
    pub async fn get_job(&self, job_id: JobId) -> Result<Option<Job>, StoreError> {
        self.store.get_job(job_id).await
    }

    /// Snapshot a run.
    pub async fn get_run(&self, run_id: RunId) -> Result<Option<Run>, StoreError> {
        self.store.get_run(run_id).await
    }

    /// Snapshot every job in a run.
    pub async fn list_run_jobs(&self, run_id: RunId) -> Result<Vec<Job>, StoreError> {
        self.store.list_run_jobs(run_id).await
    }

    /// Spawn the background workers (dispatcher + cancel sweep + optional orphan).
    /// Idempotent: a second call is a no-op.
    pub async fn start(&self) {
        let mut h = self.handles.lock().await;
        if !h.is_empty() {
            return;
        }
        let store = self.store.clone();
        let registry = self.registry.clone();
        let cfg = self.config.clone();
        let cancel = self.cancel.clone();

        let s1 = store.clone();
        let r1 = registry.clone();
        let cancel1 = cancel.clone();
        let dispatch_tick = cfg.dispatch_tick;
        h.push(tokio::spawn(async move {
            dispatcher::run_dispatcher(s1, r1, dispatch_tick, cancel1).await;
        }));

        let s2 = store.clone();
        let r2 = registry.clone();
        let cancel2 = cancel.clone();
        let sweep_tick = cfg.cancel_sweep_tick;
        h.push(tokio::spawn(async move {
            cancel_sweep::run_cancel_sweep(s2, r2, sweep_tick, cancel2).await;
        }));

        if let Some(max_runtime) = cfg.orphan_max_runtime {
            let s3 = store.clone();
            let cancel3 = cancel.clone();
            let orphan_tick = cfg.orphan_tick;
            h.push(tokio::spawn(async move {
                orphan::run_orphan(s3, max_runtime, orphan_tick, cancel3).await;
            }));
        }
    }

    /// Signal all workers to exit and await them.
    pub async fn shutdown(&self) {
        self.cancel.cancel();
        let handles: Vec<_> = std::mem::take(&mut *self.handles.lock().await);
        for h in handles {
            let _ = h.await;
        }
    }
}
