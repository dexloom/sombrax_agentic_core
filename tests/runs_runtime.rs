//! End-to-end runtime tests: submit / wait / cancel / per-kind cap / timeout.

#![cfg(feature = "runs")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use sombrax_agentic_core::runs::{
    EdgeCondition, EdgeId, EdgeTemplate, HandlerRegistryBuilder, InMemoryStore, JobContext,
    JobError, JobHandler, JobId, JobOutput, JobStatus, JobTemplate, JoinPolicy, MergePolicy,
    OutputProjection, PipelineBuilder, PipelineId, RunStatus, Runtime, RuntimeConfig,
};

fn job(kind: &str) -> JobTemplate {
    JobTemplate {
        id: JobId::new(),
        kind: kind.into(),
        default_inputs: serde_json::Value::Null,
        bundle_id: None,
        join_policy: JoinPolicy::AllRequired,
    }
}

fn edge(from: JobId, to: JobId, target: &str) -> EdgeTemplate {
    EdgeTemplate {
        id: EdgeId::new(),
        from,
        to,
        source: OutputProjection::Whole,
        target: target.into(),
        condition: EdgeCondition::Always,
        merge: MergePolicy::LastWriteWins,
        required: true,
    }
}

/// Echo handler: returns its inputs as the output.
struct Echo {
    kind: &'static str,
}
#[async_trait]
impl JobHandler for Echo {
    fn kind(&self) -> &str {
        self.kind
    }
    async fn run(&self, ctx: JobContext) -> Result<JobOutput, JobError> {
        ctx.log.info(format!("echo {}", self.kind)).await;
        Ok(JobOutput::new(ctx.inputs))
    }
}

/// Slow handler: sleeps then returns the configured value.
struct Slow {
    kind: &'static str,
    delay: Duration,
    counter: Arc<AtomicUsize>,
}
#[async_trait]
impl JobHandler for Slow {
    fn kind(&self) -> &str {
        self.kind
    }
    fn max_concurrent(&self) -> Option<usize> {
        Some(1)
    }
    async fn run(&self, _ctx: JobContext) -> Result<JobOutput, JobError> {
        let now = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        let max = self
            .counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some(v.max(now)))
            .unwrap_or(0);
        let _ = max;
        tokio::time::sleep(self.delay).await;
        self.counter.fetch_sub(1, Ordering::SeqCst);
        Ok(JobOutput::new(json!({"slept": true})))
    }
}

/// Forever handler: ignores cancel for this test, used to exercise dispatcher timeout.
struct Forever;
#[async_trait]
impl JobHandler for Forever {
    fn kind(&self) -> &str {
        "forever"
    }
    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(50))
    }
    async fn run(&self, _ctx: JobContext) -> Result<JobOutput, JobError> {
        // Sleep beyond the 50ms timeout the dispatcher will impose via tokio::time::timeout.
        tokio::time::sleep(Duration::from_secs(60)).await;
        Ok(JobOutput::new(json!({})))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_job_submit_wait() {
    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(Echo { kind: "echo" }))
        .build()
        .unwrap();

    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(5),
            cancel_sweep_tick: Duration::from_millis(5),
            orphan_tick: Duration::from_millis(50),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let pid = PipelineId::new();
    let a = job("echo");
    let spec = PipelineBuilder::new("p")
        .with_id(pid)
        .with_known_kinds(["echo"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();
    let run_id = runtime
        .submit(pid, json!({"hello": "world"}))
        .await
        .unwrap();

    let outcome = runtime.wait(run_id).await.unwrap();
    assert_eq!(outcome.status, RunStatus::Completed);
    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Completed);
    assert_eq!(aj.output, Some(json!({"hello": "world"})));

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn diamond_e2e() {
    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(Echo { kind: "root" }))
        .register(Arc::new(Echo { kind: "branch" }))
        .register(Arc::new(Echo { kind: "tail" }))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(2),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(100),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let root = job("root");
    let b1 = job("branch");
    let b2 = job("branch");
    let tail = job("tail");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("diamond")
        .with_id(pid)
        .with_known_kinds(["root", "branch", "tail"])
        .job(root.clone())
        .job(b1.clone())
        .job(b2.clone())
        .job(tail.clone())
        .edge(edge(root.id, b1.id, "from_root"))
        .edge(edge(root.id, b2.id, "from_root"))
        .edge(edge(b1.id, tail.id, "left"))
        .edge(edge(b2.id, tail.id, "right"))
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();

    let run_id = runtime.submit(pid, json!({"seed": 42})).await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("run did not complete in 5s")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Completed);
    let tj = runtime.get_job(tail.id).await.unwrap().unwrap();
    assert_eq!(tj.status, JobStatus::Completed);

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn handler_timeout_yields_timeout_terminal() {
    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(Forever))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(2),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(50),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let a = job("forever");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("p")
        .with_id(pid)
        .with_known_kinds(["forever"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();
    let run_id = runtime.submit(pid, json!({})).await.unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("run never terminalised")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Failed); // Timeout terminal counts as Failed.
    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Timeout);

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn per_kind_concurrency_cap_is_honoured() {
    let counter = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));
    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let slow = Slow {
        kind: "slow",
        delay: Duration::from_millis(100),
        counter: counter.clone(),
    };
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(slow))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(2),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(100),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    // Independent pipelines so jobs are root + Queued from the start.
    let mut pids = Vec::new();
    let mut jobs = Vec::new();
    for _ in 0..4u32 {
        let a = job("slow");
        let pid = PipelineId::new();
        let spec = PipelineBuilder::new("conc")
            .with_id(pid)
            .with_known_kinds(["slow"])
            .job(a.clone())
            .build()
            .unwrap();
        runtime.put_pipeline(spec).await.unwrap();
        pids.push(pid);
        jobs.push(a);
    }

    // Watcher task: every 10ms snapshot the counter and remember the maximum
    // observed concurrent runs of `slow`.
    let watcher_max = max_seen.clone();
    let watcher_counter = counter.clone();
    let watch = tokio::spawn(async move {
        for _ in 0..150 {
            let cur = watcher_counter.load(Ordering::SeqCst);
            let mut peak = watcher_max.load(Ordering::SeqCst);
            while cur > peak {
                match watcher_max.compare_exchange(peak, cur, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    let mut runs = Vec::new();
    for pid in &pids {
        runs.push(runtime.submit(*pid, json!({})).await.unwrap());
    }
    for r in runs {
        runtime.wait(r).await.unwrap();
    }
    watch.abort();

    assert!(
        max_seen.load(Ordering::SeqCst) <= 1,
        "max_concurrent=1 must cap to one concurrent slow run, observed {}",
        max_seen.load(Ordering::SeqCst)
    );

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_run_kills_active_pipeline() {
    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let slow = Slow {
        kind: "slow",
        delay: Duration::from_secs(60),
        counter: Arc::new(AtomicUsize::new(0)),
    };
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(slow))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(2),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(50),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let a = job("slow");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("p")
        .with_id(pid)
        .with_known_kinds(["slow"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();
    let run_id = runtime.submit(pid, json!({})).await.unwrap();

    // Wait until the job is Running, then cancel.
    for _ in 0..200 {
        if let Some(j) = runtime.get_job(a.id).await.unwrap() {
            if matches!(j.status, JobStatus::Running) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    runtime.cancel(run_id).await.unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("cancel did not terminalise run in 5s")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Cancelled);
    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Cancelled);

    runtime.shutdown().await;
}
