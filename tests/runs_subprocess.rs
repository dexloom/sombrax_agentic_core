//! End-to-end test for [`sombrax_agentic_core::runs::SubprocessHandler`].
//!
//! Uses `/bin/sh -c` so we don't need to ship a separate test binary.

#![cfg(all(feature = "runs", unix))]

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use sombrax_agentic_core::runs::{
    ArgTpl, EnvTpl, HandlerRegistryBuilder, InMemoryStore, JobId, JobStatus, JobTemplate,
    JoinPolicy, OutputParser, PipelineBuilder, PipelineId, RunStatus, Runtime, RuntimeConfig,
    SubprocessHandler,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subprocess_emits_output_result_sentinel() {
    // /bin/sh -c 'echo hello; echo "OUTPUT_RESULT:/tmp/foo.json"'
    let handler = SubprocessHandler::new("shell", "/bin/sh")
        .arg(ArgTpl::lit("-c"))
        .arg(ArgTpl::input("script"))
        .parser(OutputParser::OutputResultSentinel);

    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(handler))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(5),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(100),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let a = job("shell");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("subproc")
        .with_id(pid)
        .with_known_kinds(["shell"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();

    let inputs = json!({
        "script": "echo hello; echo 'OUTPUT_RESULT:/tmp/foo.json'"
    });
    let run_id = runtime.submit(pid, inputs).await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("subprocess test timed out")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Completed);

    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Completed);
    let out = aj.output.expect("output recorded");
    assert_eq!(out.get("output_file"), Some(&json!("/tmp/foo.json")));

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subprocess_failure_propagates() {
    let handler = SubprocessHandler::new("shell", "/bin/sh")
        .arg(ArgTpl::lit("-c"))
        .arg(ArgTpl::lit("exit 7"));

    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(handler))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(5),
            cancel_sweep_tick: Duration::from_millis(20),
            orphan_tick: Duration::from_millis(100),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let a = job("shell");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("subproc-fail")
        .with_id(pid)
        .with_known_kinds(["shell"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();

    let run_id = runtime.submit(pid, json!({})).await.unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("subprocess fail timed out")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Failed);
    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Failed);
    assert!(aj.error.unwrap_or_default().contains("status"));

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subprocess_cancel_sigterm() {
    // Long-running shell process; cancel_run should SIGTERM it.
    let handler = SubprocessHandler::new("shell", "/bin/sh")
        .arg(ArgTpl::lit("-c"))
        .arg(ArgTpl::lit("sleep 60"));

    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(handler))
        .build()
        .unwrap();
    let runtime = Runtime::new(
        store.clone() as Arc<_>,
        registry,
        RuntimeConfig {
            dispatch_tick: Duration::from_millis(5),
            cancel_sweep_tick: Duration::from_millis(10),
            orphan_tick: Duration::from_millis(100),
            orphan_max_runtime: None,
        },
    );
    runtime.start().await;

    let a = job("shell");
    let pid_pipeline = PipelineId::new();
    let spec = PipelineBuilder::new("subproc-cancel")
        .with_id(pid_pipeline)
        .with_known_kinds(["shell"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();

    let run_id = runtime.submit(pid_pipeline, json!({})).await.unwrap();
    // Wait until pid is registered (handler had time to spawn).
    for _ in 0..200 {
        if let Some(j) = runtime.get_job(a.id).await.unwrap() {
            if j.pid.is_some() && matches!(j.status, JobStatus::Running) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    runtime.cancel(run_id).await.unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("cancel did not terminalise run within 5s")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Cancelled);
    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(aj.status, JobStatus::Cancelled);

    runtime.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subprocess_env_and_input_templating() {
    // Confirm the input dotted-path resolves into both args and env.
    let handler = SubprocessHandler::new("shell", "/bin/sh")
        .arg(ArgTpl::lit("-c"))
        .arg(ArgTpl::lit(
            "printf '%s' \"$GREETING-$NAME\"; printf '\\nOUTPUT_RESULT:%s\\n' \"$NAME-out\"",
        ))
        .env("GREETING", EnvTpl::Literal("hi".into()))
        .env("NAME", EnvTpl::FromInput("user.name".into()));

    let store: Arc<InMemoryStore> = Arc::new(InMemoryStore::new());
    let registry = HandlerRegistryBuilder::new()
        .register(Arc::new(handler))
        .build()
        .unwrap();
    let runtime = Runtime::new(store.clone() as Arc<_>, registry, RuntimeConfig::default());
    runtime.start().await;

    let a = job("shell");
    let pid = PipelineId::new();
    let spec = PipelineBuilder::new("subproc-env")
        .with_id(pid)
        .with_known_kinds(["shell"])
        .job(a.clone())
        .build()
        .unwrap();
    runtime.put_pipeline(spec).await.unwrap();
    let run_id = runtime
        .submit(pid, json!({"user": {"name": "ada"}}))
        .await
        .unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(5), runtime.wait(run_id))
        .await
        .expect("subprocess env test timed out")
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Completed);

    let aj = runtime.get_job(a.id).await.unwrap().unwrap();
    assert_eq!(
        aj.output.as_ref().and_then(|v| v.get("output_file")),
        Some(&json!("ada-out")),
        "input.user.name should have been interpolated into env and surfaced via OUTPUT_RESULT"
    );

    runtime.shutdown().await;
}
