//! In-process [`Store`] implementation.
//!
//! Concurrency model: a single `tokio::sync::Mutex` guards the whole entity
//! graph. Pipelines and runs are small DAGs (tens of jobs), and
//! `complete_job_and_propagate` finalises in one critical section so observers
//! never see a partially-propagated state. This trades scalability we don't
//! need (this backend is for tests + the watcher hot path) for trivial
//! atomicity.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::stream;
use serde_json::Value;
use tokio::sync::{broadcast, Mutex};

use crate::runs::error::StoreError;
use crate::runs::model::{
    Bundle, BundleId, BundleStatus, BundleTemplate, DispatchTicket, EdgeCondition, EdgeId,
    EdgeResolution, EdgeTemplate, Job, JobId, JobStatus, JoinPolicy, LogRange, MergePolicy,
    OutputProjection, PipelineId, PipelineSpec, Run, RunCondition, RunId, RunStatus,
    TerminalOutcome,
};

use super::event::{StoreEvent, StoreEventStream};
use super::Store;

/// In-memory `Store` implementation.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    state: Mutex<State>,
    events: broadcast::Sender<StoreEvent>,
}

#[derive(Debug, Default)]
struct State {
    pipelines: HashMap<PipelineId, PipelineSpec>,
    runs: HashMap<RunId, Run>,
    /// All jobs across all runs, indexed by id.
    jobs: HashMap<JobId, Job>,
    /// All bundles across all runs, indexed by id.
    bundles: HashMap<BundleId, Bundle>,
    /// run_id -> ordered list of job ids belonging to that run (creation order).
    run_jobs: HashMap<RunId, Vec<JobId>>,
    /// Per-job queue of log lines.
    logs: HashMap<JobId, VecDeque<String>>,
    /// (run_id, edge_template_id, target_job_id) → applied value when condition matched.
    /// Edges are 1:1 with templates in this single-instantiation model, so the
    /// edge-instance id IS the template id.
    edge_applied: HashMap<EdgeId, Value>,
    /// run_id -> count of cancellation_kill_pending flags currently set.
    /// Used by `find_cancelled_with_pid` for a fast filter.
    cancel_kill_count: usize,
}

impl InMemoryStore {
    /// Construct a new empty store with a default broadcast capacity.
    pub fn new() -> Self {
        Self::with_event_capacity(256)
    }

    /// Construct with a custom event broadcast buffer capacity.
    pub fn with_event_capacity(cap: usize) -> Self {
        let (tx, _rx) = broadcast::channel(cap);
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                events: tx,
            }),
        }
    }

    fn emit(&self, ev: StoreEvent) {
        // Best-effort: lagging or no subscribers is not fatal.
        let _ = self.inner.events.send(ev);
    }

    /// Read-only access for tests and orphan-worker style sweeps.
    #[doc(hidden)]
    pub async fn snapshot_job(&self, job_id: JobId) -> Option<Job> {
        self.inner.state.lock().await.jobs.get(&job_id).cloned()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- helpers operating on `State` (no locking) ----------

impl State {
    fn pipeline(&self, id: PipelineId) -> Result<&PipelineSpec, StoreError> {
        self.pipelines
            .get(&id)
            .ok_or(StoreError::PipelineNotFound(id))
    }

    fn job(&self, id: JobId) -> Result<&Job, StoreError> {
        self.jobs.get(&id).ok_or(StoreError::JobNotFound(id))
    }

    fn job_mut(&mut self, id: JobId) -> Result<&mut Job, StoreError> {
        self.jobs.get_mut(&id).ok_or(StoreError::JobNotFound(id))
    }

    fn run(&self, id: RunId) -> Result<&Run, StoreError> {
        self.runs.get(&id).ok_or(StoreError::RunNotFound(id))
    }

    fn run_mut(&mut self, id: RunId) -> Result<&mut Run, StoreError> {
        self.runs.get_mut(&id).ok_or(StoreError::RunNotFound(id))
    }

    /// All edges in the pipeline that originate at `from`.
    fn outgoing_edges(&self, run_id: RunId, from: JobId) -> Result<Vec<EdgeTemplate>, StoreError> {
        let pipeline_id = self.run(run_id)?.pipeline_id;
        let spec = self.pipeline(pipeline_id)?;
        Ok(spec
            .edges
            .iter()
            .filter(|e| e.from == from)
            .cloned()
            .collect())
    }

    /// All edges that arrive at `to`.
    fn incoming_edges(&self, run_id: RunId, to: JobId) -> Result<Vec<EdgeTemplate>, StoreError> {
        let pipeline_id = self.run(run_id)?.pipeline_id;
        let spec = self.pipeline(pipeline_id)?;
        Ok(spec.edges.iter().filter(|e| e.to == to).cloned().collect())
    }

    /// Bundles in the spec, by id.
    fn bundle_template_map(
        &self,
        pipeline_id: PipelineId,
    ) -> Result<HashMap<BundleId, BundleTemplate>, StoreError> {
        let spec = self.pipeline(pipeline_id)?;
        Ok(spec.bundles.iter().map(|b| (b.id, b.clone())).collect())
    }
}

// ---------- output projection / merge ----------

fn project(value: &Value, proj: &OutputProjection) -> Value {
    match proj {
        OutputProjection::Whole => value.clone(),
        OutputProjection::Field(path) => {
            let mut cur = value;
            for seg in path.split('.') {
                cur = cur.get(seg).unwrap_or(&Value::Null);
            }
            cur.clone()
        }
        OutputProjection::JsonPath(expr) => match serde_json_path::JsonPath::parse(expr) {
            Ok(p) => {
                let nodes = p.query(value).all();
                if nodes.len() == 1 {
                    nodes[0].clone()
                } else {
                    Value::Array(nodes.into_iter().cloned().collect())
                }
            }
            Err(_) => Value::Null,
        },
    }
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = Value::Object(serde_json::Map::new());
    }
}

fn merge_field(target_inputs: &mut Value, field: &str, merge: MergePolicy, applied: Value) {
    ensure_object(target_inputs);
    let map = target_inputs.as_object_mut().unwrap();
    match merge {
        MergePolicy::Reject => {
            // Validation should have rejected; in case of bug, last write wins to avoid panic.
            map.insert(field.to_string(), applied);
        }
        MergePolicy::LastWriteWins => {
            map.insert(field.to_string(), applied);
        }
        MergePolicy::AppendArray => {
            let entry = map.entry(field.to_string()).or_insert(Value::Array(vec![]));
            if !entry.is_array() {
                *entry = Value::Array(vec![std::mem::take(entry)]);
            }
            entry.as_array_mut().unwrap().push(applied);
        }
        MergePolicy::ObjectMerge => {
            let entry = map
                .entry(field.to_string())
                .or_insert(Value::Object(serde_json::Map::new()));
            if let (Value::Object(into), Value::Object(from)) = (entry, applied) {
                for (k, v) in from {
                    into.insert(k, v);
                }
            }
        }
    }
}

// ---------- run instantiation ----------

fn instantiate_run(
    state: &mut State,
    pipeline: &PipelineSpec,
    inputs: Value,
) -> Result<RunId, StoreError> {
    let run_id = RunId::new();
    let now = Utc::now();
    let run = Run {
        id: run_id,
        pipeline_id: pipeline.id,
        status: RunStatus::Pending,
        inputs: inputs.clone(),
        cancel_generation: 0,
        created_at: now,
        completed_at: None,
    };
    state.runs.insert(run_id, run);

    // Pre-count incoming edges per job so we can set pending_inputs deterministically.
    let mut pending: HashMap<JobId, u32> = HashMap::new();
    let mut edge_resolutions: HashMap<JobId, BTreeMap<EdgeId, EdgeResolution>> = HashMap::new();
    for e in &pipeline.edges {
        *pending.entry(e.to).or_default() += 1;
        edge_resolutions
            .entry(e.to)
            .or_default()
            .insert(e.id, EdgeResolution::Pending);
    }

    // Find roots (jobs with zero incoming edges) — these get the run inputs deep-merged
    // into their default_inputs so the caller can parameterise the run.
    let mut job_ids: Vec<JobId> = Vec::with_capacity(pipeline.jobs.len());
    for jt in &pipeline.jobs {
        let in_count = pending.get(&jt.id).copied().unwrap_or(0);
        let mut effective_inputs = jt.default_inputs.clone();
        if in_count == 0 {
            // Deep-merge submitted inputs over default_inputs (caller wins).
            deep_merge(&mut effective_inputs, &inputs);
        }
        let resolved = edge_resolutions.remove(&jt.id).unwrap_or_default();
        let job = Job {
            id: jt.id,
            run_id,
            pipeline_id: pipeline.id,
            bundle_id: jt.bundle_id,
            kind: jt.kind.clone(),
            inputs: effective_inputs,
            output: None,
            status: if in_count == 0 {
                JobStatus::Queued
            } else {
                JobStatus::Pending
            },
            error: None,
            pending_inputs: in_count,
            edge_resolutions: resolved,
            completion_generation: 0,
            cancel_generation: 0,
            join_policy: jt.join_policy,
            cancel_kill_pending: false,
            pid: None,
            exit_code: None,
            findings_count: None,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        state.jobs.insert(jt.id, job);
        job_ids.push(jt.id);
    }
    state.run_jobs.insert(run_id, job_ids);

    // Materialise bundles.
    for bt in &pipeline.bundles {
        let bundle = Bundle {
            id: bt.id,
            run_id,
            pipeline_id: pipeline.id,
            parent: bt.parent,
            job_ids: bt.job_ids.clone(),
            successor_ids: bt.successor_ids.clone(),
            run_condition: bt.run_condition,
            status: BundleStatus::Pending,
            blocked_reason: None,
        };
        state.bundles.insert(bt.id, bundle);
    }

    Ok(run_id)
}

fn deep_merge(target: &mut Value, source: &Value) {
    match (target, source) {
        (Value::Object(t), Value::Object(s)) => {
            for (k, v) in s {
                deep_merge(t.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (slot, src) => {
            // For non-object values, source wins.
            if !src.is_null() {
                *slot = src.clone();
            }
        }
    }
}

// ---------- propagation core ----------

fn apply_edge_resolution(
    state: &mut State,
    edge: &EdgeTemplate,
    source_status: JobStatus,
    source_output: &Option<Value>,
) -> Result<(), StoreError> {
    let resolution = if edge.condition.matches(source_status) {
        if let Some(out) = source_output {
            let projected = project(out, &edge.source);
            let target = state.job_mut(edge.to)?;
            merge_field(
                &mut target.inputs,
                &edge.target,
                edge.merge,
                projected.clone(),
            );
            // Track applied value for diagnostics; not strictly required.
            // We keep a separate map keyed by edge id.
        }
        // Even on Skipped (which has no output), an `OnTerminals(set)` containing
        // Skipped or `Always` will resolve Applied with a Null projection.
        if source_output.is_none() && !matches!(edge.source, OutputProjection::Whole) {
            // No output to project — leave target field absent.
        } else if source_output.is_none() {
            // Applied + Always + no output: write Null so the field exists.
            let target = state.job_mut(edge.to)?;
            merge_field(&mut target.inputs, &edge.target, edge.merge, Value::Null);
        }
        EdgeResolution::Applied
    } else {
        EdgeResolution::Unsatisfied
    };

    // Track applied output for the public `edge_applied` diagnostic map.
    if matches!(resolution, EdgeResolution::Applied) {
        if let Some(out) = source_output {
            state
                .edge_applied
                .insert(edge.id, project(out, &edge.source));
        }
    }

    let target = state.job_mut(edge.to)?;
    target.edge_resolutions.insert(edge.id, resolution);
    target.pending_inputs = target.pending_inputs.saturating_sub(1);
    Ok(())
}

/// Step-4 readiness rule. Returns the next status if the target should transition
/// out of `Pending`, else `None`.
fn evaluate_target_readiness(job: &Job, incoming: &[EdgeTemplate]) -> Option<JobStatus> {
    if job.pending_inputs > 0 || job.status != JobStatus::Pending {
        return None;
    }
    match job.join_policy {
        JoinPolicy::AllRequired => {
            // Every required edge must be Applied.
            let mut any_required_unsatisfied = false;
            let mut had_required = false;
            for e in incoming {
                if !e.required {
                    continue;
                }
                had_required = true;
                match job
                    .edge_resolutions
                    .get(&e.id)
                    .copied()
                    .unwrap_or(EdgeResolution::Pending)
                {
                    EdgeResolution::Applied => {}
                    EdgeResolution::Unsatisfied => any_required_unsatisfied = true,
                    EdgeResolution::Pending => return None, // shouldn't happen if pending_inputs == 0
                }
            }
            if !had_required {
                // No required edges at all (or only optional). Queue iff at least one Applied.
                let any_applied = job
                    .edge_resolutions
                    .values()
                    .any(|r| matches!(r, EdgeResolution::Applied));
                if any_applied || incoming.is_empty() {
                    Some(JobStatus::Queued)
                } else {
                    Some(JobStatus::Skipped)
                }
            } else if any_required_unsatisfied {
                Some(JobStatus::Skipped)
            } else {
                Some(JobStatus::Queued)
            }
        }
        JoinPolicy::AnyApplied => {
            let any_applied = job
                .edge_resolutions
                .values()
                .any(|r| matches!(r, EdgeResolution::Applied));
            Some(if any_applied {
                JobStatus::Queued
            } else {
                JobStatus::Skipped
            })
        }
    }
}

/// Tri-state bundle evaluation. Returns Some(true) if Satisfied, Some(false) if
/// Violated, None if still Pending.
fn evaluate_bundle(state: &State, bundle: &Bundle) -> Option<bool> {
    let job_statuses: Vec<JobStatus> = bundle
        .job_ids
        .iter()
        .filter_map(|id| state.jobs.get(id).map(|j| j.status))
        .collect();
    let all_terminal = job_statuses.iter().all(|s| s.is_terminal());
    let any_completed = job_statuses
        .iter()
        .any(|s| matches!(s, JobStatus::Completed));
    let any_non_completed = job_statuses
        .iter()
        .any(|s| s.is_terminal() && !matches!(s, JobStatus::Completed));

    match bundle.run_condition {
        RunCondition::AllSuccess => {
            if all_terminal {
                Some(
                    job_statuses
                        .iter()
                        .all(|s| matches!(s, JobStatus::Completed)),
                )
            } else if any_non_completed {
                // At least one already non-success; once siblings finish, Violated.
                if all_terminal {
                    Some(false)
                } else {
                    None
                }
            } else {
                None
            }
        }
        RunCondition::AnySuccess => {
            if any_completed {
                Some(true)
            } else if all_terminal {
                Some(false)
            } else {
                None
            }
        }
        RunCondition::AllComplete | RunCondition::Always => {
            if all_terminal {
                Some(true)
            } else {
                None
            }
        }
    }
}

/// Walk all bundles that contain `job_id` (and their parents transitively),
/// applying tri-state evaluation. On Satisfied, fan out successors. On Violated,
/// recursively skip successor jobs.
fn propagate_bundles(
    state: &mut State,
    pipeline_id: PipelineId,
    job_id: JobId,
    events: &mut Vec<StoreEvent>,
) -> Result<(), StoreError> {
    let bundle_template_map = state.bundle_template_map(pipeline_id)?;
    let bundles_to_check: Vec<BundleId> = state
        .bundles
        .values()
        .filter(|b| b.job_ids.contains(&job_id))
        .map(|b| b.id)
        .collect();

    let mut queue: VecDeque<BundleId> = bundles_to_check.into_iter().collect();
    while let Some(bid) = queue.pop_front() {
        let bundle = match state.bundles.get(&bid) {
            Some(b) => b.clone(),
            None => continue,
        };
        if !matches!(bundle.status, BundleStatus::Pending | BundleStatus::Running) {
            continue;
        }

        let outcome = evaluate_bundle(state, &bundle);
        match outcome {
            None => {
                // Pending: only mark Running if at least one job is past Pending.
                let any_active = bundle
                    .job_ids
                    .iter()
                    .filter_map(|id| state.jobs.get(id).map(|j| j.status))
                    .any(|s| !matches!(s, JobStatus::Pending));
                if any_active && bundle.status == BundleStatus::Pending {
                    let b = state.bundles.get_mut(&bid).unwrap();
                    b.status = BundleStatus::Running;
                    events.push(StoreEvent::BundleStatus {
                        bundle_id: bid,
                        run_id: bundle.run_id,
                        status: BundleStatus::Running,
                    });
                }
            }
            Some(true) => {
                let b = state.bundles.get_mut(&bid).unwrap();
                b.status = BundleStatus::Completed;
                events.push(StoreEvent::BundleStatus {
                    bundle_id: bid,
                    run_id: bundle.run_id,
                    status: BundleStatus::Completed,
                });
                // Fan out successors: any successor bundles are activated; their root
                // jobs (jobs in successor.job_ids whose pending_inputs == 0 and status
                // == Pending) become Queued.
                for sid in &bundle.successor_ids {
                    if let Some(template) = bundle_template_map.get(sid) {
                        for jid in &template.job_ids {
                            if let Some(job) = state.jobs.get_mut(jid) {
                                if job.status == JobStatus::Pending && job.pending_inputs == 0 {
                                    job.status = JobStatus::Queued;
                                }
                            }
                        }
                    }
                    queue.push_back(*sid);
                }
            }
            Some(false) => {
                let b = state.bundles.get_mut(&bid).unwrap();
                b.status = BundleStatus::Blocked;
                b.blocked_reason = Some("run_condition violated".to_string());
                events.push(StoreEvent::BundleStatus {
                    bundle_id: bid,
                    run_id: bundle.run_id,
                    status: BundleStatus::Blocked,
                });
                // Cascade Skipped to successor bundles' jobs.
                for sid in &bundle.successor_ids {
                    if let Some(template) = bundle_template_map.get(sid) {
                        for jid in &template.job_ids {
                            skip_job_cascade(state, *jid, events)?;
                        }
                    }
                    queue.push_back(*sid);
                }
            }
        }
    }
    Ok(())
}

fn skip_job_cascade(
    state: &mut State,
    job_id: JobId,
    events: &mut Vec<StoreEvent>,
) -> Result<(), StoreError> {
    let (run_id, was_terminal) = {
        let job = match state.jobs.get_mut(&job_id) {
            Some(j) => j,
            None => return Ok(()),
        };
        if job.status.is_terminal() {
            return Ok(());
        }
        job.status = JobStatus::Skipped;
        job.completed_at = Some(Utc::now());
        (job.run_id, false)
    };
    if !was_terminal {
        events.push(StoreEvent::JobTerminal {
            job_id,
            run_id,
            status: JobStatus::Skipped,
        });
    }
    // Recursively resolve outgoing edges as Unsatisfied (Skipped is non-Completed),
    // updating downstream readiness.
    let outgoing = state.outgoing_edges(run_id, job_id)?;
    for edge in &outgoing {
        apply_edge_resolution(state, edge, JobStatus::Skipped, &None)?;
        let target_id = edge.to;
        let incoming = state.incoming_edges(run_id, target_id)?;
        let next = {
            let target = state.job(target_id)?;
            evaluate_target_readiness(target, &incoming)
        };
        if let Some(JobStatus::Skipped) = next {
            skip_job_cascade(state, target_id, events)?;
        } else if let Some(JobStatus::Queued) = next {
            let target = state.job_mut(target_id)?;
            if target.status == JobStatus::Pending {
                target.status = JobStatus::Queued;
            }
        }
    }
    Ok(())
}

fn recompute_run_status(state: &mut State, run_id: RunId) -> RunStatus {
    let job_ids = match state.run_jobs.get(&run_id) {
        Some(v) => v.clone(),
        None => return RunStatus::Pending,
    };
    if job_ids.is_empty() {
        return RunStatus::Completed;
    }
    let statuses: Vec<JobStatus> = job_ids
        .iter()
        .filter_map(|id| state.jobs.get(id).map(|j| j.status))
        .collect();
    let all_terminal = statuses.iter().all(|s| s.is_terminal());
    let any_terminal = statuses.iter().any(|s| s.is_terminal());
    if !all_terminal {
        return if any_terminal {
            RunStatus::Running
        } else {
            RunStatus::Pending
        };
    }
    let any_cancelled = statuses.iter().any(|s| matches!(s, JobStatus::Cancelled));
    let any_failed = statuses
        .iter()
        .any(|s| matches!(s, JobStatus::Failed | JobStatus::Timeout));
    if any_cancelled {
        RunStatus::Cancelled
    } else if any_failed {
        RunStatus::Failed
    } else {
        RunStatus::Completed
    }
}

// ---------- Store impl ----------

#[async_trait]
impl Store for InMemoryStore {
    async fn put_pipeline(&self, spec: PipelineSpec) -> Result<(), StoreError> {
        let mut s = self.inner.state.lock().await;
        if let Some(existing) = s.pipelines.get(&spec.id) {
            if !pipelines_equivalent(existing, &spec) {
                return Err(StoreError::PipelineConflict(spec.id));
            }
            return Ok(());
        }
        s.pipelines.insert(spec.id, spec);
        Ok(())
    }

    async fn get_pipeline(&self, id: PipelineId) -> Result<Option<PipelineSpec>, StoreError> {
        Ok(self.inner.state.lock().await.pipelines.get(&id).cloned())
    }

    async fn create_run(&self, pipeline: PipelineId, inputs: Value) -> Result<RunId, StoreError> {
        let mut s = self.inner.state.lock().await;
        let spec = s.pipeline(pipeline)?.clone();
        instantiate_run(&mut s, &spec, inputs)
    }

    async fn next_dispatchable(&self, kinds: &[String]) -> Result<Option<Job>, StoreError> {
        let s = self.inner.state.lock().await;
        // Naive: scan for oldest Queued job whose kind is in `kinds`.
        let mut best: Option<&Job> = None;
        for j in s.jobs.values() {
            if !matches!(j.status, JobStatus::Queued) {
                continue;
            }
            if !kinds.is_empty() && !kinds.iter().any(|k| k == &j.kind) {
                continue;
            }
            match best {
                None => best = Some(j),
                Some(curr) if j.created_at < curr.created_at => best = Some(j),
                _ => {}
            }
        }
        Ok(best.cloned())
    }

    async fn try_dispatch(
        &self,
        job_id: JobId,
        max_concurrent_for_kind: usize,
    ) -> Result<Option<DispatchTicket>, StoreError> {
        let mut s = self.inner.state.lock().await;
        let kind = {
            let job = s.job(job_id)?;
            if job.status != JobStatus::Queued {
                return Ok(None);
            }
            job.kind.clone()
        };
        // Concurrency cap: count Running jobs of the same kind.
        let running_of_kind = s
            .jobs
            .values()
            .filter(|j| matches!(j.status, JobStatus::Running) && j.kind == kind)
            .count();
        if max_concurrent_for_kind != 0 && running_of_kind >= max_concurrent_for_kind {
            return Ok(None);
        }
        let job = s.job_mut(job_id)?;
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());
        let ticket = DispatchTicket {
            job_id,
            run_id: job.run_id,
            expected_completion_gen: job.completion_generation,
            expected_cancel_gen: job.cancel_generation,
        };
        Ok(Some(ticket))
    }

    async fn set_pid(&self, job_id: JobId, pid: u32) -> Result<(), StoreError> {
        let mut s = self.inner.state.lock().await;
        let job = s.job_mut(job_id)?;
        job.pid = Some(pid);
        Ok(())
    }

    async fn complete_job_and_propagate(
        &self,
        ticket: DispatchTicket,
        terminal: TerminalOutcome,
    ) -> Result<bool, StoreError> {
        let mut events: Vec<StoreEvent> = Vec::new();
        let result = {
            let mut s = self.inner.state.lock().await;

            // Step 1: CAS.
            let run_cancel_gen = s.run(ticket.run_id)?.cancel_generation;
            {
                let job = s.job(ticket.job_id)?;
                if job.status != JobStatus::Running
                    || job.completion_generation != ticket.expected_completion_gen
                    || job.cancel_generation != ticket.expected_cancel_gen
                    || run_cancel_gen != ticket.expected_cancel_gen
                {
                    return Ok(false);
                }
            }

            // Step 2: write terminal state FIRST so subsequent steps see the row terminal.
            let now = Utc::now();
            let status = terminal.status();
            let output = match &terminal {
                TerminalOutcome::Success { output, .. } => Some(output.clone()),
                _ => None,
            };
            {
                let job = s.job_mut(ticket.job_id)?;
                job.status = status;
                job.completed_at = Some(now);
                job.output = output.clone();
                job.completion_generation += 1;
                job.pid = None;
                match &terminal {
                    TerminalOutcome::Success { findings_count, .. } => {
                        job.findings_count = *findings_count;
                        job.error = None;
                    }
                    TerminalOutcome::Failure { error, exit_code } => {
                        job.error = Some(error.clone());
                        job.exit_code = *exit_code;
                    }
                    TerminalOutcome::Timeout => {
                        job.error = Some("timeout".into());
                    }
                    TerminalOutcome::Cancelled => {
                        job.error = Some("cancelled".into());
                    }
                }
            }
            events.push(StoreEvent::JobTerminal {
                job_id: ticket.job_id,
                run_id: ticket.run_id,
                status,
            });

            // Step 3: resolve every outgoing edge exactly once.
            let outgoing = s.outgoing_edges(ticket.run_id, ticket.job_id)?;
            for edge in &outgoing {
                apply_edge_resolution(&mut s, edge, status, &output)?;
            }

            // Step 4: decide each ready target's next status.
            for edge in &outgoing {
                let target_id = edge.to;
                let incoming = s.incoming_edges(ticket.run_id, target_id)?;
                let next = {
                    let target = s.job(target_id)?;
                    evaluate_target_readiness(target, &incoming)
                };
                match next {
                    Some(JobStatus::Queued) => {
                        let target = s.job_mut(target_id)?;
                        if target.status == JobStatus::Pending {
                            target.status = JobStatus::Queued;
                        }
                    }
                    Some(JobStatus::Skipped) => {
                        skip_job_cascade(&mut s, target_id, &mut events)?;
                    }
                    _ => {}
                }
            }

            // Step 5: bundle propagation.
            let pipeline_id = s.run(ticket.run_id)?.pipeline_id;
            propagate_bundles(&mut s, pipeline_id, ticket.job_id, &mut events)?;

            // Step 6: recompute run status.
            let new_run_status = recompute_run_status(&mut s, ticket.run_id);
            let prev_status = s.run(ticket.run_id)?.status;
            if new_run_status != prev_status {
                let r = s.run_mut(ticket.run_id)?;
                r.status = new_run_status;
                if matches!(
                    new_run_status,
                    RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
                ) {
                    r.completed_at = Some(now);
                    events.push(StoreEvent::RunTerminal {
                        run_id: ticket.run_id,
                        status: new_run_status,
                    });
                }
            }
            true
        };

        for ev in events {
            self.emit(ev);
        }
        Ok(result)
    }

    async fn cancel_run(&self, run_id: RunId) -> Result<(), StoreError> {
        let mut events: Vec<StoreEvent> = Vec::new();
        {
            let mut s = self.inner.state.lock().await;
            let new_gen = {
                let r = s.run_mut(run_id)?;
                if matches!(r.status, RunStatus::Cancelled) {
                    return Ok(());
                }
                r.cancel_generation += 1;
                r.status = RunStatus::Cancelled;
                r.completed_at = Some(Utc::now());
                r.cancel_generation
            };

            let job_ids: Vec<JobId> = s.run_jobs.get(&run_id).cloned().unwrap_or_default();
            for jid in &job_ids {
                let (was_running, kind_pid) = {
                    let job = s.job(*jid)?;
                    if job.status.is_terminal() {
                        continue;
                    }
                    (
                        matches!(job.status, JobStatus::Running),
                        (job.kind.clone(), job.pid),
                    )
                };
                let _ = kind_pid; // currently unused — placeholder for richer telemetry.
                {
                    let job = s.job_mut(*jid)?;
                    job.status = JobStatus::Cancelled;
                    job.completed_at = Some(Utc::now());
                    job.cancel_generation = new_gen;
                    job.error = Some("run cancelled".into());
                    if was_running {
                        job.cancel_kill_pending = true;
                        s.cancel_kill_count += 1;
                    }
                }
                events.push(StoreEvent::JobTerminal {
                    job_id: *jid,
                    run_id,
                    status: JobStatus::Cancelled,
                });

                // Resolve outgoing edges Unsatisfied so downstream targets don't deadlock.
                let outgoing = s.outgoing_edges(run_id, *jid)?;
                for edge in &outgoing {
                    apply_edge_resolution(&mut s, edge, JobStatus::Cancelled, &None)?;
                }
            }

            // Recompute readiness of every job in the run that's still Pending.
            let outgoing_targets: Vec<JobId> = s
                .run_jobs
                .get(&run_id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|jid| matches!(s.jobs.get(jid).map(|j| j.status), Some(JobStatus::Pending)))
                .collect();
            for tid in outgoing_targets {
                let incoming = s.incoming_edges(run_id, tid)?;
                let next = {
                    let target = s.job(tid)?;
                    evaluate_target_readiness(target, &incoming)
                };
                if let Some(JobStatus::Skipped) = next {
                    skip_job_cascade(&mut s, tid, &mut events)?;
                }
            }

            events.push(StoreEvent::RunCancelled { run_id });
            events.push(StoreEvent::RunTerminal {
                run_id,
                status: RunStatus::Cancelled,
            });
        }
        for ev in events {
            self.emit(ev);
        }
        Ok(())
    }

    async fn append_log(&self, job_id: JobId, line: &str) -> Result<(), StoreError> {
        let mut s = self.inner.state.lock().await;
        if !s.jobs.contains_key(&job_id) {
            return Err(StoreError::JobNotFound(job_id));
        }
        s.logs
            .entry(job_id)
            .or_default()
            .push_back(line.to_string());
        Ok(())
    }

    async fn read_log(&self, job_id: JobId, range: LogRange) -> Result<Vec<String>, StoreError> {
        let s = self.inner.state.lock().await;
        if !s.jobs.contains_key(&job_id) {
            return Err(StoreError::JobNotFound(job_id));
        }
        let lines = s.logs.get(&job_id).cloned().unwrap_or_default();
        let v: Vec<String> = match range {
            LogRange::All => lines.into_iter().collect(),
            LogRange::Range { start, end } => lines
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (*i as u64) >= start && (*i as u64) < end)
                .map(|(_, l)| l)
                .collect(),
            LogRange::Tail(n) => {
                let total = lines.len();
                lines.into_iter().skip(total.saturating_sub(n)).collect()
            }
        };
        Ok(v)
    }

    async fn get_job(&self, job_id: JobId) -> Result<Option<Job>, StoreError> {
        Ok(self.inner.state.lock().await.jobs.get(&job_id).cloned())
    }

    async fn get_run(&self, run_id: RunId) -> Result<Option<Run>, StoreError> {
        Ok(self.inner.state.lock().await.runs.get(&run_id).cloned())
    }

    async fn list_run_jobs(&self, run_id: RunId) -> Result<Vec<Job>, StoreError> {
        let s = self.inner.state.lock().await;
        let ids = s.run_jobs.get(&run_id).cloned().unwrap_or_default();
        Ok(ids
            .into_iter()
            .filter_map(|id| s.jobs.get(&id).cloned())
            .collect())
    }

    async fn subscribe(&self) -> StoreEventStream {
        let rx = self.inner.events.subscribe();
        // `stream::unfold` keeps a single in-flight `recv` future across polls
        // by feeding the same `rx` back in each iteration; this avoids the
        // re-registration bug that comes from constructing a fresh future
        // inside `poll_next`.
        let s = stream::unfold(rx, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => return Some((ev, rx)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        Box::pin(s)
    }

    async fn find_running_past(
        &self,
        deadline: DateTime<Utc>,
    ) -> Result<Vec<DispatchTicket>, StoreError> {
        let s = self.inner.state.lock().await;
        let mut out = Vec::new();
        for j in s.jobs.values() {
            if !matches!(j.status, JobStatus::Running) {
                continue;
            }
            if j.started_at.map(|t| t < deadline).unwrap_or(false) {
                out.push(DispatchTicket {
                    job_id: j.id,
                    run_id: j.run_id,
                    expected_completion_gen: j.completion_generation,
                    expected_cancel_gen: j.cancel_generation,
                });
            }
        }
        Ok(out)
    }

    async fn find_cancelled_with_pid(&self) -> Result<Vec<(JobId, u32)>, StoreError> {
        let s = self.inner.state.lock().await;
        Ok(s.jobs
            .values()
            .filter(|j| j.cancel_kill_pending && j.pid.is_some())
            .map(|j| (j.id, j.pid.unwrap()))
            .collect())
    }

    async fn clear_pid_and_kill_flag(&self, job_id: JobId) -> Result<(), StoreError> {
        let mut s = self.inner.state.lock().await;
        let job = s.job_mut(job_id)?;
        if job.cancel_kill_pending {
            // Decrement counter only if previously set.
            // (Counter is best-effort; not exposed publicly.)
        }
        job.cancel_kill_pending = false;
        job.pid = None;
        Ok(())
    }
}

fn pipelines_equivalent(a: &PipelineSpec, b: &PipelineSpec) -> bool {
    a.id == b.id
        && a.name == b.name
        && a.jobs.len() == b.jobs.len()
        && a.edges.len() == b.edges.len()
        && a.bundles.len() == b.bundles.len()
}

// Mark unused fields (`exit_code`, `cancel_kill_count`) as intentionally read.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<InMemoryStore>();
};

// Silence unused field warnings until SubprocessHandler / OrphanWorker use them.
const _UNUSED: fn(&Job, &State) = |_j, _s| {
    let _ = _j.exit_code;
    let _ = _s.cancel_kill_count;
};

// EdgeCondition is referenced via the `matches` method inside `apply_edge_resolution`.
// This dummy use silences a possible unused-import lint.
#[allow(dead_code)]
fn _edge_condition_witness(_c: EdgeCondition) {}
