//! Domain types for the runs runtime.
//!
//! Mirrors §2 of the design plan. Every type here is plain data — the lifecycle
//! semantics live behind the [`super::store::Store`] trait.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use uuid::Uuid;

macro_rules! define_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name(pub Uuid);

        impl $name {
            /// Generate a new random v4 id.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_id!(JobId, "Stable id for a single Job within a Run.");
define_id!(BundleId, "Stable id for a Bundle within a Run.");
define_id!(PipelineId, "Stable id for a registered PipelineSpec.");
define_id!(RunId, "Stable id for one execution of a Pipeline.");
define_id!(EdgeId, "Stable id for a single Edge instance within a Run.");

/// Job lifecycle states.
///
/// `Skipped` is assigned exclusively by [`super::store::Store::complete_job_and_propagate`]
/// step 4 — there is no separate Skipped pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Awaiting incoming edges to resolve.
    Pending,
    /// Inputs ready, waiting for the dispatcher.
    Queued,
    /// Handler is executing.
    Running,
    /// Terminal — handler returned success.
    Completed,
    /// Terminal — handler returned an error.
    Failed,
    /// Terminal — handler exceeded its timeout.
    Timeout,
    /// Terminal — `cancel_run` cascaded here.
    Cancelled,
    /// Terminal — every required incoming producer resolved Unsatisfied.
    Skipped,
}

impl JobStatus {
    /// True for any terminal state (no further transitions allowed).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Timeout | Self::Cancelled | Self::Skipped
        )
    }

    /// True for the success terminal.
    pub fn is_success(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Bundle lifecycle states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleStatus {
    /// Awaiting parent bundle / sibling jobs.
    Pending,
    /// Run condition could fire; all member jobs queued.
    Queued,
    /// At least one member job in flight.
    Running,
    /// Run condition Satisfied.
    Completed,
    /// Run condition Violated — successors will be marked Skipped.
    Blocked,
    /// `cancel_run` cascaded here.
    Cancelled,
}

/// Run lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// No jobs reached terminal yet (or no jobs in run).
    Pending,
    /// At least one terminal but not all.
    Running,
    /// Every job terminal and none Failed/Timeout/Cancelled (Skipped allowed).
    Completed,
    /// Every job terminal and at least one Failed/Timeout (no Cancelled).
    Failed,
    /// Run was cancelled.
    Cancelled,
}

/// How a target picks the source value out of an upstream output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputProjection {
    /// Pass the entire upstream output object through.
    Whole,
    /// Dotted-path field selection (e.g. `"a.b.c"`). Fast; first choice.
    Field(String),
    /// RFC 9535 JSON path. Escape hatch — slower; only when `Field` is too weak.
    JsonPath(String),
}

/// What to do when ≥2 edges write the same `(target_job, target_field)`.
///
/// Validated at pipeline-build time: edges that share a target field must agree
/// on the policy, and `Reject` makes a same-target collision a build error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergePolicy {
    /// Pipeline build fails if two edges share the target.
    Reject,
    /// The latest edge applied (by completion order) wins.
    LastWriteWins,
    /// Wrap target in array; deterministic order = `(edge.id)` lexicographic.
    AppendArray,
    /// Shallow object merge if both target and producer write objects.
    ObjectMerge,
}

/// When does an edge apply (i.e. resolve as `Applied` rather than `Unsatisfied`)?
///
/// `OnFailure` deliberately includes `Skipped`, since downstream cleanup that
/// reacts to "a producer didn't fire" should react the same way to skip and to
/// hard failure. Use `OnTerminals` for fine-grained control.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EdgeCondition {
    /// Source must be `Completed`.
    OnSuccess,
    /// Source must be one of `Failed | Timeout | Cancelled | Skipped`.
    OnFailure,
    /// Explicit set of source terminals.
    OnTerminals(SmallVec<[JobStatus; 4]>),
    /// Any terminal.
    Always,
}

impl EdgeCondition {
    /// Does this edge fire (resolve `Applied`) given the source's terminal status?
    pub fn matches(&self, source: JobStatus) -> bool {
        debug_assert!(source.is_terminal());
        match self {
            Self::OnSuccess => matches!(source, JobStatus::Completed),
            Self::OnFailure => matches!(
                source,
                JobStatus::Failed | JobStatus::Timeout | JobStatus::Cancelled | JobStatus::Skipped
            ),
            Self::OnTerminals(set) => set.contains(&source),
            Self::Always => true,
        }
    }
}

/// Per-job fan-in policy. Default = `AllRequired`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinPolicy {
    /// Every required incoming edge must resolve `Applied`. Default.
    #[default]
    AllRequired,
    /// Queue as soon as ≥1 incoming edge resolves `Applied` AND `pending_inputs == 0`.
    AnyApplied,
}

/// Run-condition for a Bundle. Tri-state evaluation lives in the Store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunCondition {
    /// Every job in the bundle must reach `Completed`. Default.
    #[default]
    AllSuccess,
    /// At least one job must reach `Completed`.
    AnySuccess,
    /// Every job terminal — outcome irrelevant.
    AllComplete,
    /// Same as `AllComplete`. Reserved for future divergence.
    Always,
}

/// State of one edge instance within a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeResolution {
    /// Source not terminal yet.
    Pending,
    /// Source terminal AND condition matched: target field populated.
    Applied,
    /// Source terminal AND condition did not match: edge contributes nothing.
    Unsatisfied,
}

/// Outcome the dispatcher passes to `complete_job_and_propagate`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminalOutcome {
    /// Handler returned success with this output value.
    Success {
        /// Output value the handler produced.
        output: serde_json::Value,
        /// Optional findings count for telemetry.
        findings_count: Option<usize>,
    },
    /// Handler returned an error.
    Failure {
        /// Human-readable error message.
        error: String,
        /// Optional process exit code (for `SubprocessHandler`).
        exit_code: Option<i32>,
    },
    /// Handler exceeded its timeout.
    Timeout,
    /// `cancel_run` already terminalised this job; the dispatcher's call
    /// is a no-op (CAS will fail). Included for completeness so the dispatcher
    /// can forward the reason it observed.
    Cancelled,
}

impl TerminalOutcome {
    /// What `JobStatus` this outcome maps to.
    pub fn status(&self) -> JobStatus {
        match self {
            Self::Success { .. } => JobStatus::Completed,
            Self::Failure { .. } => JobStatus::Failed,
            Self::Timeout => JobStatus::Timeout,
            Self::Cancelled => JobStatus::Cancelled,
        }
    }
}

/// CAS witness handed back from `try_dispatch` to `complete_job_and_propagate`.
///
/// Carries the completion-generation and cancel-generation observed at dispatch
/// time. Both are checked atomically inside `complete_job_and_propagate` so that
/// a `cancel_run` (which bumps cancel_generation on the run AND every non-terminal
/// job in the same tx) invalidates an in-flight completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchTicket {
    /// The job being dispatched.
    pub job_id: JobId,
    /// The run this job belongs to.
    pub run_id: RunId,
    /// Job's completion_generation at dispatch time.
    pub expected_completion_gen: u64,
    /// Run/job's cancel_generation at dispatch time.
    pub expected_cancel_gen: u64,
}

/// One job instance within a [`Run`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    /// Stable id (within run scope).
    pub id: JobId,
    /// Run this job belongs to.
    pub run_id: RunId,
    /// Pipeline that produced this job instance.
    pub pipeline_id: PipelineId,
    /// Optional bundle membership (for run-condition aggregation).
    pub bundle_id: Option<BundleId>,
    /// Handler kind (matches `JobHandler::kind`).
    pub kind: String,
    /// Effective inputs after edges resolve. Mutated by the Store.
    pub inputs: serde_json::Value,
    /// Handler output, if terminal == Completed.
    pub output: Option<serde_json::Value>,
    /// Current lifecycle state.
    pub status: JobStatus,
    /// Human-readable error if terminal ≠ Completed.
    pub error: Option<String>,
    /// Number of incoming edges still `Pending`.
    pub pending_inputs: u32,
    /// Per-incoming-edge resolution map. Created `Pending` for every edge at
    /// `create_run` time, mutated atomically by `complete_job_and_propagate`.
    pub edge_resolutions: BTreeMap<EdgeId, EdgeResolution>,
    /// Idempotency CAS. Bumped by every successful completion.
    pub completion_generation: u64,
    /// Cancel-fence CAS. Updated by `cancel_run` to invalidate stale completions.
    pub cancel_generation: u64,
    /// Fan-in policy from the template.
    pub join_policy: JoinPolicy,
    /// `cancel_run` set this; dispatcher's cancel sweep should kill the pid.
    pub cancel_kill_pending: bool,
    /// Process id, when the handler is a subprocess.
    pub pid: Option<u32>,
    /// Process exit code, when known.
    pub exit_code: Option<i32>,
    /// Findings count surfaced by the handler.
    pub findings_count: Option<usize>,
    /// When the row was first created.
    pub created_at: DateTime<Utc>,
    /// When `try_dispatch` last transitioned this to Running.
    pub started_at: Option<DateTime<Utc>>,
    /// When the row reached a terminal state.
    pub completed_at: Option<DateTime<Utc>>,
}

/// One bundle instance within a [`Run`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bundle {
    /// Stable id.
    pub id: BundleId,
    /// Run this bundle belongs to.
    pub run_id: RunId,
    /// Pipeline that produced this bundle instance.
    pub pipeline_id: PipelineId,
    /// Optional parent bundle (for nested bundles).
    pub parent: Option<BundleId>,
    /// Member jobs whose status feeds the run-condition.
    pub job_ids: Vec<JobId>,
    /// Successor bundles fanned out when this bundle reaches Satisfied.
    pub successor_ids: Vec<BundleId>,
    /// Run condition for this bundle.
    pub run_condition: RunCondition,
    /// Current bundle status.
    pub status: BundleStatus,
    /// Optional reason recorded when the bundle is Blocked or Cancelled.
    pub blocked_reason: Option<String>,
}

/// One run instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Run {
    /// Stable id.
    pub id: RunId,
    /// Pipeline this run was started from.
    pub pipeline_id: PipelineId,
    /// Current run status.
    pub status: RunStatus,
    /// Inputs supplied at `create_run`.
    pub inputs: serde_json::Value,
    /// Cancel-fence generation. Bumped by `cancel_run`.
    pub cancel_generation: u64,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// When the run reached a terminal state.
    pub completed_at: Option<DateTime<Utc>>,
}

/// Job template inside a [`PipelineSpec`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobTemplate {
    /// Stable id (re-used as the runtime [`JobId`] for single-instantiation runs).
    pub id: JobId,
    /// Handler kind.
    pub kind: String,
    /// Default inputs merged with the run's submitted inputs at `create_run`.
    #[serde(default)]
    pub default_inputs: serde_json::Value,
    /// Optional bundle this job belongs to.
    pub bundle_id: Option<BundleId>,
    /// Fan-in policy for this template.
    #[serde(default)]
    pub join_policy: JoinPolicy,
}

/// Edge template inside a [`PipelineSpec`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeTemplate {
    /// Stable edge id (re-used per run for `runs_edge_resolutions`).
    pub id: EdgeId,
    /// Source job template.
    pub from: JobId,
    /// Target job template.
    pub to: JobId,
    /// How to project the source's output.
    pub source: OutputProjection,
    /// Field on the target's `inputs` to write.
    pub target: String,
    /// When the edge fires.
    pub condition: EdgeCondition,
    /// How to merge if multiple edges share `(to, target)`.
    pub merge: MergePolicy,
    /// Whether this edge is required for `JoinPolicy::AllRequired`. Default true.
    #[serde(default = "default_required")]
    pub required: bool,
}

fn default_required() -> bool {
    true
}

/// Bundle template inside a [`PipelineSpec`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleTemplate {
    /// Stable bundle id.
    pub id: BundleId,
    /// Optional parent bundle.
    pub parent: Option<BundleId>,
    /// Member job ids.
    pub job_ids: Vec<JobId>,
    /// Successor bundle ids.
    pub successor_ids: Vec<BundleId>,
    /// Run condition.
    #[serde(default)]
    pub run_condition: RunCondition,
}

/// Validated, immutable pipeline blueprint. Build via [`super::pipeline::PipelineBuilder`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Stable id.
    pub id: PipelineId,
    /// Human-readable name.
    pub name: String,
    /// Job templates.
    pub jobs: Vec<JobTemplate>,
    /// Edge templates.
    pub edges: Vec<EdgeTemplate>,
    /// Bundle templates.
    pub bundles: Vec<BundleTemplate>,
}

/// Range selector for log reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LogRange {
    /// Every line.
    #[default]
    All,
    /// `[start, end)` by sequence number.
    Range {
        /// Inclusive start.
        start: u64,
        /// Exclusive end.
        end: u64,
    },
    /// Last `n` lines.
    Tail(usize),
}

/// Final state of a run, surfaced via [`super::runtime::Runtime::wait`].
#[derive(Clone, Debug)]
pub struct RunOutcome {
    /// The terminal run status.
    pub status: RunStatus,
    /// Snapshot of every job at termination.
    pub jobs: Vec<Job>,
}
