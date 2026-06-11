//! Error types for the runs runtime.

use thiserror::Error;

use super::model::{BundleId, EdgeId, JobId, PipelineId, RunId};

/// Top-level error from the [`super::store::Store`] trait.
#[derive(Debug, Error)]
pub enum StoreError {
    /// Pipeline id already exists with a different spec.
    #[error("pipeline already exists with a different spec: {0}")]
    PipelineConflict(PipelineId),
    /// Pipeline id not found.
    #[error("pipeline not found: {0}")]
    PipelineNotFound(PipelineId),
    /// Run id not found.
    #[error("run not found: {0}")]
    RunNotFound(RunId),
    /// Job id not found.
    #[error("job not found: {0}")]
    JobNotFound(JobId),
    /// Bundle id not found.
    #[error("bundle not found: {0}")]
    BundleNotFound(BundleId),
    /// Edge id not found.
    #[error("edge not found: {0}")]
    EdgeNotFound(EdgeId),
    /// Pipeline spec failed validation.
    #[error("pipeline spec invalid: {0}")]
    PipelineInvalid(String),
    /// Surreal/IO error etc.
    #[error("backend error: {0}")]
    Backend(String),
    /// Invariant the implementation expected to hold did not.
    #[error("invariant violated: {0}")]
    Invariant(String),
}

/// Pipeline build-time validation errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PipelineValidationError {
    /// Pipeline contains a cycle. Lists one cycle for the user.
    #[error("pipeline has a cycle through jobs: {0:?}")]
    Cycle(Vec<JobId>),
    /// Edge references a job that is not in the spec.
    #[error("edge {edge_id} references unknown job {job_id}")]
    UnknownJob {
        /// Offending edge.
        edge_id: EdgeId,
        /// Job that is missing.
        job_id: JobId,
    },
    /// Two edges write the same `(to, target)` and at least one declares
    /// `MergePolicy::Reject`, or they disagree on the policy.
    #[error("conflicting edges into job {to} field {target:?}: {detail}")]
    EdgeConflict {
        /// Target job.
        to: JobId,
        /// Target field name.
        target: String,
        /// Human-readable detail.
        detail: String,
    },
    /// A bundle references a job that is not in the spec.
    #[error("bundle {bundle_id} references unknown job {job_id}")]
    BundleUnknownJob {
        /// Offending bundle.
        bundle_id: BundleId,
        /// Job that is missing.
        job_id: JobId,
    },
    /// Bundle's parent does not exist.
    #[error("bundle {bundle_id} declares unknown parent {parent}")]
    BundleUnknownParent {
        /// Offending bundle.
        bundle_id: BundleId,
        /// Parent bundle that is missing.
        parent: BundleId,
    },
    /// Two job templates share an id.
    #[error("duplicate job id {0}")]
    DuplicateJob(JobId),
    /// Two edge templates share an id.
    #[error("duplicate edge id {0}")]
    DuplicateEdge(EdgeId),
    /// Two bundle templates share an id.
    #[error("duplicate bundle id {0}")]
    DuplicateBundle(BundleId),
    /// A job kind is not registered with the [`super::registry::HandlerRegistry`].
    #[error("job {job_id} kind {kind:?} is not registered with the handler registry")]
    UnknownKind {
        /// Job whose kind is missing.
        job_id: JobId,
        /// The kind string.
        kind: String,
    },
}
