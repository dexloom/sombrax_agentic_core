//! Pluggable pipeline / bundle / job runtime.
//!
//! ## Layout
//!
//! - [`model`] — plain-data domain types (Job, Bundle, Pipeline, Edge, statuses, ids).
//! - [`error`] — `StoreError`, `PipelineValidationError`.
//! - [`store`] — `Store` trait + `InMemoryStore` impl + reusable conformance suite.
//! - [`pipeline`] — `PipelineBuilder` with build-time validation.
//! - [`registry`] — `HandlerRegistry` (kind → `Arc<dyn JobHandler>`).
//! - [`handler`] — `JobHandler` trait, `JobContext`, `JobOutput`, `JobError`.
//! - [`log`] — `LogWriter` bridging `Store::append_log` and `tracing`.
//! - [`worker`] — dispatcher / orphan / cancel-sweep loops.
//! - [`runtime`] — public façade.

pub mod error;
pub mod handler;
pub mod log;
pub mod model;
pub mod pipeline;
pub mod registry;
pub mod runtime;
pub mod store;
pub mod subprocess;
pub mod worker;

pub use error::{PipelineValidationError, StoreError};
pub use handler::{JobContext, JobError, JobHandler, JobOutput};
pub use log::LogWriter;
pub use model::{
    Bundle, BundleId, BundleStatus, BundleTemplate, DispatchTicket, EdgeCondition, EdgeId,
    EdgeResolution, EdgeTemplate, Job, JobId, JobStatus, JobTemplate, JoinPolicy, LogRange,
    MergePolicy, OutputProjection, PipelineId, PipelineSpec, Run, RunCondition, RunId, RunOutcome,
    RunStatus, TerminalOutcome,
};
pub use pipeline::PipelineBuilder;
pub use registry::{HandlerRegistry, HandlerRegistryBuilder, RegistryError};
pub use runtime::{Runtime, RuntimeConfig};
pub use store::{InMemoryStore, Store, StoreEvent};
pub use subprocess::{ArgTpl, EnvTpl, OutputParser, SubprocessHandler};
