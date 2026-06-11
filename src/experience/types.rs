//! Core types and traits for the experience system.

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// A lesson extracted from a past task execution.
///
/// Lessons are the atomic unit of knowledge. Each lesson captures a specific
/// insight from a task outcome — what went wrong, what worked, or a general
/// pattern to follow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    /// Unique identifier (auto-generated if not provided).
    #[serde(default = "default_id")]
    pub id: String,

    /// Human-readable title.
    pub title: String,

    /// The lesson content — actionable guidance for future agents.
    pub content: String,

    /// Domain this lesson applies to (e.g., "fork_test", "audit", "onchain").
    pub domain: String,

    /// Category for organizing and retrieving lessons.
    pub category: LessonCategory,

    /// Tags for similarity matching during retrieval.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Contract address this lesson is specific to (None = general pattern).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_address: Option<String>,

    /// Source trace/job that produced this lesson.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Task outcome that triggered this lesson (e.g., "INCONCLUSIVE", "DISPROVED").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,

    /// When this lesson was created (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// Source-pattern triggers for **pre-audit** injection.
    ///
    /// The default keyword-matcher fires mid-run on agent tool output — good
    /// for refining approach as the agent encounters a pattern. But some
    /// knowledge (framework semantics — Seaport tipping, Safe fallback,
    /// LayerZero composer) needs to reach the agent BEFORE analysis starts,
    /// keyed off what the target contract imports / inherits / calls.
    ///
    /// When any field matches the target source at audit-start, this lesson
    /// is pre-injected alongside any mid-run matches. Absent = mid-run match
    /// only (legacy behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triggers: Option<LessonTriggers>,
}

/// Source-pattern triggers used by `pre_audit_inject` to decide which
/// lessons are relevant for a given target contract before analysis runs.
/// All fields are OR'd — any match pre-injects the lesson.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LessonTriggers {
    /// Substrings to match against `import ...` statements. E.g.,
    /// `["seaport-sol", "@safe-global/"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<String>,

    /// Base-contract names to match against `contract X is <Name>` clauses.
    /// E.g., `["ZoneInterface", "IERC4626"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inherits: Vec<String>,

    /// Function-name substrings to match against method calls in the source.
    /// E.g., `["fulfillAdvancedOrder", "lzCompose", "latestRoundData"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
}

impl LessonTriggers {
    /// Returns true when at least one trigger field is populated.
    pub fn is_empty(&self) -> bool {
        self.imports.is_empty() && self.inherits.is_empty() && self.calls.is_empty()
    }

    /// Test all triggers against a source blob. Performs case-insensitive
    /// substring match for `imports` / `calls` and anchored word-boundary
    /// match for `inherits` (to avoid `IERC4626` matching `IERC4626Upgradeable`
    /// only when intended — still substring by default for simplicity).
    pub fn matches_source(&self, source: &str) -> bool {
        if self.is_empty() {
            return false;
        }
        let lower = source.to_ascii_lowercase();
        for imp in &self.imports {
            if lower.contains(&imp.to_ascii_lowercase()) {
                return true;
            }
        }
        for inh in &self.inherits {
            if lower.contains(&inh.to_ascii_lowercase()) {
                return true;
            }
        }
        for call in &self.calls {
            if lower.contains(&call.to_ascii_lowercase()) {
                return true;
            }
        }
        false
    }
}

fn default_id() -> String {
    uuid::Uuid::new_v4().to_string()[..8].to_string()
}

impl Default for Lesson {
    fn default() -> Self {
        Self {
            id: default_id(),
            title: String::new(),
            content: String::new(),
            domain: String::new(),
            category: LessonCategory::Pattern,
            tags: vec![],
            contract_address: None,
            source: None,
            outcome: None,
            created_at: None,
            triggers: None,
        }
    }
}

/// Category for organizing lessons in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LessonCategory {
    /// General reusable pattern (stored in patterns/).
    Pattern,
    /// Contract-specific lesson (stored in contracts/).
    Contract,
    /// Successful approach worth repeating (stored in successes/).
    Success,
}

impl LessonCategory {
    /// Filesystem subdirectory for this category.
    pub fn dir_name(&self) -> &str {
        match self {
            LessonCategory::Pattern => "patterns",
            LessonCategory::Contract => "contracts",
            LessonCategory::Success => "successes",
        }
    }
}

/// Query for retrieving relevant lessons from the knowledge base.
#[derive(Debug, Clone, Default)]
pub struct LessonQuery {
    /// Domain to search in (e.g., "fork_test").
    pub domain: String,

    /// Tags to match (OR-logic: any matching tag counts).
    pub tags: Vec<String>,

    /// Contract address for contract-specific lessons.
    pub contract_address: Option<String>,

    /// Maximum number of lessons to return.
    pub limit: usize,
}

impl LessonQuery {
    /// Create a new query for a domain with default limit.
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            limit: 10,
            ..Default::default()
        }
    }

    /// Filter by tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Filter by contract address.
    pub fn with_contract(mut self, address: impl Into<String>) -> Self {
        self.contract_address = Some(address.into());
        self
    }

    /// Set maximum results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Trait for analyzing task execution traces and extracting lessons.
///
/// Implementors provide domain-specific logic for understanding what
/// happened during a task and producing actionable lessons.
///
/// The generic `Trace` type allows different domains to define their
/// own trace format (e.g., fork test logs, audit reports, etc.).
pub trait Reflector: Send + Sync {
    /// The trace type this reflector can analyze.
    type Trace: Send + Sync;

    /// Analyze a task execution trace and extract lessons.
    ///
    /// Returns zero or more lessons. An empty vec means nothing
    /// noteworthy was found in the trace.
    fn reflect<'a>(
        &'a self,
        trace: &'a Self::Trace,
    ) -> Pin<Box<dyn Future<Output = Vec<Lesson>> + Send + 'a>>;
}
