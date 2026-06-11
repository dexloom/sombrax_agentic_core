//! File-History Context Management - Message Classification
//!
//! This module implements classification of conversation messages for file-based
//! context management, enabling de-duplication of file reads/edits and ensuring
//! the latest file content is maintained at the end of the dialog.
//!
//! # Architecture Overview
//!
//! The classification system operates on conversation messages to:
//! 1. Extract metadata signals from tool calls and results
//! 2. Track file artifacts and their versions
//! 3. Apply supersedence rules to determine which messages to keep/drop
//! 4. Reorder context to place latest file snapshots at the end
//!
//! # Classification Signals
//!
//! Each message is classified based on:
//! - **Message Role**: `user` (tool results) or `assistant` (tool calls)
//! - **Operation Type**: `read`, `write`, `edit`, or `other`
//! - **File Path**: Normalized absolute path to the file
//! - **Line Range**: Optional `(offset, limit)` for partial reads
//! - **Content Hash**: BLAKE3 hash of file content for version tracking
//! - **Message ID**: Unique identifier for the message
//! - **Timestamp**: Monotonic sequence number for ordering
//!
//! # Important Design Notes
//!
//! ## Classification Keys
//!
//! Classifications are keyed by a composite key of `(message_id, tool_call_id)` to
//! handle multiple tool calls in a single assistant message correctly. Each tool
//! call gets its own classification entry.
//!
//! ## Unclassified Messages
//!
//! The optimization result only tracks file-related operations. Messages without
//! file operations (normal chat, non-file tools) are not classified and should
//! be preserved by default. Use `is_classified()` to check if a message ID has
//! any classifications before applying optimization decisions.

pub mod edge_cases;
pub mod hook;
#[cfg(test)]
mod test_transcripts;

pub use hook::{FileContextExt, FileContextHook, FILE_CONTEXT_KEY};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Classification Metadata Types
// ============================================================================

/// Operation type extracted from tool calls/results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileOperation {
    /// Full or partial file read (read tool)
    Read,
    /// Complete file write/creation (write tool)
    Write,
    /// Patch/edit to existing file (edit tool)
    Edit,
    /// Non-file operation
    Other,
}

impl FileOperation {
    /// Returns true if this operation produces a full file snapshot
    pub fn is_full_snapshot(&self) -> bool {
        matches!(self, FileOperation::Write)
    }

    /// Returns true if this operation reads file content
    pub fn is_read(&self) -> bool {
        matches!(self, FileOperation::Read)
    }

    /// Returns true if this operation modifies file content
    pub fn is_mutating(&self) -> bool {
        matches!(self, FileOperation::Write | FileOperation::Edit)
    }
}

/// Line range for partial file reads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LineRange {
    /// Starting line (0-indexed offset)
    pub offset: usize,
    /// Maximum lines to read (None = unlimited)
    pub limit: Option<usize>,
}

impl LineRange {
    /// Create a new line range
    pub fn new(offset: usize, limit: Option<usize>) -> Self {
        Self { offset, limit }
    }

    /// Check if this is a full file read (offset=0, no limit)
    pub fn is_full_read(&self) -> bool {
        self.offset == 0 && self.limit.is_none()
    }

    /// Check if this range contains another range
    pub fn contains(&self, other: &LineRange) -> bool {
        // This range contains other if:
        // 1. This starts at or before other
        // 2. This ends at or after other (or is unlimited)
        if self.offset > other.offset {
            return false;
        }

        match (self.limit, other.limit) {
            (None, _) => true,        // Unlimited contains everything after offset
            (Some(_), None) => false, // Limited cannot contain unlimited
            (Some(self_limit), Some(other_limit)) => {
                let self_end = self.offset + self_limit;
                let other_end = other.offset + other_limit;
                self_end >= other_end
            }
        }
    }

    /// Check if two ranges overlap
    pub fn overlaps(&self, other: &LineRange) -> bool {
        let self_start = self.offset;
        let other_start = other.offset;

        match (self.limit, other.limit) {
            (None, None) => true, // Both unlimited overlap
            (None, Some(_)) => {
                // self is unlimited, overlaps if other starts >= self start
                // OR if other ends after self starts
                other_start >= self_start || other_start + other.limit.unwrap_or(0) > self_start
            }
            (Some(_), None) => {
                // other is unlimited, overlaps if self starts >= other start
                // OR if self ends after other starts
                self_start >= other_start || self_start + self.limit.unwrap_or(0) > other_start
            }
            (Some(self_limit), Some(other_limit)) => {
                let self_end = self_start + self_limit;
                let other_end = other_start + other_limit;
                self_start < other_end && other_start < self_end
            }
        }
    }
}

/// Content hash for file version tracking (using BLAKE3 for speed)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl ContentHash {
    /// Compute hash from content string
    pub fn from_content(content: &str) -> Self {
        let hash = blake3::hash(content.as_bytes());
        Self(hash.to_hex().to_string())
    }

    /// Create from pre-computed hex string
    pub fn from_hex(hex: impl Into<String>) -> Self {
        Self(hex.into())
    }
}

/// Composite key for classification lookups
///
/// This enables tracking multiple tool calls within a single message.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClassificationKey {
    /// The message ID
    pub message_id: String,
    /// The tool call ID (unique per tool call)
    pub tool_call_id: String,
}

impl ClassificationKey {
    /// Create a new classification key
    pub fn new(message_id: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            message_id: message_id.into(),
            tool_call_id: tool_call_id.into(),
        }
    }
}

impl std::fmt::Display for ClassificationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.message_id, self.tool_call_id)
    }
}

/// Classification metadata extracted from a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageClassification {
    /// Unique message identifier
    pub message_id: String,
    /// Tool call ID (unique per tool call within a message)
    pub tool_call_id: String,
    /// Monotonic sequence number for ordering
    pub sequence: u64,
    /// Message role (user for tool results, assistant for tool calls)
    pub role: MessageRole,
    /// Type of file operation
    pub operation: FileOperation,
    /// Normalized file path (if file-related)
    pub file_path: Option<PathBuf>,
    /// Line range for partial reads
    pub line_range: Option<LineRange>,
    /// Content hash for version tracking
    pub content_hash: Option<ContentHash>,
    /// Size of content in bytes (for prioritization)
    pub content_size: usize,
}

impl MessageClassification {
    /// Get the classification key for this classification
    pub fn key(&self) -> ClassificationKey {
        ClassificationKey::new(&self.message_id, &self.tool_call_id)
    }
}

/// Message role in classification context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// User message (contains tool results)
    User,
    /// Assistant message (contains tool calls)
    Assistant,
}

// ============================================================================
// File Artifact Model
// ============================================================================

/// Represents a tracked file artifact with version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileArtifact {
    /// Normalized absolute path
    pub path: PathBuf,
    /// Current known state of the file
    pub state: FileState,
    /// Operations on this file (ordered by sequence)
    pub operations: Vec<FileOperationRecord>,
    /// Latest full snapshot classification key (if any) - can be from read or write
    pub latest_snapshot_key: Option<ClassificationKey>,
    /// Latest content hash
    pub latest_hash: Option<ContentHash>,
}

/// Current state of a file artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileState {
    /// File has been read but not modified in this session
    Read,
    /// File has been created or overwritten (full snapshot exists)
    Written,
    /// File has been edited (patches applied, may need synthetic snapshot)
    Edited,
    /// File has been deleted
    Deleted,
    /// File was renamed (contains old path reference)
    Renamed,
}

/// Record of an operation on a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperationRecord {
    /// Classification key for this operation
    pub key: ClassificationKey,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Type of operation
    pub operation: FileOperation,
    /// Line range (for reads)
    pub line_range: Option<LineRange>,
    /// Content hash after operation
    pub content_hash: Option<ContentHash>,
    /// Whether this is the tool call or result message
    pub is_result: bool,
}

impl FileArtifact {
    /// Create a new file artifact
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            state: FileState::Read,
            operations: Vec::new(),
            latest_snapshot_key: None,
            latest_hash: None,
        }
    }

    /// Record an operation on this file
    pub fn record_operation(&mut self, record: FileOperationRecord) {
        // Update state based on operation
        match record.operation {
            FileOperation::Read => {
                // Read doesn't change written/edited state
                if self.state == FileState::Deleted {
                    self.state = FileState::Read;
                }
                // Full read results become the latest snapshot
                if record.is_result && record.line_range.map(|r| r.is_full_read()).unwrap_or(true) {
                    self.latest_snapshot_key = Some(record.key.clone());
                }
            }
            FileOperation::Write => {
                self.state = FileState::Written;
                if record.is_result {
                    self.latest_snapshot_key = Some(record.key.clone());
                }
            }
            FileOperation::Edit => {
                // Edit after write keeps written state (write is more complete)
                if self.state != FileState::Written {
                    self.state = FileState::Edited;
                }
            }
            FileOperation::Other => {}
        }

        // Update hash if present
        if record.content_hash.is_some() {
            self.latest_hash = record.content_hash.clone();
        }

        self.operations.push(record);
    }

    /// Get the most recent full read that covers the entire file
    pub fn latest_full_read(&self) -> Option<&FileOperationRecord> {
        self.operations.iter().rev().find(|op| {
            op.operation == FileOperation::Read
                && op.is_result
                && op.line_range.map(|r| r.is_full_read()).unwrap_or(true)
        })
    }

    /// Check if there's a newer operation after the given sequence
    pub fn has_newer_operation(&self, sequence: u64) -> bool {
        self.operations.iter().any(|op| op.sequence > sequence)
    }
}

// ============================================================================
// Supersedence Rules
// ============================================================================

/// Decision on what to do with a message during context optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupersedenceAction {
    /// Keep the message in place
    Keep,
    /// Drop the message entirely (superseded by newer operation)
    Drop,
    /// Move the message to the end of context
    MoveToEnd,
    /// Replace with a synthetic snapshot
    ReplaceWithSnapshot,
}

/// Supersedence rules for file operations
///
/// # Rules
///
/// 1. **Read supersedence**: A newer full read of the same file supersedes
///    older reads (full or partial) of that file.
///
/// 2. **Write supersedence**: A write operation supersedes all prior reads
///    and edits of the same file.
///
/// 3. **Edit accumulation**: Edits don't supersede each other unless a newer
///    write or full read exists. Multiple edits may be kept to show change history.
///
/// 4. **Partial read rules**: A full read supersedes partial reads. Overlapping
///    partial reads: the newer one supersedes if it fully contains the older,
///    or if they overlap and the newer has broader coverage.
///
/// 5. **Snapshot requirement**: The latest full file state must always be available.
///    If only edits exist, patches must be applied to create a synthetic snapshot,
///    or a read-after-write must be enforced.
#[derive(Debug, Clone)]
pub struct SupersedenceRules {
    /// Whether to drop older reads when a newer full read exists
    pub drop_superseded_reads: bool,
    /// Whether to drop reads after a write to the same file
    pub drop_reads_after_write: bool,
    /// Whether to keep edit history or only the final state
    pub preserve_edit_history: bool,
    /// Maximum number of edits to preserve per file
    pub max_edits_per_file: usize,
    /// Whether to enforce read-after-write for files without snapshots
    pub enforce_read_after_write: bool,
}

impl Default for SupersedenceRules {
    fn default() -> Self {
        Self {
            drop_superseded_reads: true,
            drop_reads_after_write: true,
            preserve_edit_history: false,
            max_edits_per_file: 3,
            enforce_read_after_write: true,
        }
    }
}

impl SupersedenceRules {
    /// Determine action for a read operation
    pub fn evaluate_read(
        &self,
        artifact: &FileArtifact,
        classification: &MessageClassification,
    ) -> SupersedenceAction {
        let current_range = classification.line_range;
        let current_seq = classification.sequence;

        // Check if there's a newer write
        let has_newer_write = artifact.operations.iter().any(|op| {
            op.sequence > current_seq && op.operation == FileOperation::Write && op.is_result
        });

        if has_newer_write && self.drop_reads_after_write {
            return SupersedenceAction::Drop;
        }

        // Check if there's a newer full read
        let has_newer_full_read = artifact.operations.iter().any(|op| {
            op.sequence > current_seq
                && op.operation == FileOperation::Read
                && op.is_result
                && op.line_range.map(|r| r.is_full_read()).unwrap_or(true)
        });

        if has_newer_full_read && self.drop_superseded_reads {
            // If this is a partial read and the newer is full, drop this
            if current_range.map(|r| !r.is_full_read()).unwrap_or(false) {
                return SupersedenceAction::Drop;
            }

            // If both are full reads, drop the older one
            return SupersedenceAction::Drop;
        }

        // Check for overlapping partial reads
        // Drop this read if a newer read either contains it OR overlaps with broader coverage
        if let Some(current_range) = current_range {
            for op in &artifact.operations {
                if op.sequence > current_seq && op.operation == FileOperation::Read && op.is_result
                {
                    if let Some(other_range) = op.line_range {
                        // Drop if newer contains this one
                        if other_range.contains(&current_range) {
                            return SupersedenceAction::Drop;
                        }
                        // Also drop if they overlap and newer is broader
                        // (starts earlier or ends later)
                        if other_range.overlaps(&current_range) {
                            let other_coverage = other_range.limit.unwrap_or(usize::MAX);
                            let self_coverage = current_range.limit.unwrap_or(usize::MAX);
                            if other_range.offset <= current_range.offset
                                && other_coverage >= self_coverage
                            {
                                return SupersedenceAction::Drop;
                            }
                        }
                    }
                }
            }
        }

        // This read should be kept, but may need to move to end
        // if it's the latest full snapshot
        if artifact.latest_snapshot_key.as_ref() == Some(&classification.key()) {
            return SupersedenceAction::MoveToEnd;
        }

        SupersedenceAction::Keep
    }

    /// Determine action for a write operation
    pub fn evaluate_write(
        &self,
        artifact: &FileArtifact,
        classification: &MessageClassification,
    ) -> SupersedenceAction {
        // Write results are always the canonical snapshot
        // Move to end if this is the latest write
        let is_latest_write = !artifact.operations.iter().any(|op| {
            op.sequence > classification.sequence
                && op.operation == FileOperation::Write
                && op.is_result
        });

        if is_latest_write && classification.role == MessageRole::User {
            SupersedenceAction::MoveToEnd
        } else {
            SupersedenceAction::Keep
        }
    }

    /// Determine action for an edit operation
    pub fn evaluate_edit(
        &self,
        artifact: &FileArtifact,
        classification: &MessageClassification,
    ) -> SupersedenceAction {
        // Count edits after this one
        let edits_after = artifact
            .operations
            .iter()
            .filter(|op| {
                op.sequence > classification.sequence
                    && op.operation == FileOperation::Edit
                    && op.is_result
            })
            .count();

        // Check if there's a newer write or full read
        let has_newer_snapshot = artifact.operations.iter().any(|op| {
            op.sequence > classification.sequence
                && (op.operation == FileOperation::Write
                    || (op.operation == FileOperation::Read
                        && op.line_range.map(|r| r.is_full_read()).unwrap_or(true)))
                && op.is_result
        });

        if has_newer_snapshot {
            // Edits before a snapshot can be dropped (snapshot reflects final state)
            return SupersedenceAction::Drop;
        }

        if !self.preserve_edit_history {
            // Only keep the most recent edit
            if edits_after > 0 {
                return SupersedenceAction::Drop;
            }
        } else if edits_after >= self.max_edits_per_file {
            // Too many edits, drop older ones
            return SupersedenceAction::Drop;
        }

        SupersedenceAction::Keep
    }

    /// Evaluate a message classification and return the appropriate action
    pub fn evaluate(
        &self,
        artifact: &FileArtifact,
        classification: &MessageClassification,
    ) -> SupersedenceAction {
        match classification.operation {
            FileOperation::Read => self.evaluate_read(artifact, classification),
            FileOperation::Write => self.evaluate_write(artifact, classification),
            FileOperation::Edit => self.evaluate_edit(artifact, classification),
            FileOperation::Other => SupersedenceAction::Keep,
        }
    }
}

// ============================================================================
// Context Classifier
// ============================================================================

/// Global sequence counter for unique IDs across classifier instances
static GLOBAL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Classifier for extracting file operation metadata from messages
#[derive(Debug, Default)]
pub struct ContextClassifier {
    /// Tracked file artifacts by normalized path
    artifacts: HashMap<PathBuf, FileArtifact>,
    /// Message classifications by composite key (message_id, tool_call_id)
    classifications: HashMap<ClassificationKey, MessageClassification>,
    /// Index from message_id to all classification keys for that message
    message_index: HashMap<String, Vec<ClassificationKey>>,
    /// Current sequence counter (local, but seeded from global)
    sequence_counter: u64,
}

impl ContextClassifier {
    /// Create a new context classifier
    pub fn new() -> Self {
        Self {
            sequence_counter: GLOBAL_SEQUENCE.fetch_add(1000, Ordering::SeqCst),
            ..Default::default()
        }
    }

    /// Generate next sequence number (globally unique)
    fn next_sequence(&mut self) -> u64 {
        self.sequence_counter += 1;
        self.sequence_counter
    }

    /// Normalize a file path to absolute form
    fn normalize_path(&self, path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            // In a real implementation, this would resolve against working directory
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("/"))
                .join(path)
        }
    }

    /// Classify a tool call (from assistant message)
    pub fn classify_tool_call(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        message_id: &str,
        tool_call_id: &str,
    ) -> MessageClassification {
        let sequence = self.next_sequence();
        let (operation, file_path, line_range) = self.extract_operation_metadata(tool_name, args);

        let classification = MessageClassification {
            message_id: message_id.to_string(),
            tool_call_id: tool_call_id.to_string(),
            sequence,
            role: MessageRole::Assistant,
            operation,
            file_path: file_path.clone(),
            line_range,
            content_hash: None, // Calls don't have content yet
            content_size: 0,
        };

        let key = classification.key();

        // Record in artifact if file-related
        if let Some(path) = file_path {
            let artifact = self
                .artifacts
                .entry(path.clone())
                .or_insert_with(|| FileArtifact::new(path));

            artifact.record_operation(FileOperationRecord {
                key: key.clone(),
                sequence,
                operation,
                line_range,
                content_hash: None,
                is_result: false,
            });
        }

        // Add to message index
        self.message_index
            .entry(message_id.to_string())
            .or_default()
            .push(key.clone());

        self.classifications.insert(key, classification.clone());
        classification
    }

    /// Classify a tool result (from user message)
    pub fn classify_tool_result(
        &mut self,
        tool_name: &str,
        result: &str,
        message_id: &str,
        tool_call_id: &str,
    ) -> MessageClassification {
        let sequence = self.next_sequence();

        // Try to find the original tool call to get file path
        // Note: The call would have a different message_id, so search by tool_call_id
        let call_classification = self
            .classifications
            .values()
            .find(|c| c.tool_call_id == tool_call_id && c.role == MessageRole::Assistant);

        let (operation, file_path, line_range) = if let Some(call) = call_classification {
            (call.operation, call.file_path.clone(), call.line_range)
        } else {
            // Parse result to extract metadata
            self.extract_result_metadata(tool_name, result)
        };

        let content_hash = if operation.is_read() || operation.is_mutating() {
            Some(ContentHash::from_content(result))
        } else {
            None
        };

        let classification = MessageClassification {
            message_id: message_id.to_string(),
            tool_call_id: tool_call_id.to_string(),
            sequence,
            role: MessageRole::User,
            operation,
            file_path: file_path.clone(),
            line_range,
            content_hash: content_hash.clone(),
            content_size: result.len(),
        };

        let key = classification.key();

        // Record in artifact if file-related
        if let Some(path) = file_path {
            let artifact = self
                .artifacts
                .entry(path.clone())
                .or_insert_with(|| FileArtifact::new(path));

            artifact.record_operation(FileOperationRecord {
                key: key.clone(),
                sequence,
                operation,
                line_range,
                content_hash,
                is_result: true,
            });
        }

        // Add to message index
        self.message_index
            .entry(message_id.to_string())
            .or_default()
            .push(key.clone());

        self.classifications.insert(key, classification.clone());
        classification
    }

    /// Extract operation metadata from tool call arguments
    fn extract_operation_metadata(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> (FileOperation, Option<PathBuf>, Option<LineRange>) {
        match tool_name {
            "read" => {
                let file_path = args
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(|s| self.normalize_path(s));

                let offset = args
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .unwrap_or(0);

                let limit = args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);

                let line_range = if offset == 0 && limit.is_none() {
                    None // Full read
                } else {
                    Some(LineRange::new(offset, limit))
                };

                (FileOperation::Read, file_path, line_range)
            }
            "write" => {
                let file_path = args
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(|s| self.normalize_path(s));

                (FileOperation::Write, file_path, None)
            }
            "edit" => {
                let file_path = args
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(|s| self.normalize_path(s));

                (FileOperation::Edit, file_path, None)
            }
            _ => (FileOperation::Other, None, None),
        }
    }

    /// Extract metadata from tool result (fallback when call not found)
    fn extract_result_metadata(
        &self,
        tool_name: &str,
        result: &str,
    ) -> (FileOperation, Option<PathBuf>, Option<LineRange>) {
        // Try to parse JSON result for file path
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(result) {
            let file_path = json
                .get("file_path")
                .and_then(|v| v.as_str())
                .map(|s| self.normalize_path(s));

            let operation = match tool_name {
                "read" => FileOperation::Read,
                "write" => FileOperation::Write,
                "edit" => FileOperation::Edit,
                _ => FileOperation::Other,
            };

            return (operation, file_path, None);
        }

        (FileOperation::Other, None, None)
    }

    /// Get artifact for a file path
    pub fn get_artifact(&self, path: &PathBuf) -> Option<&FileArtifact> {
        self.artifacts.get(path)
    }

    /// Get all tracked artifacts
    pub fn artifacts(&self) -> &HashMap<PathBuf, FileArtifact> {
        &self.artifacts
    }

    /// Get classification by composite key
    pub fn get_classification_by_key(
        &self,
        key: &ClassificationKey,
    ) -> Option<&MessageClassification> {
        self.classifications.get(key)
    }

    /// Get all classifications for a message ID
    pub fn get_classifications_for_message(&self, message_id: &str) -> Vec<&MessageClassification> {
        self.message_index
            .get(message_id)
            .map(|keys| {
                keys.iter()
                    .filter_map(|k| self.classifications.get(k))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a message has any classifications
    pub fn is_message_classified(&self, message_id: &str) -> bool {
        self.message_index.contains_key(message_id)
    }

    /// Get classification for a message (legacy API - returns first classification)
    #[deprecated(note = "Use get_classifications_for_message for multi-tool support")]
    pub fn get_classification(&self, message_id: &str) -> Option<&MessageClassification> {
        self.get_classifications_for_message(message_id)
            .into_iter()
            .next()
    }

    /// Get all classifications
    pub fn classifications(&self) -> &HashMap<ClassificationKey, MessageClassification> {
        &self.classifications
    }

    /// Get all classification keys for a message
    pub fn get_keys_for_message(&self, message_id: &str) -> Option<&Vec<ClassificationKey>> {
        self.message_index.get(message_id)
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.artifacts.clear();
        self.classifications.clear();
        self.message_index.clear();
        self.sequence_counter = GLOBAL_SEQUENCE.fetch_add(1000, Ordering::SeqCst);
    }
}

// ============================================================================
// Context Manager
// ============================================================================

/// Manages context optimization for file operations
///
/// The ContextManager uses the classifier and supersedence rules to:
/// 1. Track all file operations in a conversation
/// 2. Determine which messages should be kept, dropped, or moved
/// 3. Produce an optimized message ordering for LLM context
#[derive(Debug)]
pub struct ContextManager {
    /// The classifier for extracting metadata
    classifier: ContextClassifier,
    /// Supersedence rules to apply
    rules: SupersedenceRules,
}

impl ContextManager {
    /// Create a new context manager with default rules
    pub fn new() -> Self {
        Self {
            classifier: ContextClassifier::new(),
            rules: SupersedenceRules::default(),
        }
    }

    /// Create with custom supersedence rules
    pub fn with_rules(rules: SupersedenceRules) -> Self {
        Self {
            classifier: ContextClassifier::new(),
            rules,
        }
    }

    /// Get a reference to the classifier
    pub fn classifier(&self) -> &ContextClassifier {
        &self.classifier
    }

    /// Get a mutable reference to the classifier
    pub fn classifier_mut(&mut self) -> &mut ContextClassifier {
        &mut self.classifier
    }

    /// Get the supersedence rules
    pub fn rules(&self) -> &SupersedenceRules {
        &self.rules
    }

    /// Compute the optimized order of classification keys
    ///
    /// Returns actions for each classification (not message). A single message
    /// may have multiple classifications if it contains multiple tool calls.
    pub fn compute_optimized_order(&self) -> ContextOptimizationResult<'_> {
        let mut keep: Vec<ClassificationKey> = Vec::new();
        let mut drop: Vec<ClassificationKey> = Vec::new();
        let mut move_to_end: Vec<ClassificationKey> = Vec::new();

        // Sort classifications by sequence
        let mut sorted_classifications: Vec<_> = self.classifier.classifications.values().collect();
        sorted_classifications.sort_by_key(|c| c.sequence);

        for classification in sorted_classifications {
            let action = if let Some(path) = &classification.file_path {
                if let Some(artifact) = self.classifier.artifacts.get(path) {
                    self.rules.evaluate(artifact, classification)
                } else {
                    SupersedenceAction::Keep
                }
            } else {
                SupersedenceAction::Keep
            };

            let key = classification.key();

            match action {
                SupersedenceAction::Keep => {
                    keep.push(key);
                }
                SupersedenceAction::Drop => {
                    drop.push(key);
                }
                SupersedenceAction::MoveToEnd => {
                    move_to_end.push(key);
                }
                SupersedenceAction::ReplaceWithSnapshot => {
                    // For now, treat as keep - synthetic snapshot generation
                    // would be handled separately
                    keep.push(key);
                }
            }
        }

        ContextOptimizationResult {
            keep,
            drop,
            move_to_end,
            classifier_ref: &self.classifier,
        }
    }

    /// Check if a file needs a read-after-write
    ///
    /// Returns true if the file has been written/edited but has no
    /// subsequent full read, and enforce_read_after_write is enabled.
    pub fn needs_read_after_write(&self, path: &PathBuf) -> bool {
        if !self.rules.enforce_read_after_write {
            return false;
        }

        if let Some(artifact) = self.classifier.artifacts.get(path) {
            // Check if file was mutated
            let was_mutated = artifact
                .operations
                .iter()
                .any(|op| op.operation.is_mutating() && op.is_result);

            if !was_mutated {
                return false;
            }

            // Check if there's a full read after the last mutation
            let last_mutation_seq = artifact
                .operations
                .iter()
                .filter(|op| op.operation.is_mutating() && op.is_result)
                .map(|op| op.sequence)
                .max()
                .unwrap_or(0);

            let has_read_after = artifact.operations.iter().any(|op| {
                op.sequence > last_mutation_seq
                    && op.operation == FileOperation::Read
                    && op.is_result
                    && op.line_range.map(|r| r.is_full_read()).unwrap_or(true)
            });

            !has_read_after
        } else {
            false
        }
    }

    /// Get files that need read-after-write
    pub fn files_needing_read(&self) -> Vec<PathBuf> {
        self.classifier
            .artifacts
            .keys()
            .filter(|path| self.needs_read_after_write(path))
            .cloned()
            .collect()
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.classifier.clear();
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of context optimization computation
#[derive(Debug)]
pub struct ContextOptimizationResult<'a> {
    /// Classification keys to keep in their original position
    pub keep: Vec<ClassificationKey>,
    /// Classification keys to drop
    pub drop: Vec<ClassificationKey>,
    /// Classification keys to move to the end of context
    pub move_to_end: Vec<ClassificationKey>,
    /// Reference to the classifier for message lookups
    classifier_ref: &'a ContextClassifier,
}

impl<'a> ContextOptimizationResult<'a> {
    /// Get the final ordered list of classification keys
    pub fn final_order(&self) -> Vec<ClassificationKey> {
        let mut result = self.keep.clone();
        result.extend(self.move_to_end.iter().cloned());
        result
    }

    /// Check if a classification key should be included
    pub fn should_include_key(&self, key: &ClassificationKey) -> bool {
        self.keep.contains(key) || self.move_to_end.contains(key)
    }

    /// Check if a classification key should be dropped
    pub fn should_drop_key(&self, key: &ClassificationKey) -> bool {
        self.drop.contains(key)
    }

    /// Check if a message (by ID) has any classifications that should be included
    ///
    /// For messages with multiple tool calls, returns true if ANY of them should be included.
    pub fn should_include(&self, message_id: &str) -> bool {
        if let Some(keys) = self.classifier_ref.get_keys_for_message(message_id) {
            keys.iter().any(|k| self.should_include_key(k))
        } else {
            // Message is not classified - it's not a file operation
            // The caller should decide how to handle unclassified messages
            false
        }
    }

    /// Check if a message (by ID) should be dropped
    ///
    /// Returns true only if ALL classifications for this message should be dropped.
    /// Returns false if the message has no classifications (it's not a file operation).
    pub fn should_drop(&self, message_id: &str) -> bool {
        if let Some(keys) = self.classifier_ref.get_keys_for_message(message_id) {
            !keys.is_empty() && keys.iter().all(|k| self.should_drop_key(k))
        } else {
            // Message is not classified - don't drop it
            false
        }
    }

    /// Check if a message ID has any classifications
    pub fn is_classified(&self, message_id: &str) -> bool {
        self.classifier_ref.is_message_classified(message_id)
    }

    /// Get all unique message IDs in final order
    ///
    /// Note: This only includes classified messages. Unclassified messages
    /// (normal chat, non-file tools) should be preserved separately.
    pub fn final_message_order(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for key in self.final_order() {
            if seen.insert(key.message_id.clone()) {
                result.push(key.message_id);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_range_is_full_read() {
        assert!(LineRange::new(0, None).is_full_read());
        assert!(!LineRange::new(10, None).is_full_read());
        assert!(!LineRange::new(0, Some(100)).is_full_read());
    }

    #[test]
    fn test_line_range_contains() {
        let full = LineRange::new(0, None);
        let partial1 = LineRange::new(10, Some(50));
        let partial2 = LineRange::new(20, Some(30));

        assert!(full.contains(&partial1));
        assert!(full.contains(&partial2));
        assert!(!partial1.contains(&full));
        assert!(partial1.contains(&partial2));
    }

    #[test]
    fn test_line_range_overlaps() {
        let range1 = LineRange::new(0, Some(50));
        let range2 = LineRange::new(40, Some(30));
        let range3 = LineRange::new(60, Some(20));

        assert!(range1.overlaps(&range2));
        assert!(!range1.overlaps(&range3));
    }

    #[test]
    fn test_content_hash() {
        let hash1 = ContentHash::from_content("hello world");
        let hash2 = ContentHash::from_content("hello world");
        let hash3 = ContentHash::from_content("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_classifier_read_operation() {
        let mut classifier = ContextClassifier::new();

        let args = serde_json::json!({
            "file_path": "/test/file.rs"
        });

        let classification = classifier.classify_tool_call("read", &args, "msg-1", "call-1");

        assert_eq!(classification.operation, FileOperation::Read);
        assert!(classification.file_path.is_some());
        assert!(classification.line_range.is_none()); // Full read
    }

    #[test]
    fn test_classifier_partial_read() {
        let mut classifier = ContextClassifier::new();

        let args = serde_json::json!({
            "file_path": "/test/file.rs",
            "offset": 10,
            "limit": 50
        });

        let classification = classifier.classify_tool_call("read", &args, "msg-1", "call-1");

        assert_eq!(classification.operation, FileOperation::Read);
        assert!(classification.line_range.is_some());

        let range = classification.line_range.unwrap();
        assert_eq!(range.offset, 10);
        assert_eq!(range.limit, Some(50));
    }

    #[test]
    fn test_classifier_write_operation() {
        let mut classifier = ContextClassifier::new();

        let args = serde_json::json!({
            "file_path": "/test/new_file.rs",
            "content": "fn main() {}"
        });

        let classification = classifier.classify_tool_call("write", &args, "msg-1", "call-1");

        assert_eq!(classification.operation, FileOperation::Write);
        assert!(classification.file_path.is_some());
    }

    #[test]
    fn test_classifier_edit_operation() {
        let mut classifier = ContextClassifier::new();

        let args = serde_json::json!({
            "file_path": "/test/file.rs",
            "old_string": "old",
            "new_string": "new"
        });

        let classification = classifier.classify_tool_call("edit", &args, "msg-1", "call-1");

        assert_eq!(classification.operation, FileOperation::Edit);
        assert!(classification.file_path.is_some());
    }

    #[test]
    fn test_classifier_multiple_tool_calls_same_message() {
        let mut classifier = ContextClassifier::new();

        // Two tool calls in the same assistant message
        classifier.classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/test/file1.rs"}),
            "msg-1",
            "call-1",
        );
        classifier.classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/test/file2.rs"}),
            "msg-1",
            "call-2",
        );

        // Both should be recorded
        let classifications = classifier.get_classifications_for_message("msg-1");
        assert_eq!(classifications.len(), 2);

        // Check both files are tracked
        assert!(classifier.is_message_classified("msg-1"));
        let keys = classifier.get_keys_for_message("msg-1").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_file_artifact_state_transitions() {
        let mut artifact = FileArtifact::new(PathBuf::from("/test/file.rs"));

        // Initial read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-1", "call-1"),
            sequence: 1,
            operation: FileOperation::Read,
            line_range: None,
            content_hash: Some(ContentHash::from_content("v1")),
            is_result: true,
        });
        assert_eq!(artifact.state, FileState::Read);
        // Full read should set latest_snapshot_key
        assert!(artifact.latest_snapshot_key.is_some());

        // Edit changes state
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-2", "call-2"),
            sequence: 2,
            operation: FileOperation::Edit,
            line_range: None,
            content_hash: Some(ContentHash::from_content("v2")),
            is_result: true,
        });
        assert_eq!(artifact.state, FileState::Edited);

        // Write trumps edit
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-3", "call-3"),
            sequence: 3,
            operation: FileOperation::Write,
            line_range: None,
            content_hash: Some(ContentHash::from_content("v3")),
            is_result: true,
        });
        assert_eq!(artifact.state, FileState::Written);
        assert_eq!(
            artifact.latest_snapshot_key,
            Some(ClassificationKey::new("msg-3", "call-3"))
        );
    }

    #[test]
    fn test_supersedence_read_after_write() {
        let rules = SupersedenceRules::default();
        let mut artifact = FileArtifact::new(PathBuf::from("/test/file.rs"));

        // Record a read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-1", "call-1"),
            sequence: 1,
            operation: FileOperation::Read,
            line_range: None,
            content_hash: Some(ContentHash::from_content("v1")),
            is_result: true,
        });

        // Record a write after the read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-2", "call-2"),
            sequence: 2,
            operation: FileOperation::Write,
            line_range: None,
            content_hash: Some(ContentHash::from_content("v2")),
            is_result: true,
        });

        // The old read should be dropped
        let classification = MessageClassification {
            message_id: "msg-1".to_string(),
            tool_call_id: "call-1".to_string(),
            sequence: 1,
            role: MessageRole::User,
            operation: FileOperation::Read,
            file_path: Some(PathBuf::from("/test/file.rs")),
            line_range: None,
            content_hash: Some(ContentHash::from_content("v1")),
            content_size: 100,
        };

        let action = rules.evaluate(&artifact, &classification);
        assert_eq!(action, SupersedenceAction::Drop);
    }

    #[test]
    fn test_supersedence_newer_full_read() {
        let rules = SupersedenceRules::default();
        let mut artifact = FileArtifact::new(PathBuf::from("/test/file.rs"));

        // Old partial read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-1", "call-1"),
            sequence: 1,
            operation: FileOperation::Read,
            line_range: Some(LineRange::new(10, Some(50))),
            content_hash: Some(ContentHash::from_content("partial")),
            is_result: true,
        });

        // Newer full read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-2", "call-2"),
            sequence: 2,
            operation: FileOperation::Read,
            line_range: None,
            content_hash: Some(ContentHash::from_content("full")),
            is_result: true,
        });

        // Old partial read should be dropped
        let classification = MessageClassification {
            message_id: "msg-1".to_string(),
            tool_call_id: "call-1".to_string(),
            sequence: 1,
            role: MessageRole::User,
            operation: FileOperation::Read,
            file_path: Some(PathBuf::from("/test/file.rs")),
            line_range: Some(LineRange::new(10, Some(50))),
            content_hash: Some(ContentHash::from_content("partial")),
            content_size: 100,
        };

        let action = rules.evaluate(&artifact, &classification);
        assert_eq!(action, SupersedenceAction::Drop);
    }

    #[test]
    fn test_latest_full_read_moves_to_end() {
        let rules = SupersedenceRules::default();
        let mut artifact = FileArtifact::new(PathBuf::from("/test/file.rs"));

        // Full read
        artifact.record_operation(FileOperationRecord {
            key: ClassificationKey::new("msg-1", "call-1"),
            sequence: 1,
            operation: FileOperation::Read,
            line_range: None,
            content_hash: Some(ContentHash::from_content("full")),
            is_result: true,
        });

        // The latest full read should move to end
        let classification = MessageClassification {
            message_id: "msg-1".to_string(),
            tool_call_id: "call-1".to_string(),
            sequence: 1,
            role: MessageRole::User,
            operation: FileOperation::Read,
            file_path: Some(PathBuf::from("/test/file.rs")),
            line_range: None,
            content_hash: Some(ContentHash::from_content("full")),
            content_size: 100,
        };

        let action = rules.evaluate(&artifact, &classification);
        assert_eq!(action, SupersedenceAction::MoveToEnd);
    }

    #[test]
    fn test_context_manager_optimization() {
        let mut manager = ContextManager::new();

        // Simulate: read -> edit -> read sequence
        let args = serde_json::json!({"file_path": "/test/file.rs"});

        // First read
        manager
            .classifier
            .classify_tool_call("read", &args, "msg-1", "call-1");
        manager.classifier.classify_tool_result(
            "read",
            r#"{"file_path":"/test/file.rs","content":"v1"}"#,
            "msg-2",
            "call-1",
        );

        // Edit
        let edit_args = serde_json::json!({
            "file_path": "/test/file.rs",
            "old_string": "old",
            "new_string": "new"
        });
        manager
            .classifier
            .classify_tool_call("edit", &edit_args, "msg-3", "call-2");
        manager.classifier.classify_tool_result(
            "edit",
            r#"{"file_path":"/test/file.rs"}"#,
            "msg-4",
            "call-2",
        );

        // Second read (after edit)
        manager
            .classifier
            .classify_tool_call("read", &args, "msg-5", "call-3");
        manager.classifier.classify_tool_result(
            "read",
            r#"{"file_path":"/test/file.rs","content":"v2"}"#,
            "msg-6",
            "call-3",
        );

        let result = manager.compute_optimized_order();

        // First read result (msg-2) should be dropped
        // because there's a newer full read (msg-6)
        assert!(result.should_drop("msg-2"));

        // Edit should be dropped because there's a newer read
        assert!(result.should_drop("msg-4"));

        // Latest read should be kept and moved to end
        assert!(!result.should_drop("msg-6"));
    }

    #[test]
    fn test_unclassified_messages_not_dropped() {
        let mut manager = ContextManager::new();

        // Only classify one message
        manager.classifier.classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/test/file.rs"}),
            "msg-1",
            "call-1",
        );

        let result = manager.compute_optimized_order();

        // Unclassified messages should not be reported as dropped
        assert!(!result.should_drop("msg-unclassified"));
        assert!(!result.is_classified("msg-unclassified"));
    }

    #[test]
    fn test_needs_read_after_write() {
        let mut manager = ContextManager::new();
        let path = PathBuf::from("/test/file.rs");

        // Write without subsequent read
        let args = serde_json::json!({
            "file_path": "/test/file.rs",
            "content": "fn main() {}"
        });
        manager
            .classifier
            .classify_tool_call("write", &args, "msg-1", "call-1");
        manager.classifier.classify_tool_result(
            "write",
            r#"{"file_path":"/test/file.rs"}"#,
            "msg-2",
            "call-1",
        );

        assert!(manager.needs_read_after_write(&path));

        // Add a full read
        let read_args = serde_json::json!({"file_path": "/test/file.rs"});
        manager
            .classifier
            .classify_tool_call("read", &read_args, "msg-3", "call-2");
        manager.classifier.classify_tool_result(
            "read",
            r#"{"file_path":"/test/file.rs","content":"fn main() {}"}"#,
            "msg-4",
            "call-2",
        );

        assert!(!manager.needs_read_after_write(&path));
    }
}
