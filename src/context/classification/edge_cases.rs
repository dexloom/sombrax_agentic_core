//! Edge Case Handling for File-History Context Management
//!
//! This module handles complex scenarios that require special treatment:
//! - Partial reads with overlapping or non-contiguous ranges
//! - Multi-file patches (atomic operations across files)
//! - File renames and deletions
//! - Binary and large files
//! - Synthetic snapshot generation

use super::{
    ContentHash, ContextManager, FileArtifact, FileOperation, LineRange, MessageClassification,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// Partial Read Handling
// ============================================================================

/// Tracks partial reads of a file to determine if they can be merged
/// or if a full read is needed
#[derive(Debug, Clone)]
pub struct PartialReadTracker {
    /// File path being tracked
    pub path: PathBuf,
    /// Collected partial reads, sorted by offset
    pub ranges: Vec<TrackedRange>,
    /// Whether a full read exists
    pub has_full_read: bool,
}

/// A tracked partial read range with its message ID
#[derive(Debug, Clone)]
pub struct TrackedRange {
    /// Line range of the read
    pub range: LineRange,
    /// Message ID containing this read
    pub message_id: String,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Content hash of this range
    pub content_hash: Option<ContentHash>,
}

impl PartialReadTracker {
    /// Create a new tracker for a file
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            ranges: Vec::new(),
            has_full_read: false,
        }
    }

    /// Add a partial read
    pub fn add_range(&mut self, range: TrackedRange) {
        if range.range.is_full_read() {
            self.has_full_read = true;
        }
        self.ranges.push(range);
        self.ranges.sort_by_key(|r| r.range.offset);
    }

    /// Check if the tracked ranges cover the entire file (contiguous from 0)
    ///
    /// Note: Without knowing the actual file length, we can only determine
    /// if ranges are contiguous from the start.
    pub fn has_contiguous_coverage_from_start(&self) -> bool {
        if self.has_full_read {
            return true;
        }

        if self.ranges.is_empty() {
            return false;
        }

        // Check if we have a range starting at 0
        let first = &self.ranges[0];
        if first.range.offset != 0 {
            return false;
        }

        // Check contiguity
        let mut end = first.range.limit.unwrap_or(usize::MAX);
        for range in self.ranges.iter().skip(1) {
            if range.range.offset > end {
                // Gap detected
                return false;
            }
            if let Some(limit) = range.range.limit {
                end = end.max(range.range.offset + limit);
            } else {
                // Unlimited range encountered
                return true;
            }
        }

        true
    }

    /// Get ranges that are superseded by other ranges
    pub fn superseded_ranges(&self) -> Vec<String> {
        let mut superseded = Vec::new();

        for (i, range_a) in self.ranges.iter().enumerate() {
            for range_b in self.ranges.iter().skip(i + 1) {
                // If b completely contains a and is newer, a is superseded
                if range_b.range.contains(&range_a.range) && range_b.sequence > range_a.sequence {
                    superseded.push(range_a.message_id.clone());
                    break;
                }
                // If a is a full read and b is newer partial, b is redundant (not a)
            }
        }

        superseded
    }

    /// Compute gaps in coverage (ranges not covered by any read)
    pub fn coverage_gaps(&self) -> Vec<LineRange> {
        if self.has_full_read || self.ranges.is_empty() {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        let mut current_end = 0;

        for range in &self.ranges {
            if range.range.offset > current_end {
                gaps.push(LineRange::new(
                    current_end,
                    Some(range.range.offset - current_end),
                ));
            }
            if let Some(limit) = range.range.limit {
                current_end = current_end.max(range.range.offset + limit);
            } else {
                // Unlimited - no more gaps possible after this
                return gaps;
            }
        }

        gaps
    }
}

// ============================================================================
// Multi-File Patch Handling
// ============================================================================

/// Represents an atomic operation spanning multiple files
#[derive(Debug, Clone)]
pub struct MultiFilePatch {
    /// Unique identifier for this patch group
    pub patch_id: String,
    /// Files involved in this atomic operation
    pub files: Vec<PathBuf>,
    /// Message IDs belonging to this patch group
    pub message_ids: Vec<String>,
    /// Whether all operations succeeded
    pub is_complete: bool,
}

impl MultiFilePatch {
    /// Create a new multi-file patch
    pub fn new(patch_id: impl Into<String>) -> Self {
        Self {
            patch_id: patch_id.into(),
            files: Vec::new(),
            message_ids: Vec::new(),
            is_complete: false,
        }
    }

    /// Add a file to this patch
    pub fn add_file(&mut self, path: PathBuf, message_id: String) {
        if !self.files.contains(&path) {
            self.files.push(path);
        }
        if !self.message_ids.contains(&message_id) {
            self.message_ids.push(message_id);
        }
    }

    /// Mark the patch as complete
    pub fn mark_complete(&mut self) {
        self.is_complete = true;
    }
}

/// Tracks multi-file patches for atomic handling
#[derive(Debug, Default)]
pub struct MultiFilePatchTracker {
    /// Active patch groups by ID
    patches: HashMap<String, MultiFilePatch>,
    /// Mapping from message ID to patch ID
    message_to_patch: HashMap<String, String>,
}

impl MultiFilePatchTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new patch group
    pub fn start_patch(&mut self, patch_id: impl Into<String>) {
        let patch_id = patch_id.into();
        self.patches
            .insert(patch_id.clone(), MultiFilePatch::new(patch_id));
    }

    /// Add a file operation to a patch
    pub fn add_to_patch(&mut self, patch_id: &str, path: PathBuf, message_id: String) {
        if let Some(patch) = self.patches.get_mut(patch_id) {
            patch.add_file(path, message_id.clone());
            self.message_to_patch
                .insert(message_id, patch_id.to_string());
        }
    }

    /// Complete a patch group
    pub fn complete_patch(&mut self, patch_id: &str) {
        if let Some(patch) = self.patches.get_mut(patch_id) {
            patch.mark_complete();
        }
    }

    /// Get the patch group for a message
    pub fn get_patch_for_message(&self, message_id: &str) -> Option<&MultiFilePatch> {
        self.message_to_patch
            .get(message_id)
            .and_then(|patch_id| self.patches.get(patch_id))
    }

    /// Check if a message is part of an incomplete atomic patch
    pub fn is_incomplete_atomic(&self, message_id: &str) -> bool {
        self.get_patch_for_message(message_id)
            .map(|p| !p.is_complete)
            .unwrap_or(false)
    }
}

// ============================================================================
// Rename and Delete Handling
// ============================================================================

/// Tracks file renames for proper artifact linking
#[derive(Debug, Clone)]
pub struct FileRename {
    /// Original path before rename
    pub old_path: PathBuf,
    /// New path after rename
    pub new_path: PathBuf,
    /// Message ID of the rename operation
    pub message_id: String,
    /// Sequence number
    pub sequence: u64,
}

/// Tracks file deletions
#[derive(Debug, Clone)]
pub struct FileDeletion {
    /// Path of deleted file
    pub path: PathBuf,
    /// Message ID of the delete operation
    pub message_id: String,
    /// Sequence number
    pub sequence: u64,
}

/// Manager for rename and delete operations
#[derive(Debug, Default)]
pub struct FileLifecycleTracker {
    /// Recorded renames
    pub renames: Vec<FileRename>,
    /// Recorded deletions
    pub deletions: Vec<FileDeletion>,
    /// Path alias map (old -> new)
    path_aliases: HashMap<PathBuf, PathBuf>,
}

impl FileLifecycleTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a file rename
    pub fn record_rename(&mut self, rename: FileRename) {
        self.path_aliases
            .insert(rename.old_path.clone(), rename.new_path.clone());
        self.renames.push(rename);
    }

    /// Record a file deletion
    pub fn record_deletion(&mut self, deletion: FileDeletion) {
        self.deletions.push(deletion);
    }

    /// Resolve a path through any renames
    pub fn resolve_path(&self, path: &Path) -> PathBuf {
        let mut current = path.to_path_buf();
        let mut visited = std::collections::HashSet::new();

        while let Some(new_path) = self.path_aliases.get(&current) {
            if visited.contains(new_path) {
                // Cycle detected, break
                break;
            }
            visited.insert(current.clone());
            current = new_path.clone();
        }

        current
    }

    /// Check if a file has been deleted
    pub fn is_deleted(&self, path: &Path) -> bool {
        let resolved = self.resolve_path(path);
        self.deletions.iter().any(|d| d.path == resolved)
    }

    /// Get the deletion sequence for a path (if deleted)
    pub fn deletion_sequence(&self, path: &Path) -> Option<u64> {
        let resolved = self.resolve_path(path);
        self.deletions
            .iter()
            .find(|d| d.path == resolved)
            .map(|d| d.sequence)
    }
}

// ============================================================================
// Binary and Large File Handling
// ============================================================================

/// Policy for handling binary files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFilePolicy {
    /// Keep only the most recent reference, don't store content
    KeepReference,
    /// Store hash only, no content
    HashOnly,
    /// Treat as text (may cause issues)
    TreatAsText,
    /// Exclude from context entirely
    Exclude,
}

/// Policy for handling large files
#[derive(Debug, Clone)]
pub struct LargeFilePolicy {
    /// Maximum file size in bytes to include in context
    pub max_size_bytes: usize,
    /// Maximum lines to include
    pub max_lines: usize,
    /// Whether to truncate or exclude
    pub truncate: bool,
    /// Whether to include a summary instead
    pub include_summary: bool,
}

impl Default for LargeFilePolicy {
    fn default() -> Self {
        Self {
            max_size_bytes: 100_000, // 100KB
            max_lines: 2000,
            truncate: true,
            include_summary: true,
        }
    }
}

/// Detects binary content in a string
pub fn is_likely_binary(content: &str) -> bool {
    // Check for null bytes or high proportion of non-printable characters
    let non_printable = content
        .chars()
        .take(1000) // Sample first 1000 chars
        .filter(|c| !c.is_ascii_graphic() && !c.is_ascii_whitespace())
        .count();

    let sample_size = content.chars().take(1000).count();
    if sample_size == 0 {
        return false;
    }

    // If more than 10% non-printable, likely binary
    (non_printable as f64 / sample_size as f64) > 0.1
}

/// Metadata about a potentially problematic file
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// Whether the file appears to be binary
    pub is_binary: bool,
    /// File size in bytes (if known)
    pub size_bytes: Option<usize>,
    /// Line count (if known)
    pub line_count: Option<usize>,
    /// Recommended action based on policies
    pub recommended_action: FileAction,
}

/// Recommended action for a file
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileAction {
    /// Include normally
    Include,
    /// Truncate to specified limits
    Truncate {
        /// Maximum number of lines to include
        max_lines: usize,
    },
    /// Include only a reference/hash
    ReferenceOnly,
    /// Exclude from context
    Exclude,
}

/// Evaluate a file against policies and return metadata
pub fn evaluate_file(
    content: &str,
    binary_policy: BinaryFilePolicy,
    large_policy: &LargeFilePolicy,
) -> FileMetadata {
    let is_binary = is_likely_binary(content);
    let size_bytes = content.len();
    let line_count = content.lines().count();

    let recommended_action = if is_binary {
        match binary_policy {
            BinaryFilePolicy::KeepReference | BinaryFilePolicy::HashOnly => {
                FileAction::ReferenceOnly
            }
            BinaryFilePolicy::TreatAsText => FileAction::Include,
            BinaryFilePolicy::Exclude => FileAction::Exclude,
        }
    } else if size_bytes > large_policy.max_size_bytes || line_count > large_policy.max_lines {
        if large_policy.truncate {
            FileAction::Truncate {
                max_lines: large_policy.max_lines,
            }
        } else {
            FileAction::Exclude
        }
    } else {
        FileAction::Include
    };

    FileMetadata {
        is_binary,
        size_bytes: Some(size_bytes),
        line_count: Some(line_count),
        recommended_action,
    }
}

// ============================================================================
// Synthetic Snapshot Generation
// ============================================================================

/// Represents a synthetic file snapshot built from patches
#[derive(Debug, Clone)]
pub struct SyntheticSnapshot {
    /// File path
    pub path: PathBuf,
    /// Reconstructed content
    pub content: String,
    /// Content hash of synthetic content
    pub content_hash: ContentHash,
    /// Message IDs of patches used to build this
    pub source_patches: Vec<String>,
    /// Whether reconstruction was successful
    pub is_complete: bool,
    /// Any warnings during reconstruction
    pub warnings: Vec<String>,
}

/// Configuration for synthetic snapshot generation
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Whether to attempt synthetic reconstruction
    pub enable_reconstruction: bool,
    /// Maximum number of patches to apply
    pub max_patches: usize,
    /// Whether to fail on patch conflict
    pub strict_mode: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            enable_reconstruction: true,
            max_patches: 50,
            strict_mode: false,
        }
    }
}

/// Attempt to build a synthetic snapshot from patches
///
/// This is a simplified implementation. In production, you would need:
/// - Proper diff/patch application logic
/// - Conflict resolution
/// - Base content tracking
pub fn build_synthetic_snapshot(
    artifact: &FileArtifact,
    _config: &SnapshotConfig,
) -> Result<SyntheticSnapshot, String> {
    // This is a placeholder implementation
    // Real implementation would:
    // 1. Find the last known good state (read or write)
    // 2. Apply patches in sequence
    // 3. Handle conflicts

    let source_patches: Vec<String> = artifact
        .operations
        .iter()
        .filter(|op| op.operation == FileOperation::Edit && op.is_result)
        .map(|op| op.key.message_id.clone())
        .collect();

    // For now, return an incomplete snapshot indicating reconstruction is needed
    Ok(SyntheticSnapshot {
        path: artifact.path.clone(),
        content: String::new(), // Would be reconstructed content
        content_hash: ContentHash::from_content(""),
        source_patches,
        is_complete: false,
        warnings: vec!["Synthetic reconstruction not fully implemented".to_string()],
    })
}

// ============================================================================
// Safe Fallback Strategies
// ============================================================================

/// Fallback strategy when normal processing fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Keep all messages (no optimization)
    KeepAll,
    /// Keep only the most recent operation per file
    KeepLatestOnly,
    /// Request a fresh read of all files
    RequestFreshReads,
    /// Truncate to most recent N operations
    TruncateRecent {
        /// Maximum number of operations to keep
        max_operations: usize,
    },
}

/// Determine the appropriate fallback strategy based on the error
pub fn determine_fallback(
    error: &str,
    file_count: usize,
    operation_count: usize,
) -> FallbackStrategy {
    // If we have too many operations, truncate
    if operation_count > 100 {
        return FallbackStrategy::TruncateRecent { max_operations: 50 };
    }

    // If we have many files and a reconstruction error, just keep latest
    if file_count > 10 && error.contains("reconstruction") {
        return FallbackStrategy::KeepLatestOnly;
    }

    // Default to keeping all to be safe
    FallbackStrategy::KeepAll
}

/// Apply a fallback strategy to the context manager
pub fn apply_fallback(manager: &ContextManager, strategy: FallbackStrategy) -> Vec<String> {
    match strategy {
        FallbackStrategy::KeepAll => {
            // Return all message IDs in order
            let mut ids: Vec<_> = manager.classifier().classifications().values().collect();
            ids.sort_by_key(|c| c.sequence);
            ids.into_iter().map(|c| c.message_id.clone()).collect()
        }
        FallbackStrategy::KeepLatestOnly => {
            // Keep only the latest operation per file
            let mut latest_per_file: HashMap<PathBuf, &MessageClassification> = HashMap::new();

            for classification in manager.classifier().classifications().values() {
                if let Some(path) = &classification.file_path {
                    let entry = latest_per_file
                        .entry(path.clone())
                        .or_insert(classification);
                    if classification.sequence > entry.sequence {
                        *entry = classification;
                    }
                }
            }

            let mut ids: Vec<_> = latest_per_file.values().collect();
            ids.sort_by_key(|c| c.sequence);
            ids.into_iter().map(|c| c.message_id.clone()).collect()
        }
        FallbackStrategy::RequestFreshReads => {
            // Return empty - caller should trigger fresh reads
            Vec::new()
        }
        FallbackStrategy::TruncateRecent { max_operations } => {
            let mut ids: Vec<_> = manager.classifier().classifications().values().collect();
            ids.sort_by_key(|c| c.sequence);

            // Take the most recent N
            ids.into_iter()
                .rev()
                .take(max_operations)
                .map(|c| c.message_id.clone())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_read_tracker_contiguous() {
        let mut tracker = PartialReadTracker::new(PathBuf::from("/test/file.rs"));

        // Add contiguous ranges
        tracker.add_range(TrackedRange {
            range: LineRange::new(0, Some(50)),
            message_id: "msg-1".to_string(),
            sequence: 1,
            content_hash: None,
        });

        tracker.add_range(TrackedRange {
            range: LineRange::new(50, Some(50)),
            message_id: "msg-2".to_string(),
            sequence: 2,
            content_hash: None,
        });

        assert!(tracker.has_contiguous_coverage_from_start());
    }

    #[test]
    fn test_partial_read_tracker_gap() {
        let mut tracker = PartialReadTracker::new(PathBuf::from("/test/file.rs"));

        // Add ranges with gap
        tracker.add_range(TrackedRange {
            range: LineRange::new(0, Some(50)),
            message_id: "msg-1".to_string(),
            sequence: 1,
            content_hash: None,
        });

        tracker.add_range(TrackedRange {
            range: LineRange::new(100, Some(50)),
            message_id: "msg-2".to_string(),
            sequence: 2,
            content_hash: None,
        });

        assert!(!tracker.has_contiguous_coverage_from_start());

        let gaps = tracker.coverage_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].offset, 50);
        assert_eq!(gaps[0].limit, Some(50));
    }

    #[test]
    fn test_file_lifecycle_rename() {
        let mut tracker = FileLifecycleTracker::new();

        tracker.record_rename(FileRename {
            old_path: PathBuf::from("/old/path.rs"),
            new_path: PathBuf::from("/new/path.rs"),
            message_id: "msg-1".to_string(),
            sequence: 1,
        });

        let resolved = tracker.resolve_path(&PathBuf::from("/old/path.rs"));
        assert_eq!(resolved, PathBuf::from("/new/path.rs"));
    }

    #[test]
    fn test_file_lifecycle_chained_rename() {
        let mut tracker = FileLifecycleTracker::new();

        tracker.record_rename(FileRename {
            old_path: PathBuf::from("/a.rs"),
            new_path: PathBuf::from("/b.rs"),
            message_id: "msg-1".to_string(),
            sequence: 1,
        });

        tracker.record_rename(FileRename {
            old_path: PathBuf::from("/b.rs"),
            new_path: PathBuf::from("/c.rs"),
            message_id: "msg-2".to_string(),
            sequence: 2,
        });

        let resolved = tracker.resolve_path(&PathBuf::from("/a.rs"));
        assert_eq!(resolved, PathBuf::from("/c.rs"));
    }

    #[test]
    fn test_is_likely_binary() {
        assert!(!is_likely_binary("Hello, world!"));
        assert!(!is_likely_binary("fn main() { println!(\"test\"); }"));

        // Create content with null bytes
        let binary = "Hello\0World\0Binary\0Content";
        assert!(is_likely_binary(binary));
    }

    #[test]
    fn test_evaluate_file_normal() {
        let content = "fn main() {\n    println!(\"Hello\");\n}";
        let metadata = evaluate_file(
            content,
            BinaryFilePolicy::Exclude,
            &LargeFilePolicy::default(),
        );

        assert!(!metadata.is_binary);
        assert_eq!(metadata.recommended_action, FileAction::Include);
    }

    #[test]
    fn test_evaluate_file_large() {
        let content = "line\n".repeat(5000); // 5000 lines
        let metadata = evaluate_file(
            &content,
            BinaryFilePolicy::Exclude,
            &LargeFilePolicy {
                max_lines: 2000,
                truncate: true,
                ..Default::default()
            },
        );

        assert!(!metadata.is_binary);
        assert_eq!(
            metadata.recommended_action,
            FileAction::Truncate { max_lines: 2000 }
        );
    }

    #[test]
    fn test_fallback_strategy() {
        let strategy = determine_fallback("generic error", 5, 50);
        assert_eq!(strategy, FallbackStrategy::KeepAll);

        let strategy = determine_fallback("generic error", 5, 150);
        assert_eq!(
            strategy,
            FallbackStrategy::TruncateRecent { max_operations: 50 }
        );
    }
}
