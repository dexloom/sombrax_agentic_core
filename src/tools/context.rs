//! Tool execution context

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::message::Message;
use crate::tools::agent::TodoItem;
use crate::tools::error::ToolError;
use crate::tools::resolver::{FileResolver, PathResolution};

/// Execution context for tools
#[derive(Clone)]
pub struct ToolContext {
    inner: Arc<ToolContextInner>,
}

struct ToolContextInner {
    session_id: String,
    workspace_directory: PathBuf,
    todos: RwLock<HashMap<String, TodoItem>>,
    current_depth: usize,
    max_depth: usize,
    /// Initial messages to seed sub-agent conversation history.
    /// These are typically source code context or other pre-loaded content.
    initial_messages: Vec<Message>,
    /// File resolver for path resolution from Glob/Grep results
    file_resolver: FileResolver,
    /// Excluded directory/file patterns (can be modified at runtime).
    /// Patterns can be:
    /// - Exact directory names (e.g., "node_modules", ".git")
    /// - Path prefixes (e.g., "vendor/", "build/")
    /// - Glob patterns (e.g., "*.log", "**/__pycache__")
    exclude_patterns: RwLock<HashSet<String>>,
}

/// Default excluded patterns for common directories that should be ignored
const DEFAULT_EXCLUDES: &[&str] = &[
    // Version control
    ".git",
    ".hg",
    ".svn",
    // Dependencies
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    // Build artifacts
    "target",
    "build",
    "dist",
    ".next",
    // IDE/Editor
    ".idea",
    ".vscode",
    // Caches
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    // Solidity/Foundry test libraries
    "forge-std",
    // Audit artifacts (SombraX specific)
    "audit",
    ".clone.meta",
];

impl ToolContext {
    /// Create a new tool context with default exclude patterns
    pub fn new(session_id: String, workspace_directory: PathBuf) -> Self {
        let exclude_patterns: HashSet<String> =
            DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();

        Self {
            inner: Arc::new(ToolContextInner {
                session_id,
                workspace_directory,
                todos: RwLock::new(HashMap::new()),
                current_depth: 0,
                max_depth: 5,
                initial_messages: Vec::new(),
                file_resolver: FileResolver::new(),
                exclude_patterns: RwLock::new(exclude_patterns),
            }),
        }
    }

    /// Create a new tool context with no default excludes
    pub fn new_without_excludes(session_id: String, workspace_directory: PathBuf) -> Self {
        Self {
            inner: Arc::new(ToolContextInner {
                session_id,
                workspace_directory,
                todos: RwLock::new(HashMap::new()),
                current_depth: 0,
                max_depth: 5,
                initial_messages: Vec::new(),
                file_resolver: FileResolver::new(),
                exclude_patterns: RwLock::new(HashSet::new()),
            }),
        }
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Get the workspace directory
    pub fn workspace_directory(&self) -> &PathBuf {
        &self.inner.workspace_directory
    }

    /// Get current recursion depth
    pub fn current_depth(&self) -> usize {
        self.inner.current_depth
    }

    /// Get maximum recursion depth
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }

    /// Get initial messages for sub-agent spawning.
    ///
    /// These messages are automatically passed to child agents when TaskTool
    /// spawns subagents, preserving source code context.
    pub fn initial_messages(&self) -> &[Message] {
        &self.inner.initial_messages
    }

    /// Set initial messages for sub-agent spawning.
    ///
    /// These messages will be automatically passed to child agents when TaskTool
    /// spawns subagents, preserving the source code context.
    pub fn with_initial_messages(self, messages: Vec<Message>) -> Self {
        Self {
            inner: Arc::new(ToolContextInner {
                session_id: self.inner.session_id.clone(),
                workspace_directory: self.inner.workspace_directory.clone(),
                todos: RwLock::new(self.inner.todos.read().unwrap().clone()),
                current_depth: self.inner.current_depth,
                max_depth: self.inner.max_depth,
                initial_messages: messages,
                file_resolver: FileResolver::new(),
                exclude_patterns: RwLock::new(self.inner.exclude_patterns.read().unwrap().clone()),
            }),
        }
    }

    /// Create a child context with incremented depth
    pub fn child_context(&self, child_session_id: String) -> Result<Self, ToolError> {
        if self.inner.current_depth >= self.inner.max_depth {
            return Err(ToolError::MaxRecursionDepth);
        }

        Ok(Self {
            inner: Arc::new(ToolContextInner {
                session_id: child_session_id,
                workspace_directory: self.inner.workspace_directory.clone(),
                todos: RwLock::new(HashMap::new()),
                current_depth: self.inner.current_depth + 1,
                max_depth: self.inner.max_depth,
                initial_messages: self.inner.initial_messages.clone(),
                file_resolver: FileResolver::new(),
                exclude_patterns: RwLock::new(self.inner.exclude_patterns.read().unwrap().clone()),
            }),
        })
    }

    /// Validate that a path is within the workspace
    pub fn validate_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        // Try various path normalizations to find the correct file
        if let Some(normalized) = self.try_normalize_path(path) {
            return self.validate_normalized_path(&normalized);
        }

        // Fall back to standard path handling
        self.validate_normalized_path(path)
    }

    /// Try to normalize a potentially malformed path.
    ///
    /// Handles common issues:
    /// - Missing leading `/` on absolute paths (e.g., `Users/foo/...` -> `/Users/foo/...`)
    /// - Duplicated workspace folder name (e.g., `0xABC/src/...` when CWD is `0xABC/`)
    /// - Path that contains workspace path as suffix
    /// - Fuzzy filename matching for typos
    fn try_normalize_path(&self, path: &str) -> Option<String> {
        let workspace = &self.inner.workspace_directory;
        let workspace_str = workspace.to_string_lossy();

        // Helper to check if a path is within workspace
        let is_in_workspace = |p: &Path| -> bool {
            if let (Ok(canonical), Ok(ws_canonical)) = (p.canonicalize(), workspace.canonicalize())
            {
                canonical.starts_with(&ws_canonical)
            } else {
                false
            }
        };

        // Case 1: Path looks absolute but missing leading `/`
        // e.g., "Users/alice/project/..." -> "/Users/alice/project/..."
        if !path.starts_with('/') && path.contains('/') {
            let with_slash = format!("/{}", path);
            let with_slash_path = PathBuf::from(&with_slash);
            if with_slash_path.exists() && is_in_workspace(&with_slash_path) {
                return Some(with_slash);
            }
        }

        // Case 2: Path starts with a component that matches the workspace folder name
        // e.g., CWD is "/Users/foo/0xABC/" and path is "0xABC/src/file.sol"
        // Should resolve to "/Users/foo/0xABC/src/file.sol"
        if let Some(workspace_name) = workspace.file_name() {
            let workspace_name_str = workspace_name.to_string_lossy();
            let prefix_with_slash = format!("{}/", workspace_name_str);

            if path.starts_with(&prefix_with_slash) {
                // Remove the duplicated folder name
                let suffix = &path[prefix_with_slash.len()..];
                let normalized = workspace.join(suffix);
                if normalized.exists() {
                    return Some(normalized.to_string_lossy().to_string());
                }
            }
        }

        // Case 3: Path contains the workspace path somewhere in it
        // e.g., path is "some/prefix/Users/foo/workspace/src/file.sol"
        // and workspace is "/Users/foo/workspace"
        // Should find and use the suffix after workspace match
        if let Some(pos) = path.find(workspace_str.as_ref()) {
            let suffix_start = pos + workspace_str.len();
            if suffix_start < path.len() {
                let suffix = path[suffix_start..].trim_start_matches('/');
                let normalized = workspace.join(suffix);
                if normalized.exists() {
                    return Some(normalized.to_string_lossy().to_string());
                }
            }
        }

        // Case 4: Try to find workspace folder name anywhere in path and extract suffix
        if let Some(workspace_name) = workspace.file_name() {
            let workspace_name_str = workspace_name.to_string_lossy();
            let search_pattern = format!("/{}/", workspace_name_str);

            if let Some(pos) = path.find(&search_pattern) {
                let suffix_start = pos + search_pattern.len();
                if suffix_start < path.len() {
                    let suffix = &path[suffix_start..];
                    let normalized = workspace.join(suffix);
                    if normalized.exists() {
                        return Some(normalized.to_string_lossy().to_string());
                    }
                }
            }

            // Also try without leading slash in pattern
            let search_pattern_no_slash = format!("{}/", workspace_name_str);
            if path.starts_with(&search_pattern_no_slash) {
                let suffix = &path[search_pattern_no_slash.len()..];
                let normalized = workspace.join(suffix);
                if normalized.exists() {
                    return Some(normalized.to_string_lossy().to_string());
                }
            }
        }

        // Case 5: Fuzzy filename search - find files with similar names in workspace
        // This handles LLM-generated paths with typos
        if let Some(result) = self.fuzzy_find_file(path) {
            return Some(result);
        }

        None
    }

    /// Fuzzy search for a file in the workspace.
    ///
    /// Extracts the filename from the path and searches the workspace for files
    /// with the same or similar name. Uses directory hints from the original path
    /// to pick the best match when multiple candidates exist.
    fn fuzzy_find_file(&self, path: &str) -> Option<String> {
        let input_path = Path::new(path);
        let filename = input_path.file_name()?.to_string_lossy();

        // Skip if filename is too short (likely not useful)
        if filename.len() < 3 {
            return None;
        }

        // Extract directory hints from the input path for ranking matches
        let dir_hints: Vec<String> = input_path
            .components()
            .filter_map(|c| {
                if let std::path::Component::Normal(s) = c {
                    Some(s.to_string_lossy().to_lowercase())
                } else {
                    None
                }
            })
            .collect();

        // Search workspace for matching files
        let candidates = self.search_files_by_name(&filename, &dir_hints);

        match candidates.len() {
            0 => None,
            1 => Some(candidates[0].clone()),
            _ => {
                // Multiple matches - pick the best one based on path similarity
                self.pick_best_match(&candidates, path)
            }
        }
    }

    /// Search for files matching the given filename in the workspace.
    ///
    /// Returns files with exact name match, or fuzzy matches if no exact match found.
    fn search_files_by_name(&self, filename: &str, dir_hints: &[String]) -> Vec<String> {
        let workspace = &self.inner.workspace_directory;
        let mut exact_matches = Vec::new();
        let mut fuzzy_matches: Vec<(String, f64)> = Vec::new();

        // Extract extension from input filename for exact matching
        let input_ext = Path::new(filename)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase());

        // Extract stem (filename without extension) for fuzzy matching
        let input_stem = Path::new(filename)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string());

        // Walk the workspace directory (limited depth to avoid performance issues)
        let max_depth = 10;
        if let Ok(entries) = Self::walk_dir_limited(workspace, max_depth, &self.get_excludes()) {
            for entry in entries {
                if let Some(entry_name) = entry.file_name() {
                    let entry_name_str = entry_name.to_string_lossy();

                    // Exact filename match
                    if entry_name_str == filename {
                        exact_matches.push(entry.to_string_lossy().to_string());
                        continue;
                    }

                    // For fuzzy matching, require exact extension match
                    let entry_ext = entry
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase());

                    if input_ext != entry_ext {
                        continue; // Skip if extensions don't match
                    }

                    // Get the stem for fuzzy comparison
                    let entry_stem = entry.file_stem().map(|s| s.to_string_lossy().to_string());

                    if let (Some(ref in_stem), Some(ref ent_stem)) = (&input_stem, &entry_stem) {
                        // Fuzzy match on stem only (not extension)
                        let similarity = strsim::normalized_damerau_levenshtein(in_stem, ent_stem);

                        // High threshold (0.85) to avoid false positives like file1/file2
                        // Require at least 5 char stem for fuzzy matching
                        if similarity >= 0.85 && in_stem.len() >= 5 {
                            fuzzy_matches.push((entry.to_string_lossy().to_string(), similarity));
                        }
                    }
                }
            }
        }

        // If we have exact matches, filter by directory hints
        if !exact_matches.is_empty() {
            if exact_matches.len() == 1 {
                return exact_matches;
            }

            // Score matches by directory hint overlap
            let mut scored: Vec<(String, usize)> = exact_matches
                .into_iter()
                .map(|p| {
                    let score = Self::score_path_by_hints(&p, dir_hints);
                    (p, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.cmp(&a.1));
            return vec![scored[0].0.clone()];
        }

        // Return fuzzy matches sorted by similarity
        if !fuzzy_matches.is_empty() {
            fuzzy_matches
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return vec![fuzzy_matches[0].0.clone()];
        }

        Vec::new()
    }

    /// Walk directory with depth limit, respecting excludes.
    fn walk_dir_limited(
        dir: &Path,
        max_depth: usize,
        excludes: &[String],
    ) -> std::io::Result<Vec<PathBuf>> {
        let mut results = Vec::new();
        Self::walk_dir_recursive(dir, 0, max_depth, excludes, &mut results)?;
        Ok(results)
    }

    fn walk_dir_recursive(
        dir: &Path,
        current_depth: usize,
        max_depth: usize,
        excludes: &[String],
        results: &mut Vec<PathBuf>,
    ) -> std::io::Result<()> {
        if current_depth > max_depth {
            return Ok(());
        }

        // Limit total results to avoid memory issues
        if results.len() > 10000 {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if this path component is excluded
            if let Some(name) = path.file_name() {
                let name_str = name.to_string_lossy();
                if excludes.iter().any(|e| e == name_str.as_ref()) {
                    continue;
                }
            }

            if path.is_file() {
                results.push(path);
            } else if path.is_dir() {
                Self::walk_dir_recursive(&path, current_depth + 1, max_depth, excludes, results)?;
            }
        }

        Ok(())
    }

    /// Score a path based on how many directory hints it contains.
    fn score_path_by_hints(path: &str, hints: &[String]) -> usize {
        let path_lower = path.to_lowercase();
        hints
            .iter()
            .filter(|hint| path_lower.contains(hint.as_str()))
            .count()
    }

    /// Pick the best match from multiple candidates based on path similarity.
    fn pick_best_match(&self, candidates: &[String], original_path: &str) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }

        // Score each candidate by Levenshtein distance to original path
        let mut scored: Vec<(&String, f64)> = candidates
            .iter()
            .map(|c| {
                let similarity = strsim::normalized_damerau_levenshtein(original_path, c);
                (c, similarity)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(scored[0].0.clone())
    }

    /// Validate a normalized path (internal helper)
    fn validate_normalized_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        let path = PathBuf::from(path);

        // Make path absolute if relative
        let absolute_path = if path.is_absolute() {
            path
        } else {
            self.inner.workspace_directory.join(path)
        };

        // Canonicalize to resolve symlinks and ..
        let canonical = absolute_path.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                // For new files, check parent directory
                if let Some(parent) = absolute_path.parent() {
                    if parent.exists() {
                        // Parent exists, path is valid for creation
                        return ToolError::FileNotFound(absolute_path.display().to_string());
                    }
                }
                ToolError::FileNotFound(absolute_path.display().to_string())
            } else {
                ToolError::Io(e)
            }
        })?;

        // Check if path is within workspace
        let workspace_canonical = self
            .inner
            .workspace_directory
            .canonicalize()
            .map_err(ToolError::Io)?;

        if !canonical.starts_with(&workspace_canonical) {
            return Err(ToolError::PathOutsideWorkspace(
                canonical.display().to_string(),
            ));
        }

        Ok(canonical)
    }

    /// Validate path for write operations (allows non-existent files)
    pub fn validate_write_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        let path = PathBuf::from(path);

        // Make path absolute if relative
        let absolute_path = if path.is_absolute() {
            path
        } else {
            self.inner.workspace_directory.join(path)
        };

        // For write operations, check if parent directory is valid
        let parent = absolute_path.parent().ok_or_else(|| {
            ToolError::Validation("Invalid path: no parent directory".to_string())
        })?;

        let parent_canonical = parent
            .canonicalize()
            .map_err(|_| ToolError::FileNotFound(parent.display().to_string()))?;

        // Check if parent is within workspace
        let workspace_canonical = self
            .inner
            .workspace_directory
            .canonicalize()
            .map_err(ToolError::Io)?;

        if !parent_canonical.starts_with(&workspace_canonical) {
            return Err(ToolError::PathOutsideWorkspace(
                absolute_path.display().to_string(),
            ));
        }

        Ok(absolute_path)
    }

    /// Get all todos for this session (sorted by ID)
    pub fn get_todos(&self) -> Vec<TodoItem> {
        let mut todos: Vec<TodoItem> = self.inner.todos.read().unwrap().values().cloned().collect();
        todos.sort_by(|a, b| a.id.cmp(&b.id));
        todos
    }

    /// Set todos for this session (replaces all)
    pub fn set_todos(&self, todos: Vec<TodoItem>) {
        let mut todo_map = self.inner.todos.write().unwrap();
        todo_map.clear();
        for todo in todos {
            todo_map.insert(todo.id.clone(), todo);
        }
    }

    /// Register files from Glob/Grep results for path resolution
    pub fn register_known_files(&self, paths: &[String]) {
        self.inner.file_resolver.register_files(paths);
    }

    /// Resolve a path against known files from previous Glob/Grep results
    pub fn resolve_path(&self, input: &str) -> Option<PathResolution> {
        self.inner.file_resolver.resolve(input)
    }

    /// Validate path with resolution fallback (for read/edit operations)
    ///
    /// First tries direct path validation. If the file is not found,
    /// attempts to resolve against known files from Glob/Grep results.
    /// Ensures the resolved file actually exists before returning.
    ///
    /// Returns the validated path and optional resolution info.
    pub fn validate_path_with_resolution(
        &self,
        path: &str,
    ) -> Result<(PathBuf, Option<PathResolution>), ToolError> {
        // 1. Try direct validation first
        match self.validate_path(path) {
            Ok(p) => {
                // Verify file exists
                if !p.exists() {
                    return Err(ToolError::FileNotFound(p.display().to_string()));
                }
                return Ok((p, None));
            }
            Err(ToolError::FileNotFound(_)) => {
                // 2. Try resolution
                if let Some(resolution) = self.resolve_path(path) {
                    let resolved = self.validate_path(&resolution.resolved_path)?;
                    // Verify resolved file exists
                    if !resolved.exists() {
                        return Err(ToolError::FileNotFound(resolved.display().to_string()));
                    }
                    return Ok((resolved, Some(resolution)));
                }
            }
            Err(e) => return Err(e),
        }
        Err(ToolError::FileNotFound(path.to_string()))
    }

    /// Validate write path with resolution fallback (for write operations)
    ///
    /// First tries direct write path validation. If the parent directory is not found,
    /// attempts to resolve against known files from Glob/Grep results.
    /// Ensures the parent directory of the resolved path exists.
    ///
    /// Returns the validated path and optional resolution info.
    pub fn validate_write_path_with_resolution(
        &self,
        path: &str,
    ) -> Result<(PathBuf, Option<PathResolution>), ToolError> {
        // 1. Try direct write path validation first
        match self.validate_write_path(path) {
            Ok(p) => return Ok((p, None)),
            Err(ToolError::FileNotFound(_)) => {
                // 2. Try resolution for existing file that might have a typo
                if let Some(resolution) = self.resolve_path(path) {
                    let resolved = self.validate_write_path(&resolution.resolved_path)?;
                    // Verify parent directory exists
                    if let Some(parent) = resolved.parent() {
                        if !parent.exists() {
                            return Err(ToolError::FileNotFound(format!(
                                "Parent directory does not exist: {}",
                                parent.display()
                            )));
                        }
                    }
                    return Ok((resolved, Some(resolution)));
                }
            }
            Err(e) => return Err(e),
        }
        Err(ToolError::FileNotFound(path.to_string()))
    }

    /// Get count of known files (for testing/debugging)
    pub fn known_file_count(&self) -> usize {
        self.inner.file_resolver.known_file_count()
    }

    // =========================================================================
    // Exclude Pattern Management
    // =========================================================================

    /// Add an exclude pattern at runtime.
    ///
    /// Patterns can be:
    /// - Directory names: "node_modules", ".git"
    /// - Path segments: "vendor/", "build/"
    /// - File extensions: "*.log"
    ///
    /// # Example
    ///
    /// ```ignore
    /// context.add_exclude("secrets");
    /// context.add_exclude("*.tmp");
    /// ```
    pub fn add_exclude(&self, pattern: impl Into<String>) {
        let mut excludes = self.inner.exclude_patterns.write().unwrap();
        excludes.insert(pattern.into());
    }

    /// Add multiple exclude patterns at runtime.
    pub fn add_excludes<I, S>(&self, patterns: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut excludes = self.inner.exclude_patterns.write().unwrap();
        for pattern in patterns {
            excludes.insert(pattern.into());
        }
    }

    /// Remove an exclude pattern.
    pub fn remove_exclude(&self, pattern: &str) -> bool {
        let mut excludes = self.inner.exclude_patterns.write().unwrap();
        excludes.remove(pattern)
    }

    /// Clear all exclude patterns (including defaults).
    pub fn clear_excludes(&self) {
        let mut excludes = self.inner.exclude_patterns.write().unwrap();
        excludes.clear();
    }

    /// Reset excludes to default patterns.
    pub fn reset_excludes(&self) {
        let mut excludes = self.inner.exclude_patterns.write().unwrap();
        excludes.clear();
        for pattern in DEFAULT_EXCLUDES {
            excludes.insert(pattern.to_string());
        }
    }

    /// Get current exclude patterns.
    pub fn get_excludes(&self) -> Vec<String> {
        let excludes = self.inner.exclude_patterns.read().unwrap();
        excludes.iter().cloned().collect()
    }

    /// Check if a path matches any exclude pattern.
    ///
    /// Returns `true` if the path should be excluded.
    pub fn is_excluded(&self, path: &Path) -> bool {
        let excludes = self.inner.exclude_patterns.read().unwrap();

        // Check each component of the path against patterns
        for component in path.components() {
            if let std::path::Component::Normal(name) = component {
                let name_str = name.to_string_lossy();

                // Check exact match
                if excludes.contains(name_str.as_ref()) {
                    return true;
                }

                // Check glob patterns (simple wildcards)
                for pattern in excludes.iter() {
                    if pattern.contains('*') && Self::matches_glob_pattern(&name_str, pattern) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Simple glob pattern matching (supports * and ?)
    fn matches_glob_pattern(name: &str, pattern: &str) -> bool {
        // Handle common patterns efficiently
        if let Some(ext) = pattern.strip_prefix("*.") {
            // Extension pattern like "*.log"
            return name.ends_with(&format!(".{}", ext));
        }

        if pattern.ends_with("/*") || pattern.ends_with("/**") {
            // Directory prefix pattern
            let prefix = pattern.trim_end_matches("/*").trim_end_matches("/**");
            return name.starts_with(prefix);
        }

        // Simple * wildcard (matches any sequence)
        if pattern.contains('*') && !pattern.contains("**") {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                return name.starts_with(parts[0]) && name.ends_with(parts[1]);
            }
        }

        // Fall back to exact match
        name == pattern
    }

    /// Filter a list of paths, removing excluded ones.
    ///
    /// This is useful for glob/grep results.
    pub fn filter_excluded(&self, paths: Vec<String>) -> Vec<String> {
        paths
            .into_iter()
            .filter(|p| !self.is_excluded(Path::new(p)))
            .collect()
    }
}
