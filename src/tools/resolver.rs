//! File path resolver for fuzzy matching against known files
//!
//! This module provides functionality to resolve potentially incorrect file paths
//! against a set of known files discovered by Glob/Grep tools.

use std::cmp::Ordering;
use std::path::Path;
use std::sync::RwLock;

/// Maximum number of files to track for resolution
const MAX_TRACKED_FILES: usize = 5000;

/// Minimum similarity threshold for fuzzy matching (0.0 - 1.0)
const FUZZY_THRESHOLD: f64 = 0.6;

/// Entry tracking a known file path with recency
#[derive(Debug, Clone)]
pub struct KnownFile {
    /// Absolute path to the file
    pub path: String,
    /// Recency counter (higher = more recent)
    pub recency: u64,
}

/// Type of match used for path resolution
#[derive(Debug, Clone, PartialEq)]
pub enum MatchType {
    /// Exact match or path suffix match
    Exact,
    /// Partial path match (filename matches)
    PartialPath,
    /// Fuzzy match using edit distance on filename
    FuzzyFilename {
        /// Filename similarity score (0.0 - 1.0)
        similarity: f64,
    },
    /// Fuzzy match using edit distance on full path
    FuzzyPath {
        /// Filename similarity score (0.0 - 1.0)
        filename_similarity: f64,
        /// Full path similarity score (0.0 - 1.0)
        path_similarity: f64,
    },
}

impl MatchType {
    /// Get priority order (lower = higher priority)
    fn priority(&self) -> u8 {
        match self {
            MatchType::Exact => 0,
            MatchType::PartialPath => 1,
            MatchType::FuzzyPath { .. } => 2, // Full path fuzzy is preferred over filename-only
            MatchType::FuzzyFilename { .. } => 3,
        }
    }

    /// Get combined similarity score for fuzzy matches
    fn similarity(&self) -> f64 {
        match self {
            MatchType::FuzzyFilename { similarity } => *similarity,
            MatchType::FuzzyPath {
                filename_similarity,
                path_similarity,
            } => {
                // Weight: 60% filename, 40% path for combined score
                filename_similarity * 0.6 + path_similarity * 0.4
            }
            _ => 1.0,
        }
    }
}

/// Result of path resolution
#[derive(Debug, Clone)]
pub struct PathResolution {
    /// The resolved absolute path
    pub resolved_path: String,
    /// Original path that was provided
    pub original_path: String,
    /// Whether resolution was performed (false if exact match)
    pub was_resolved: bool,
    /// Match type used
    pub match_type: MatchType,
}

/// File path resolver using known files from Glob/Grep results
pub struct FileResolver {
    /// Known files indexed by recency
    known_files: RwLock<Vec<KnownFile>>,
    /// Counter for recency tracking
    recency_counter: RwLock<u64>,
}

impl Default for FileResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl FileResolver {
    /// Create a new file resolver
    pub fn new() -> Self {
        Self {
            known_files: RwLock::new(Vec::new()),
            recency_counter: RwLock::new(0),
        }
    }

    /// Register files, updating recency for existing ones
    pub fn register_files(&self, paths: &[String]) {
        let mut files = self.known_files.write().unwrap();
        let mut counter = self.recency_counter.write().unwrap();

        for path in paths {
            *counter += 1;

            // Update existing or add new
            if let Some(existing) = files.iter_mut().find(|f| &f.path == path) {
                existing.recency = *counter;
            } else {
                files.push(KnownFile {
                    path: path.clone(),
                    recency: *counter,
                });
            }
        }

        // Prune if too many files (keep most recent)
        if files.len() > MAX_TRACKED_FILES {
            files.sort_by(|a, b| b.recency.cmp(&a.recency));
            files.truncate(MAX_TRACKED_FILES);
        }
    }

    /// Resolve input path against known files
    pub fn resolve(&self, input: &str) -> Option<PathResolution> {
        let files = self.known_files.read().unwrap();
        if files.is_empty() {
            return None;
        }

        // Extract filename from input for matching
        let input_path = Path::new(input);
        let input_filename = input_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string());

        let mut candidates: Vec<(String, MatchType, u64)> = Vec::new();

        // Check if input contains directory components
        let input_has_dir = input.contains('/');

        for known in files.iter() {
            // 1. Exact match: full path match
            if known.path == input {
                candidates.push((known.path.clone(), MatchType::Exact, known.recency));
                continue;
            }

            // Suffix match (only if input has directory components)
            if input_has_dir && known.path.ends_with(&format!("/{}", input)) {
                candidates.push((known.path.clone(), MatchType::Exact, known.recency));
                continue;
            }

            let known_path = Path::new(&known.path);
            if let Some(ref input_fn) = input_filename {
                if let Some(known_fn) = known_path.file_name() {
                    let known_fn_str = known_fn.to_string_lossy();

                    // Calculate filename similarity
                    let filename_similarity = if known_fn_str == *input_fn {
                        1.0 // Exact filename match
                    } else {
                        strsim::normalized_damerau_levenshtein(input_fn, &known_fn_str)
                    };

                    // Skip if filename similarity is too low
                    if filename_similarity < FUZZY_THRESHOLD {
                        continue;
                    }

                    // 2. If input has directory components, use path-aware matching
                    if input_has_dir {
                        let path_similarity =
                            strsim::normalized_damerau_levenshtein(input, &known.path);

                        if filename_similarity == 1.0 {
                            // Exact filename match - use PartialPath but store path similarity for sorting
                            candidates.push((
                                known.path.clone(),
                                MatchType::FuzzyPath {
                                    filename_similarity,
                                    path_similarity,
                                },
                                known.recency,
                            ));
                        } else {
                            // Fuzzy filename match with path context
                            candidates.push((
                                known.path.clone(),
                                MatchType::FuzzyPath {
                                    filename_similarity,
                                    path_similarity,
                                },
                                known.recency,
                            ));
                        }
                    } else {
                        // No directory in input - simple matching
                        if filename_similarity == 1.0 {
                            candidates.push((
                                known.path.clone(),
                                MatchType::PartialPath,
                                known.recency,
                            ));
                        } else {
                            candidates.push((
                                known.path.clone(),
                                MatchType::FuzzyFilename {
                                    similarity: filename_similarity,
                                },
                                known.recency,
                            ));
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort by: match type priority, then (for fuzzy: similarity first, for others: recency first)
        candidates.sort_by(|a, b| {
            // First compare by match type priority
            let type_cmp = a.1.priority().cmp(&b.1.priority());
            if type_cmp != Ordering::Equal {
                return type_cmp;
            }

            // For fuzzy matches, prioritize similarity over recency
            // (the most similar file is more likely to be what was intended)
            if matches!(
                a.1,
                MatchType::FuzzyFilename { .. } | MatchType::FuzzyPath { .. }
            ) {
                let sim_cmp =
                    b.1.similarity()
                        .partial_cmp(&a.1.similarity())
                        .unwrap_or(Ordering::Equal);
                if sim_cmp != Ordering::Equal {
                    return sim_cmp;
                }
            }

            // For Exact/PartialPath, or fuzzy ties, prefer more recent
            b.2.cmp(&a.2)
        });

        let (resolved_path, match_type, _) = candidates.remove(0);

        Some(PathResolution {
            resolved_path,
            original_path: input.to_string(),
            was_resolved: match_type != MatchType::Exact
                || input != candidates.first().map(|c| c.0.as_str()).unwrap_or(input),
            match_type,
        })
    }

    /// Get count of known files (for testing/debugging)
    pub fn known_file_count(&self) -> usize {
        self.known_files.read().unwrap().len()
    }

    /// Clear all known files (useful for testing)
    pub fn clear(&self) {
        self.known_files.write().unwrap().clear();
        *self.recency_counter.write().unwrap() = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/src/main.rs".to_string(),
            "/workspace/src/lib.rs".to_string(),
        ]);

        let result = resolver.resolve("/workspace/src/main.rs").unwrap();
        assert_eq!(result.resolved_path, "/workspace/src/main.rs");
        assert_eq!(result.match_type, MatchType::Exact);
    }

    #[test]
    fn test_suffix_match() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/src/providers/mlxlm/client.rs".to_string(),
            "/workspace/src/tools/context.rs".to_string(),
        ]);

        let result = resolver.resolve("src/providers/mlxlm/client.rs").unwrap();
        assert_eq!(
            result.resolved_path,
            "/workspace/src/providers/mlxlm/client.rs"
        );
        assert_eq!(result.match_type, MatchType::Exact);
    }

    #[test]
    fn test_partial_path_match() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/src/providers/mlxlm/client.rs".to_string(),
            "/workspace/src/tools/context.rs".to_string(),
        ]);

        let result = resolver.resolve("client.rs").unwrap();
        assert_eq!(
            result.resolved_path,
            "/workspace/src/providers/mlxlm/client.rs"
        );
        assert_eq!(result.match_type, MatchType::PartialPath);
    }

    #[test]
    fn test_fuzzy_match_typo() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/src/providers/mlxlm/client.rs".to_string(),
            "/workspace/src/tools/context.rs".to_string(),
        ]);

        // Typo: "clent" instead of "client"
        let result = resolver.resolve("clent.rs").unwrap();
        assert_eq!(
            result.resolved_path,
            "/workspace/src/providers/mlxlm/client.rs"
        );
        assert!(matches!(result.match_type, MatchType::FuzzyFilename { .. }));
    }

    #[test]
    fn test_fuzzy_full_path_match() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/data/0x70e6D4ce7CF0a6835F51F6D5043E4BB5C8aC2d05/src/RockPaperScissors.sol"
                .to_string(),
            "/workspace/data/0xAABBCCDDEEFF/src/OtherContract.sol".to_string(),
        ]);

        // Typo in extension: .ol instead of .sol
        let result = resolver
            .resolve(
                "/workspace/data/0x70e6D4ce7CF0a6835F51F6D5043E4BB5C8aC2d05/src/RockPaperScissors.ol",
            )
            .unwrap();
        assert_eq!(
            result.resolved_path,
            "/workspace/data/0x70e6D4ce7CF0a6835F51F6D5043E4BB5C8aC2d05/src/RockPaperScissors.sol"
        );
        assert!(matches!(result.match_type, MatchType::FuzzyPath { .. }));
    }

    #[test]
    fn test_fuzzy_path_prefers_similar_directory() {
        let resolver = FileResolver::new();
        resolver.register_files(&[
            "/workspace/contracts/0xAABB/src/Game.sol".to_string(),
            "/workspace/contracts/0xCCDD/src/Game.sol".to_string(),
        ]);

        // Should prefer the path with more similar directory
        let result = resolver
            .resolve("/workspace/contracts/0xAABC/src/Game.sol") // typo in address: AABC vs AABB
            .unwrap();
        // Both have exact filename match, but 0xAABB is more similar to 0xAABC than 0xCCDD
        assert_eq!(
            result.resolved_path,
            "/workspace/contracts/0xAABB/src/Game.sol"
        );
    }

    #[test]
    fn test_recency_ordering() {
        let resolver = FileResolver::new();

        // Register files in order
        resolver.register_files(&["/workspace/src/old/client.rs".to_string()]);
        resolver.register_files(&["/workspace/src/new/client.rs".to_string()]);

        // Should prefer more recent file
        let result = resolver.resolve("client.rs").unwrap();
        assert_eq!(result.resolved_path, "/workspace/src/new/client.rs");
    }

    #[test]
    fn test_max_files_pruning() {
        let resolver = FileResolver::new();

        // Register more than MAX_TRACKED_FILES
        let paths: Vec<String> = (0..MAX_TRACKED_FILES + 100)
            .map(|i| format!("/workspace/file{}.rs", i))
            .collect();

        resolver.register_files(&paths);

        assert_eq!(resolver.known_file_count(), MAX_TRACKED_FILES);
    }

    #[test]
    fn test_no_match() {
        let resolver = FileResolver::new();
        resolver.register_files(&["/workspace/src/main.rs".to_string()]);

        let result = resolver.resolve("nonexistent.rs");
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_resolver() {
        let resolver = FileResolver::new();
        let result = resolver.resolve("anything.rs");
        assert!(result.is_none());
    }

    #[test]
    fn test_match_type_priority() {
        let resolver = FileResolver::new();

        // Register files that could match different ways
        resolver.register_files(&[
            "/workspace/exact/client.rs".to_string(), // Will be partial match for "client.rs"
            "/workspace/partial/client.rs".to_string(), // Will also be partial match
        ]);

        // Now register one that's an exact suffix match
        resolver.register_files(&["/workspace/src/client.rs".to_string()]);

        // Test that partial match picks most recent when multiple partial matches exist
        let result = resolver.resolve("client.rs").unwrap();
        assert_eq!(result.resolved_path, "/workspace/src/client.rs");
    }
}
