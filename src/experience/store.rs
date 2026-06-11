//! Filesystem-backed knowledge base for storing and retrieving lessons.

use super::types::{Lesson, LessonCategory, LessonQuery};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Filesystem-backed knowledge base.
///
/// Stores lessons as markdown files with YAML frontmatter in a directory tree:
/// ```text
/// {base_path}/knowledge/{domain}/{category}/{id}.md
/// ```
pub struct FileKnowledgeBase {
    base_path: PathBuf,
}

impl FileKnowledgeBase {
    /// Create a knowledge base rooted at the given path.
    ///
    /// The `knowledge/` subdirectory is appended automatically.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into().join("knowledge"),
        }
    }

    /// Store a lesson in the knowledge base.
    ///
    /// Creates the directory structure if it doesn't exist.
    pub fn store(&self, lesson: &Lesson) -> std::io::Result<PathBuf> {
        let dir = self
            .base_path
            .join(&lesson.domain)
            .join(lesson.category.dir_name());
        std::fs::create_dir_all(&dir)?;

        let filename = if lesson.category == LessonCategory::Contract {
            // Contract-specific: use address as filename
            let addr = lesson.contract_address.as_deref().unwrap_or(&lesson.id);
            format!("{}.md", sanitize_filename(addr))
        } else {
            format!("{}.md", sanitize_filename(&lesson.id))
        };

        let path = dir.join(&filename);
        let content = serialize_lesson(lesson);

        // For contract lessons, append if file exists (multiple lessons per contract)
        if lesson.category == LessonCategory::Contract && path.exists() {
            let existing = std::fs::read_to_string(&path)?;
            let separator = format!("\n\n---\n\n## {}\n\n", lesson.title);
            std::fs::write(
                &path,
                format!("{}{}{}", existing, separator, lesson.content),
            )?;
        } else {
            std::fs::write(&path, content)?;
        }

        info!(path = %path.display(), domain = %lesson.domain, category = ?lesson.category, "Stored lesson");
        Ok(path)
    }

    /// Retrieve lessons matching a query.
    ///
    /// Searches patterns/ (always), contracts/ (if address matches),
    /// and successes/ (always). Returns up to `query.limit` results
    /// sorted by relevance (tag match count).
    pub fn retrieve(&self, query: &LessonQuery) -> Vec<Lesson> {
        let mut results = Vec::new();
        let domain_path = self.base_path.join(&query.domain);

        if !domain_path.exists() {
            debug!(domain = %query.domain, "No knowledge base directory for domain");
            return results;
        }

        // Always search patterns/
        self.scan_directory(&domain_path.join("patterns"), query, &mut results);

        // Always search successes/
        self.scan_directory(&domain_path.join("successes"), query, &mut results);

        // Search contracts/ only if address specified
        if let Some(ref addr) = query.contract_address {
            let contract_file = domain_path
                .join("contracts")
                .join(format!("{}.md", sanitize_filename(addr)));
            if contract_file.exists() {
                if let Some(lesson) = self.load_lesson(&contract_file) {
                    results.push(lesson);
                }
            }
        }

        // Sort by relevance: count matching tags
        if !query.tags.is_empty() {
            results.sort_by(|a, b| {
                let score_a = a.tags.iter().filter(|t| query.tags.contains(t)).count();
                let score_b = b.tags.iter().filter(|t| query.tags.contains(t)).count();
                score_b.cmp(&score_a)
            });
        }

        results.truncate(query.limit);
        debug!(domain = %query.domain, count = results.len(), "Retrieved lessons");
        results
    }

    /// List all domains that have knowledge.
    pub fn domains(&self) -> Vec<String> {
        let mut domains = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        domains.push(name.to_string());
                    }
                }
            }
        }
        domains
    }

    /// Count total lessons in the knowledge base.
    pub fn count(&self) -> usize {
        let mut total = 0;
        for domain in self.domains() {
            let domain_path = self.base_path.join(&domain);
            for subdir in &["patterns", "contracts", "successes"] {
                let dir = domain_path.join(subdir);
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    total += entries
                        .flatten()
                        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
                        .count();
                }
            }
        }
        total
    }

    /// Format retrieved lessons as a markdown section for injection into agent context.
    pub fn format_for_injection(lessons: &[Lesson]) -> String {
        if lessons.is_empty() {
            return String::new();
        }

        let mut output = String::from("## Lessons from Past Experience\n\n");
        output.push_str("The following lessons were extracted from previous task executions. Apply them to avoid known pitfalls.\n\n");

        for (i, lesson) in lessons.iter().enumerate() {
            output.push_str(&format!("### {}. {}\n\n", i + 1, lesson.title));
            output.push_str(&lesson.content);
            output.push_str("\n\n");
        }

        output
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn scan_directory(&self, dir: &Path, _query: &LessonQuery, results: &mut Vec<Lesson>) {
        if !dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "md") {
                continue;
            }
            if let Some(lesson) = self.load_lesson(&path) {
                results.push(lesson);
            }
        }
    }

    fn load_lesson(&self, path: &Path) -> Option<Lesson> {
        let content = std::fs::read_to_string(path).ok()?;
        deserialize_lesson(&content).or_else(|| {
            // Fallback: treat entire file as content with filename as title
            let title = path.file_stem()?.to_str()?.to_string();
            Some(Lesson {
                id: title.clone(),
                title,
                content,
                domain: String::new(),
                category: LessonCategory::Pattern,
                tags: vec![],
                contract_address: None,
                source: None,
                outcome: None,
                created_at: None,
                triggers: None,
            })
        })
    }
}

// ── Serialization ───────────────────────────────────────────────────────

/// Serialize a lesson to markdown with YAML frontmatter.
fn serialize_lesson(lesson: &Lesson) -> String {
    let frontmatter = serde_yaml::to_string(lesson).unwrap_or_default();
    format!(
        "---\n{}---\n\n## {}\n\n{}\n",
        frontmatter, lesson.title, lesson.content
    )
}

/// Deserialize a lesson from markdown with YAML frontmatter.
fn deserialize_lesson(content: &str) -> Option<Lesson> {
    if !content.starts_with("---") {
        return None;
    }

    let end = content[3..].find("---")?;
    let yaml = &content[3..3 + end].trim();
    serde_yaml::from_str(yaml).ok()
}

/// Sanitize a string for use as a filename (replace non-alphanumeric with underscore).
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_store_and_retrieve_pattern() {
        let tmp = TempDir::new().unwrap();
        let kb = FileKnowledgeBase::new(tmp.path());

        let lesson = Lesson {
            id: "wrong_target".to_string(),
            title: "Always call TARGET, not attacker".to_string(),
            content: "Use TARGET.call{value}(...) not attackerAddr.call{value}(...)".to_string(),
            domain: "fork_test".to_string(),
            category: LessonCategory::Pattern,
            tags: vec!["call_target".to_string(), "payable".to_string()],
            contract_address: None,
            source: None,
            outcome: Some("INCONCLUSIVE".to_string()),
            created_at: None,
            triggers: None,
        };

        kb.store(&lesson).unwrap();

        let results = kb.retrieve(&LessonQuery::new("fork_test"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Always call TARGET, not attacker");
    }

    #[test]
    fn test_retrieve_with_tags() {
        let tmp = TempDir::new().unwrap();
        let kb = FileKnowledgeBase::new(tmp.path());

        let lesson1 = Lesson {
            id: "v3_swap".to_string(),
            title: "V3 swap needs non-zero price limit".to_string(),
            content: "sqrtPriceLimitX96 must be non-zero".to_string(),
            domain: "fork_test".to_string(),
            category: LessonCategory::Pattern,
            tags: vec!["v3".to_string(), "swap".to_string()],
            ..Default::default()
        };
        let lesson2 = Lesson {
            id: "v2_swap".to_string(),
            title: "V2 swap needs prior transfer".to_string(),
            content: "Transfer tokens before calling swap".to_string(),
            domain: "fork_test".to_string(),
            category: LessonCategory::Pattern,
            tags: vec!["v2".to_string(), "swap".to_string()],
            ..Default::default()
        };

        kb.store(&lesson1).unwrap();
        kb.store(&lesson2).unwrap();

        // Query with v3 tag should rank v3 lesson first
        let results = kb.retrieve(&LessonQuery::new("fork_test").with_tags(vec!["v3".to_string()]));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v3_swap");
    }

    #[test]
    fn test_contract_specific_lesson() {
        let tmp = TempDir::new().unwrap();
        let kb = FileKnowledgeBase::new(tmp.path());

        let lesson = Lesson {
            id: "contract_lesson".to_string(),
            title: "This pool uses custom reentrancy guard".to_string(),
            content: "storage_2 acts as lock".to_string(),
            domain: "fork_test".to_string(),
            category: LessonCategory::Contract,
            contract_address: Some("0x2D30c2B3".to_string()),
            ..Default::default()
        };

        kb.store(&lesson).unwrap();

        // Query without address shouldn't find it
        let results = kb.retrieve(&LessonQuery::new("fork_test"));
        assert_eq!(results.len(), 0);

        // Query with matching address should find it
        let results = kb.retrieve(&LessonQuery::new("fork_test").with_contract("0x2D30c2B3"));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_format_for_injection() {
        let lessons = vec![Lesson {
            id: "test".to_string(),
            title: "Use TARGET not attacker".to_string(),
            content: "Always call the target contract.".to_string(),
            ..Default::default()
        }];

        let formatted = FileKnowledgeBase::format_for_injection(&lessons);
        assert!(formatted.contains("Lessons from Past Experience"));
        assert!(formatted.contains("Use TARGET not attacker"));
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let lesson = Lesson {
            id: "test_rt".to_string(),
            title: "Roundtrip test".to_string(),
            content: "Some content".to_string(),
            domain: "test".to_string(),
            category: LessonCategory::Pattern,
            tags: vec!["a".to_string(), "b".to_string()],
            contract_address: None,
            source: Some("job_123".to_string()),
            outcome: Some("DISPROVED".to_string()),
            created_at: Some("2026-04-09T12:00:00Z".to_string()),
            triggers: None,
        };

        let serialized = serialize_lesson(&lesson);
        let deserialized = deserialize_lesson(&serialized).unwrap();
        assert_eq!(deserialized.id, "test_rt");
        assert_eq!(deserialized.title, "Roundtrip test");
        assert_eq!(deserialized.tags, vec!["a", "b"]);
        assert_eq!(deserialized.outcome, Some("DISPROVED".to_string()));
    }

    #[test]
    fn test_count_and_domains() {
        let tmp = TempDir::new().unwrap();
        let kb = FileKnowledgeBase::new(tmp.path());

        for (id, domain) in [("l1", "fork_test"), ("l2", "fork_test"), ("l3", "audit")] {
            kb.store(&Lesson {
                id: id.to_string(),
                title: format!("Lesson {}", id),
                content: "content".to_string(),
                domain: domain.to_string(),
                category: LessonCategory::Pattern,
                ..Default::default()
            })
            .unwrap();
        }

        assert_eq!(kb.count(), 3);
        let mut domains = kb.domains();
        domains.sort();
        assert_eq!(domains, vec!["audit", "fork_test"]);
    }
}
