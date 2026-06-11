use super::{Skill, SkillError, SkillMetadata};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, warn};

/// Registry for discovered skills
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    /// Skills indexed by name (sorted)
    skills: BTreeMap<String, Arc<Skill>>,
}

impl SkillRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            skills: BTreeMap::new(),
        }
    }

    /// Discover skills from multiple search paths
    ///
    /// Paths are searched in order. If the same skill name appears in multiple
    /// paths, the last occurrence wins (deduplication).
    pub async fn discover(search_paths: Vec<PathBuf>) -> Result<Self, SkillError> {
        let mut registry = Self::new();

        for search_path in search_paths {
            // Skip if path doesn't exist
            if !search_path.exists() {
                debug!("Skill search path does not exist: {:?}", search_path);
                continue;
            }

            // Discover skills in this path
            match discover_in_directory(search_path.clone()).await {
                Ok(skills) => {
                    debug!("Discovered {} skills in {:?}", skills.len(), search_path);
                    for skill in skills {
                        registry.register(skill);
                    }
                }
                Err(e) => {
                    warn!("Failed to discover skills in {:?}: {}", search_path, e);
                }
            }
        }

        Ok(registry)
    }

    /// Register a skill in the registry
    ///
    /// If a skill with the same name already exists, it will be replaced
    pub fn register(&mut self, skill: Skill) {
        let name = skill.name().to_string();
        self.skills.insert(name, Arc::new(skill));
    }

    /// Get all skill metadata for system prompt (Level 1)
    pub fn all_metadata(&self) -> Vec<SkillMetadata> {
        self.skills
            .values()
            .map(|skill| skill.metadata().clone())
            .collect()
    }

    /// Find skill by name
    pub fn get(&self, name: &str) -> Option<Arc<Skill>> {
        self.skills.get(name).cloned()
    }

    /// Find skills matching the given request
    ///
    /// Uses basic keyword matching against skill descriptions
    pub fn find_matching(&self, request: &str) -> Vec<Arc<Skill>> {
        self.skills
            .values()
            .filter(|skill| skill.matches(request))
            .cloned()
            .collect()
    }

    /// Get number of registered skills
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Get all skill names
    pub fn skill_names(&self) -> Vec<String> {
        self.skills.keys().cloned().collect()
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Discover all skills in a directory (recursive)
fn discover_in_directory(
    path: PathBuf,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Skill>, SkillError>> + Send>> {
    Box::pin(async move {
        let mut skills = Vec::new();

        // Check if path is a directory
        if !path.is_dir() {
            return Ok(skills);
        }

        // Read directory entries
        let mut entries = fs::read_dir(&path)
            .await
            .map_err(|e| SkillError::read_error(path.clone(), e))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| SkillError::read_error(path.clone(), e))?
        {
            let entry_path = entry.path();

            // Check if this directory contains SKILL.md
            if entry_path.is_dir() {
                let skill_file = entry_path.join("SKILL.md");
                if skill_file.exists() {
                    // Try to load skill
                    match Skill::from_path(entry_path.clone()).await {
                        Ok(skill) => {
                            debug!("Discovered skill '{}' at {:?}", skill.name(), entry_path);
                            skills.push(skill);
                        }
                        Err(e) => {
                            warn!("Failed to load skill from {:?}: {}", entry_path, e);
                        }
                    }
                } else {
                    // Recursively search subdirectories
                    match discover_in_directory(entry_path.clone()).await {
                        Ok(subdir_skills) => {
                            skills.extend(subdir_skills);
                        }
                        Err(e) => {
                            warn!("Failed to search {:?}: {}", entry_path, e);
                        }
                    }
                }
            }
        }

        Ok(skills)
    })
}

/// Get default skill search paths
pub fn default_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Project-local: ./.sac/skills/
    if let Ok(current_dir) = std::env::current_dir() {
        paths.push(current_dir.join(".sac").join("skills"));
    }

    // User-global SAC: ~/.sac/skills/
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".sac").join("skills"));
    }

    // User-global SombraX: ~/.sombra/skills/
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".sombra").join("skills"));
    }

    paths
}

/// Simple implementation of home_dir since it's not in std
mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        std::env::var_os("HOME")
            .and_then(|h| if h.is_empty() { None } else { Some(h) })
            .map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = SkillRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_default_search_paths() {
        let paths = default_search_paths();
        assert!(!paths.is_empty());
        // Should have at least project-local path
        assert!(paths.iter().any(|p| p.ends_with(".sac/skills")));
    }
}
