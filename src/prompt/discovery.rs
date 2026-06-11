//! Discovery and resolution of [`SystemPrompt`] assets.
//!
//! Resolution follows a dash-only hash-strip ladder (see [`candidate_names`]):
//! for request `N`, probe `N`, then `N` with dots rewritten to underscores,
//! then strip the final `-`-separated segment and repeat. Search paths are
//! scanned in order; later entries override earlier entries with the same
//! name (matches [`sombrax_agentic_core::skill::SkillRegistry::discover`] semantics).

use super::{PromptError, SystemPrompt};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, warn};

use super::PromptMetadata;

/// Registry of prompts discovered across one or more search paths.
#[derive(Debug, Clone, Default)]
pub struct PromptRegistry {
    prompts: BTreeMap<String, Arc<SystemPrompt>>,
}

impl PromptRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            prompts: BTreeMap::new(),
        }
    }

    /// Scan `search_paths` in order. Files matching `*.md` at each path's top
    /// level become prompts. Last occurrence of a given name wins.
    pub async fn discover(search_paths: Vec<PathBuf>) -> Result<Self, PromptError> {
        let mut registry = Self::new();

        for path in search_paths {
            if !path.exists() {
                debug!("prompt search path does not exist: {:?}", path);
                continue;
            }
            match discover_in_directory(path.clone()).await {
                Ok(prompts) => {
                    debug!("discovered {} prompts in {:?}", prompts.len(), path);
                    for p in prompts {
                        registry.register(p);
                    }
                }
                Err(e) => {
                    warn!("failed to discover prompts in {:?}: {}", path, e);
                }
            }
        }

        Ok(registry)
    }

    /// Register a prompt. Overrides any prompt with the same name.
    pub fn register(&mut self, prompt: SystemPrompt) {
        self.prompts
            .insert(prompt.name().to_string(), Arc::new(prompt));
    }

    /// Exact name lookup. No ladder fallback.
    pub fn get(&self, name: &str) -> Option<Arc<SystemPrompt>> {
        self.prompts.get(name).cloned()
    }

    /// Hierarchical lookup. Returns `None` if every rung of the ladder misses.
    pub fn resolve(&self, request: &str) -> Option<Arc<SystemPrompt>> {
        for candidate in candidate_names(request) {
            if let Some(p) = self.prompts.get(&candidate) {
                return Some(p.clone());
            }
        }
        None
    }

    /// All metadata, sorted by name.
    pub fn all_metadata(&self) -> Vec<PromptMetadata> {
        self.prompts
            .values()
            .map(|p| p.metadata().clone())
            .collect()
    }

    /// All prompt names, sorted.
    pub fn prompt_names(&self) -> Vec<String> {
        self.prompts.keys().cloned().collect()
    }

    /// Number of registered prompts.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// Produce the ladder of candidate names for a request.
///
/// For each current name, emit the exact name, then (if it contains `.`) the
/// same name with dots rewritten to `_`. Then strip the final `-`-separated
/// segment and repeat, until no `-` remains.
pub(crate) fn candidate_names(request: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = request.to_string();
    loop {
        out.push(current.clone());
        if current.contains('.') {
            out.push(current.replace('.', "_"));
        }
        match current.rsplit_once('-') {
            Some((head, _)) if !head.is_empty() => current = head.to_string(),
            _ => break,
        }
    }
    out
}

/// Discover prompts in a directory (non-recursive, top-level `.md` only).
async fn discover_in_directory(path: PathBuf) -> Result<Vec<SystemPrompt>, PromptError> {
    if !path.is_dir() {
        return Ok(Vec::new());
    }

    let mut prompts = Vec::new();
    let mut entries = fs::read_dir(&path)
        .await
        .map_err(|e| PromptError::io(path.clone(), e))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| PromptError::io(path.clone(), e))?
    {
        let entry_path = entry.path();
        if !entry_path.is_file() {
            continue;
        }
        if entry_path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        match SystemPrompt::from_path(entry_path.clone()).await {
            Ok(prompt) => {
                debug!("discovered prompt '{}' at {:?}", prompt.name(), entry_path);
                prompts.push(prompt);
            }
            Err(e) => {
                warn!("skipping prompt {:?}: {}", entry_path, e);
            }
        }
    }

    Ok(prompts)
}

/// Default prompt search paths (not sombrax-specific).
///
/// Returns `./.sac/prompts`, `~/.sac/prompts`, `~/.sombra/prompts` — mirrors
/// [`sombrax_agentic_core::skill::default_search_paths`].
pub fn default_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(current_dir) = std::env::current_dir() {
        paths.push(current_dir.join(".sac").join("prompts"));
    }

    if let Some(home) = home_dir() {
        paths.push(home.join(".sac").join("prompts"));
        paths.push(home.join(".sombra").join("prompts"));
    }

    paths
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .and_then(|h| if h.is_empty() { None } else { Some(h) })
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_exact_only() {
        assert_eq!(candidate_names("whitehut"), vec!["whitehut".to_string()]);
    }

    #[test]
    fn ladder_multi_strip() {
        let got = candidate_names("whitehut-glm-5.1");
        assert_eq!(
            got,
            vec![
                "whitehut-glm-5.1".to_string(),
                "whitehut-glm-5_1".to_string(),
                "whitehut-glm".to_string(),
                "whitehut".to_string(),
            ]
        );
    }

    #[test]
    fn ladder_dot_to_underscore_skipped_when_no_dot() {
        let got = candidate_names("whitehut-glm-5_1");
        assert_eq!(
            got,
            vec![
                "whitehut-glm-5_1".to_string(),
                "whitehut-glm".to_string(),
                "whitehut".to_string(),
            ]
        );
    }

    #[test]
    fn ladder_single_segment_with_dot() {
        let got = candidate_names("foo.bar");
        assert_eq!(got, vec!["foo.bar".to_string(), "foo_bar".to_string()]);
    }
}
