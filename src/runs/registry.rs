//! Map from job `kind` to a registered [`JobHandler`].

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use super::handler::JobHandler;

/// Building errors for the [`HandlerRegistry`].
#[derive(Debug, Error)]
pub enum RegistryError {
    /// Two handlers registered with the same `kind`.
    #[error("duplicate handler kind: {0}")]
    Duplicate(String),
}

/// Append-only registry of handlers. Cheaply cloneable (`Arc` inside).
#[derive(Clone)]
pub struct HandlerRegistry {
    inner: Arc<HashMap<String, Arc<dyn JobHandler>>>,
}

impl HandlerRegistry {
    /// Build a registry from an iterator of handlers, rejecting duplicates.
    pub fn from_handlers<I>(handlers: I) -> Result<Self, RegistryError>
    where
        I: IntoIterator<Item = Arc<dyn JobHandler>>,
    {
        let mut map: HashMap<String, Arc<dyn JobHandler>> = HashMap::new();
        for h in handlers {
            let kind = h.kind().to_string();
            if map.contains_key(&kind) {
                return Err(RegistryError::Duplicate(kind));
            }
            map.insert(kind, h);
        }
        Ok(Self {
            inner: Arc::new(map),
        })
    }

    /// Look up by kind.
    pub fn get(&self, kind: &str) -> Option<Arc<dyn JobHandler>> {
        self.inner.get(kind).cloned()
    }

    /// Every registered kind.
    pub fn kinds(&self) -> Vec<String> {
        self.inner.keys().cloned().collect()
    }

    /// Per-kind concurrency cap, defaulting to `usize::MAX` when the handler
    /// imposes none. The store interprets `0` as "no cap" — this method maps
    /// `None` to `0` for that contract.
    pub fn max_concurrent(&self, kind: &str) -> usize {
        self.inner
            .get(kind)
            .and_then(|h| h.max_concurrent())
            .unwrap_or(0)
    }
}

/// Builder helper.
#[derive(Default)]
pub struct HandlerRegistryBuilder {
    handlers: Vec<Arc<dyn JobHandler>>,
}

impl HandlerRegistryBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one handler.
    pub fn register(mut self, handler: Arc<dyn JobHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    /// Finalise.
    pub fn build(self) -> Result<HandlerRegistry, RegistryError> {
        HandlerRegistry::from_handlers(self.handlers)
    }
}
