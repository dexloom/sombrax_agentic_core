//! Agent registry for cross-agent communication
//!
//! Provides the AgentRegistry for agent discovery and invocation.

use crate::context::SharedContext;
use crate::error::RegistryError;
use crate::message::Message;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Request sent to an agent via the registry
#[derive(Debug, Clone)]
pub struct AgentRequest {
    /// The message/prompt to send
    pub message: String,

    /// Optional conversation history
    pub history: Vec<Message>,

    /// Metadata about the invoking agent
    pub invoker: Option<String>,
}

impl AgentRequest {
    /// Create a new agent request with just a message
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            history: Vec::new(),
            invoker: None,
        }
    }

    /// Set the conversation history
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    /// Set the invoker
    pub fn with_invoker(mut self, invoker: impl Into<String>) -> Self {
        self.invoker = Some(invoker.into());
        self
    }
}

/// Response from an agent invocation
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// The agent's response content
    pub content: String,

    /// The responding agent's name
    pub agent_name: String,
}

/// Handle for invoking an agent (FR-010)
///
/// Agents implement this trait to be invocable through the registry.
pub trait AgentHandle: Send + Sync {
    /// Returns the agent's name
    fn name(&self) -> &str;

    /// Returns the agent's advertised capabilities
    fn capabilities(&self) -> &[String];

    /// Invoke the agent with a request
    fn invoke<'a>(
        &'a self,
        request: AgentRequest,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse, RegistryError>> + Send + 'a>>;
}

/// Agent registry for cross-agent discovery and communication (FR-009)
///
/// # Example
///
/// ```ignore
/// let registry = AgentRegistry::new(10); // max depth 10
///
/// // Register agents
/// registry.register(agent_a).await?;
/// registry.register(agent_b).await?;
///
/// // Discover by capability
/// let analysts = registry.discover("analysis").await;
///
/// // Invoke an agent
/// let response = registry.invoke("agent_b", AgentRequest::new("Analyze this")).await?;
///
/// // Access shared context
/// {
///     let mut ctx = registry.shared_context().write().await;
///     ctx.set("last_result", &response.content)?;
/// }
/// ```
pub struct AgentRegistry {
    /// Registered agents by name
    agents: RwLock<HashMap<String, Arc<dyn AgentHandle>>>,

    /// Session-scoped shared state
    shared_context: Arc<RwLock<SharedContext>>,

    /// Current invocation depth (for cycle detection)
    invocation_depth: AtomicUsize,

    /// Maximum allowed invocation depth (FR-012)
    max_depth: usize,
}

impl AgentRegistry {
    /// Create a new registry with the specified max invocation depth
    pub fn new(max_depth: usize) -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            shared_context: Arc::new(RwLock::new(SharedContext::new())),
            invocation_depth: AtomicUsize::new(0),
            max_depth,
        }
    }

    /// Register an agent (FR-009)
    pub async fn register(&self, agent: Arc<dyn AgentHandle>) -> Result<(), RegistryError> {
        let name = agent.name().to_string();
        let mut agents = self.agents.write().await;

        if agents.contains_key(&name) {
            return Err(RegistryError::AlreadyRegistered(name));
        }

        tracing::info!(agent_name = %name, "agent registered");
        agents.insert(name, agent);
        Ok(())
    }

    /// Unregister an agent
    pub async fn unregister(&self, name: &str) -> Result<(), RegistryError> {
        let mut agents = self.agents.write().await;
        agents
            .remove(name)
            .ok_or_else(|| RegistryError::NotFound(name.to_string()))?;

        tracing::info!(agent_name = %name, "agent unregistered");
        Ok(())
    }

    /// Get an agent by name
    pub async fn get(&self, name: &str) -> Option<Arc<dyn AgentHandle>> {
        self.agents.read().await.get(name).cloned()
    }

    /// Discover agents by capability (FR-009)
    pub async fn discover(&self, capability: &str) -> Vec<Arc<dyn AgentHandle>> {
        self.agents
            .read()
            .await
            .values()
            .filter(|a| a.capabilities().contains(&capability.to_string()))
            .cloned()
            .collect()
    }

    /// Invoke an agent by name (FR-010)
    ///
    /// Includes cycle detection (FR-012) via depth tracking.
    pub async fn invoke(
        &self,
        name: &str,
        request: AgentRequest,
    ) -> Result<AgentResponse, RegistryError> {
        // Check depth before invocation
        let current_depth = self.invocation_depth.fetch_add(1, Ordering::SeqCst);
        if current_depth >= self.max_depth {
            self.invocation_depth.fetch_sub(1, Ordering::SeqCst);
            tracing::warn!(
                agent_name = %name,
                max_depth = %self.max_depth,
                "max invocation depth exceeded"
            );
            return Err(RegistryError::MaxDepthExceeded(self.max_depth));
        }

        let _span = tracing::info_span!(
            "agent_invocation",
            agent = %name,
            depth = %current_depth,
            invoker = ?request.invoker
        )
        .entered();

        // Get agent and invoke
        let agent = self
            .get(name)
            .await
            .ok_or_else(|| RegistryError::NotFound(name.to_string()))?;

        let result = agent.invoke(request).await;

        // Decrement depth after invocation
        self.invocation_depth.fetch_sub(1, Ordering::SeqCst);

        result
    }

    /// Get the shared context (FR-011)
    pub fn shared_context(&self) -> Arc<RwLock<SharedContext>> {
        self.shared_context.clone()
    }

    /// List all registered agent names
    pub async fn list(&self) -> Vec<String> {
        self.agents.read().await.keys().cloned().collect()
    }

    /// Check if an agent is registered
    pub async fn has(&self, name: &str) -> bool {
        self.agents.read().await.contains_key(name)
    }

    /// Get the number of registered agents
    pub async fn len(&self) -> usize {
        self.agents.read().await.len()
    }

    /// Check if the registry is empty
    pub async fn is_empty(&self) -> bool {
        self.agents.read().await.is_empty()
    }

    /// Get the current invocation depth
    pub fn current_depth(&self) -> usize {
        self.invocation_depth.load(Ordering::SeqCst)
    }

    /// Get the maximum invocation depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new(10) // Default max depth of 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAgent {
        name: String,
        capabilities: Vec<String>,
    }

    impl AgentHandle for MockAgent {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> &[String] {
            &self.capabilities
        }

        fn invoke<'a>(
            &'a self,
            request: AgentRequest,
        ) -> Pin<Box<dyn Future<Output = Result<AgentResponse, RegistryError>> + Send + 'a>>
        {
            Box::pin(async move {
                Ok(AgentResponse {
                    content: format!("Response to: {}", request.message),
                    agent_name: self.name.clone(),
                })
            })
        }
    }

    #[tokio::test]
    async fn test_register_and_invoke() {
        let registry = AgentRegistry::new(5);

        let agent = Arc::new(MockAgent {
            name: "test-agent".to_string(),
            capabilities: vec!["testing".to_string()],
        });

        registry.register(agent).await.unwrap();
        assert!(registry.has("test-agent").await);

        let response = registry
            .invoke("test-agent", AgentRequest::new("Hello"))
            .await
            .unwrap();

        assert_eq!(response.agent_name, "test-agent");
        assert!(response.content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_discover_by_capability() {
        let registry = AgentRegistry::new(5);

        let agent1 = Arc::new(MockAgent {
            name: "analyst".to_string(),
            capabilities: vec!["analysis".to_string(), "data".to_string()],
        });

        let agent2 = Arc::new(MockAgent {
            name: "writer".to_string(),
            capabilities: vec!["writing".to_string()],
        });

        registry.register(agent1).await.unwrap();
        registry.register(agent2).await.unwrap();

        let analysts = registry.discover("analysis").await;
        assert_eq!(analysts.len(), 1);
        assert_eq!(analysts[0].name(), "analyst");

        let writers = registry.discover("writing").await;
        assert_eq!(writers.len(), 1);
        assert_eq!(writers[0].name(), "writer");
    }

    #[tokio::test]
    async fn test_max_depth_exceeded() {
        let registry = AgentRegistry::new(0); // Max depth 0

        let agent = Arc::new(MockAgent {
            name: "test".to_string(),
            capabilities: vec![],
        });

        registry.register(agent).await.unwrap();

        let result = registry.invoke("test", AgentRequest::new("Hello")).await;
        assert!(matches!(result, Err(RegistryError::MaxDepthExceeded(0))));
    }

    #[tokio::test]
    async fn test_shared_context() {
        let registry = AgentRegistry::new(5);

        let shared_ctx = registry.shared_context();
        {
            let mut ctx = shared_ctx.write().await;
            ctx.set("key", "value").unwrap();
        }

        {
            let ctx = shared_ctx.read().await;
            let value: String = ctx.get("key").unwrap().unwrap();
            assert_eq!(value, "value");
        }
    }

    #[tokio::test]
    async fn test_agent_not_found() {
        let registry = AgentRegistry::new(5);

        let result = registry
            .invoke("nonexistent", AgentRequest::new("Hello"))
            .await;
        assert!(matches!(result, Err(RegistryError::NotFound(_))));
    }
}
