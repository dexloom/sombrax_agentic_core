//! Context optimization strategies
//!
//! Provides traits and implementations for managing conversation context size.

use crate::message::Message;
use crate::telemetry::Metrics;
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

/// Configuration for context optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Always preserve this many most recent messages (FR-014)
    pub preserve_recent: usize,

    /// Message IDs that should never be removed (FR-015)
    pub pinned_ids: HashSet<String>,

    /// Target token budget
    pub token_budget: usize,

    /// Number of messages at the start of the history that are frozen (immutable).
    ///
    /// When set to a non-zero value, the optimizer MUST NOT modify, drop, update content,
    /// or inject messages within the first `frozen_prefix_len` messages. This preserves
    /// the LLM's KV cache prefix when continuing from a previous agent execution.
    ///
    /// Use case: multi-stage agent pipelines where Stage 2 receives Stage 1's message
    /// history. Modifying any Stage 1 message would invalidate the cache prefix.
    pub frozen_prefix_len: usize,

    /// Number of messages included in the *previous* completion request
    /// (the cache high-water mark); 0 before the first request.
    ///
    /// Advisory hint, not enforced by SAC. Cache-aware optimizers treat the
    /// first `last_sent_len` messages as already sent (cache-hot) and avoid
    /// mutating them outside deliberate compaction points, so the provider's
    /// implicit prefix cache stays valid turn-to-turn. The agent loop sets
    /// this each turn; optimizers that ignore it behave exactly as before.
    pub last_sent_len: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            preserve_recent: 10,
            pinned_ids: HashSet::new(),
            token_budget: 4096,
            frozen_prefix_len: 0,
            last_sent_len: 0,
        }
    }
}

impl OptimizationConfig {
    /// Create a new optimization config with the given token budget
    pub fn with_budget(token_budget: usize) -> Self {
        Self {
            token_budget,
            ..Default::default()
        }
    }

    /// Set the number of recent messages to preserve
    pub fn preserve_recent(mut self, n: usize) -> Self {
        self.preserve_recent = n;
        self
    }

    /// Add a pinned message ID
    pub fn pin(mut self, id: impl Into<String>) -> Self {
        self.pinned_ids.insert(id.into());
        self
    }

    /// Add multiple pinned message IDs
    pub fn pin_all(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for id in ids {
            self.pinned_ids.insert(id.into());
        }
        self
    }

    /// Set the frozen prefix length.
    ///
    /// The first `n` messages will be treated as immutable by the optimizer,
    /// preserving the LLM's KV cache prefix for multi-stage continuations.
    pub fn frozen_prefix(mut self, n: usize) -> Self {
        self.frozen_prefix_len = n;
        self
    }

    /// Set the cache high-water mark (count of messages previously sent).
    ///
    /// Advisory; cache-aware optimizers use it to keep the already-sent prefix
    /// byte-identical between compaction points.
    pub fn last_sent_len(mut self, n: usize) -> Self {
        self.last_sent_len = n;
        self
    }
}

/// Context optimization strategy (FR-013)
///
/// Implementors provide strategies for reducing context size while preserving
/// important information.
pub trait ContextOptimizer: Send + Sync {
    /// Optimize context to fit within the configured budget
    ///
    /// Must preserve:
    /// - Messages with IDs in `config.pinned_ids` (FR-015)
    /// - The `config.preserve_recent` most recent messages (FR-014)
    fn optimize<'a>(
        &'a self,
        messages: Vec<Message>,
        config: &'a OptimizationConfig,
    ) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + 'a>>;

    /// Estimate token count for a message (for budget calculations)
    fn estimate_tokens(&self, message: &Message) -> usize {
        // Default: rough estimate of 4 chars per token
        // Implementations should use proper tokenizers
        message.content_length() / 4 + 1
    }

    /// Estimate total tokens for a list of messages
    fn estimate_total_tokens(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.estimate_tokens(m)).sum()
    }
}

/// Recency-based optimizer (drop oldest first)
#[derive(Debug, Clone, Default)]
pub struct RecencyOptimizer;

impl RecencyOptimizer {
    /// Create a new recency optimizer
    pub fn new() -> Self {
        Self
    }
}

impl ContextOptimizer for RecencyOptimizer {
    fn optimize<'a>(
        &'a self,
        messages: Vec<Message>,
        config: &'a OptimizationConfig,
    ) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + 'a>> {
        Box::pin(async move {
            let total = messages.len();
            if total <= config.preserve_recent {
                return messages;
            }

            let preserve_from = total.saturating_sub(config.preserve_recent);

            let result: Vec<Message> = messages
                .into_iter()
                .enumerate()
                .filter(|(i, m)| {
                    // Keep pinned messages
                    if let Some(id) = m.id() {
                        if config.pinned_ids.contains(id) {
                            return true;
                        }
                    }
                    // Keep recent messages
                    *i >= preserve_from
                })
                .map(|(_, m)| m)
                .collect();

            // Record optimization metric (FR-021)
            let metrics = Metrics::global();
            metrics.record_optimization("recency", total, result.len());

            result
        })
    }
}

/// Priority-based optimizer (requires messages to have priority metadata)
#[derive(Debug, Clone, Default)]
pub struct PriorityOptimizer {
    /// Priority key in message metadata
    priority_key: String,
    /// Default priority for messages without metadata
    default_priority: i32,
}

impl PriorityOptimizer {
    /// Create a new priority optimizer
    pub fn new() -> Self {
        Self {
            priority_key: "priority".to_string(),
            default_priority: 0,
        }
    }

    /// Set the priority key used in message metadata
    pub fn with_priority_key(mut self, key: impl Into<String>) -> Self {
        self.priority_key = key.into();
        self
    }

    /// Set the default priority for messages without metadata
    pub fn with_default_priority(mut self, priority: i32) -> Self {
        self.default_priority = priority;
        self
    }
}

impl ContextOptimizer for PriorityOptimizer {
    fn optimize<'a>(
        &'a self,
        messages: Vec<Message>,
        config: &'a OptimizationConfig,
    ) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + 'a>> {
        Box::pin(async move {
            let total = messages.len();
            if total <= config.preserve_recent {
                return messages;
            }

            let preserve_from = total.saturating_sub(config.preserve_recent);

            // Collect messages with their indices and priorities
            let mut indexed: Vec<(usize, Message, i32)> = messages
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    // Pinned messages get max priority
                    let priority = if m
                        .id()
                        .map(|id| config.pinned_ids.contains(id))
                        .unwrap_or(false)
                    {
                        i32::MAX
                    } else if i >= preserve_from {
                        // Recent messages get high priority
                        i32::MAX - 1
                    } else {
                        self.default_priority
                    };
                    (i, m, priority)
                })
                .collect();

            // Sort by priority (descending) then by index (ascending)
            indexed.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));

            // Calculate how many we need to keep
            let current_tokens: usize = indexed
                .iter()
                .map(|(_, m, _)| self.estimate_tokens(m))
                .sum();
            let mut tokens_to_remove = current_tokens.saturating_sub(config.token_budget);

            // Remove lowest priority messages until we're under budget
            let mut keep = vec![true; indexed.len()];
            for i in (0..indexed.len()).rev() {
                if tokens_to_remove == 0 {
                    break;
                }
                // Skip pinned and recent messages
                if indexed[i].2 >= i32::MAX - 1 {
                    continue;
                }
                let msg_tokens = self.estimate_tokens(&indexed[i].1);
                keep[i] = false;
                tokens_to_remove = tokens_to_remove.saturating_sub(msg_tokens);
            }

            // Reconstruct in original order
            let result: Vec<_> = indexed
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep[*i])
                .map(|(_, (_, m, _))| m)
                .collect();

            // Record optimization metric (FR-021)
            let metrics = Metrics::global();
            metrics.record_optimization("priority", total, result.len());

            result
        })
    }
}

/// Truncation-based optimizer (truncates long messages instead of removing)
#[derive(Debug, Clone)]
pub struct TruncationOptimizer {
    /// Maximum length per message in characters
    max_message_length: usize,
    /// Truncation suffix
    suffix: String,
}

impl Default for TruncationOptimizer {
    fn default() -> Self {
        Self {
            max_message_length: 1000,
            suffix: "... [truncated]".to_string(),
        }
    }
}

impl TruncationOptimizer {
    /// Create a new truncation optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum message length
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_message_length = max;
        self
    }

    /// Set the truncation suffix
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Truncate a single message
    fn truncate_message(&self, mut message: Message) -> Message {
        let text = message.text();
        if text.len() > self.max_message_length {
            let truncated = format!(
                "{}{}",
                &text[..self.max_message_length.saturating_sub(self.suffix.len())],
                self.suffix
            );

            // Recreate with truncated text
            match &message {
                Message::User { id, .. } => {
                    message = if let Some(id) = id {
                        Message::user_with_id(truncated, id)
                    } else {
                        Message::user(truncated)
                    };
                }
                Message::Assistant { id, .. } => {
                    message = if let Some(id) = id {
                        Message::assistant_with_id(truncated, id)
                    } else {
                        Message::assistant(truncated)
                    };
                }
            }
        }
        message
    }
}

impl ContextOptimizer for TruncationOptimizer {
    fn optimize<'a>(
        &'a self,
        messages: Vec<Message>,
        config: &'a OptimizationConfig,
    ) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send + 'a>> {
        Box::pin(async move {
            let total = messages.len();
            let preserve_from = total.saturating_sub(config.preserve_recent);

            let result: Vec<Message> = messages
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    // Don't truncate pinned or recent messages
                    let is_pinned = m
                        .id()
                        .map(|id| config.pinned_ids.contains(id))
                        .unwrap_or(false);
                    let is_recent = i >= preserve_from;

                    if is_pinned || is_recent {
                        m
                    } else {
                        self.truncate_message(m)
                    }
                })
                .collect();

            // Record optimization metric (FR-021)
            // For truncation, we report the same count since messages aren't removed
            let metrics = Metrics::global();
            metrics.record_optimization("truncation", total, result.len());

            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recency_optimizer() {
        let messages = vec![
            Message::user("Message 1"),
            Message::user("Message 2"),
            Message::user("Message 3"),
            Message::user("Message 4"),
            Message::user("Message 5"),
        ];

        let config = OptimizationConfig::default().preserve_recent(2);
        let optimizer = RecencyOptimizer::new();

        let result = optimizer.optimize(messages, &config).await;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text(), "Message 4");
        assert_eq!(result[1].text(), "Message 5");
    }

    #[tokio::test]
    async fn test_recency_optimizer_with_pinned() {
        let messages = vec![
            Message::user_with_id("Important message", "pinned-1"),
            Message::user("Message 2"),
            Message::user("Message 3"),
            Message::user("Message 4"),
            Message::user("Message 5"),
        ];

        let config = OptimizationConfig::default()
            .preserve_recent(2)
            .pin("pinned-1");
        let optimizer = RecencyOptimizer::new();

        let result = optimizer.optimize(messages, &config).await;

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text(), "Important message");
        assert_eq!(result[1].text(), "Message 4");
        assert_eq!(result[2].text(), "Message 5");
    }

    #[tokio::test]
    async fn test_truncation_optimizer() {
        let long_text = "a".repeat(2000);
        let messages = vec![Message::user(&long_text), Message::user("Short message")];

        let config = OptimizationConfig::default().preserve_recent(1);
        let optimizer = TruncationOptimizer::new().with_max_length(100);

        let result = optimizer.optimize(messages, &config).await;

        assert_eq!(result.len(), 2);
        assert!(result[0].text().len() <= 100);
        assert!(result[0].text().ends_with("[truncated]"));
        assert_eq!(result[1].text(), "Short message"); // Recent, not truncated
    }

    #[test]
    fn test_optimization_config_builder() {
        let config = OptimizationConfig::with_budget(8192)
            .preserve_recent(20)
            .pin("msg-1")
            .pin("msg-2");

        assert_eq!(config.token_budget, 8192);
        assert_eq!(config.preserve_recent, 20);
        assert!(config.pinned_ids.contains("msg-1"));
        assert!(config.pinned_ids.contains("msg-2"));
    }
}
