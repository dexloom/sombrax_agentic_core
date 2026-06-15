//! LLM Provider abstractions
//!
//! Provides the CompletionModel trait and related types for LLM provider adapters.
//!
//! For concrete provider implementations, see the [`crate::providers`] module which includes:
//! - OpenAI (`OpenAIClient`, `OpenAIClientBuilder`)
//! - Anthropic (`AnthropicClient`, `AnthropicClientBuilder`)
//! - MiniMax (`MinimaxClient`, `MinimaxClientBuilder`)
//! - ZAI (`ZaiClient`, `ZaiClientBuilder`)
//! - Cerebras (`CerebrasClient`, `CerebrasClientBuilder`)
//! - OpenRouter (`OpenRouterClient`, `OpenRouterClientBuilder`)
//!
//! ## Metrics Wrapper
//!
//! Use [`MetricsCompletionModel`] to wrap any provider with automatic metrics collection:
//!
//! ```ignore
//! use sombrax_agentic_core::provider::CompletionModelExt;
//!
//! let model = client.completion_model("gpt-4o").with_metrics();
//! let response = model.completion(request).await?;
//! // Metrics automatically recorded to OpenTelemetry and tracing
//! ```

pub mod obfuscate;

pub use obfuscate::{FnObfuscator, MapObfuscator, ObfuscatingCompletionModel, Obfuscator};

use crate::error::CompletionError;
use crate::message::Message;
use crate::telemetry::{classify_completion_error, CompletionTiming, LlmMetrics};
use crate::tool::ToolDefinition;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::future::Future;
use std::time::Instant;

/// Token usage statistics
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens
    pub input_tokens: u64,
    /// Number of output tokens
    pub output_tokens: u64,
    /// Number of tokens read from cache (cache hits)
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Number of tokens written to cache (cache creation)
    #[serde(default)]
    pub cache_creation_tokens: u64,
}

impl Usage {
    /// Create a new Usage with the given token counts
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        }
    }

    /// Create a new Usage with cache token counts
    pub fn with_cache(
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        }
    }

    /// Get the total number of tokens
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Get the total cache tokens (read + creation)
    pub fn cache_total(&self) -> u64 {
        self.cache_read_tokens + self.cache_creation_tokens
    }
}

impl std::ops::Add for Usage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            cache_read_tokens: self.cache_read_tokens + rhs.cache_read_tokens,
            cache_creation_tokens: self.cache_creation_tokens + rhs.cache_creation_tokens,
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.cache_read_tokens += rhs.cache_read_tokens;
        self.cache_creation_tokens += rhs.cache_creation_tokens;
    }
}

/// Provider-independent prompt-cache hints for a completion request.
///
/// These express *intent* — "this prefix is stable, cache it" — without
/// committing to any provider's wire format. Providers with an explicit cache
/// protocol (Anthropic-style `cache_control` breakpoints) translate these into
/// markers; providers that cache implicitly on prefix match (OpenAI-style)
/// simply ignore them and benefit from a stable, append-only message prefix.
///
/// The agent loop populates this from its own knowledge of which messages were
/// already sent (the high-water mark). Optimizers keep the prefix below that
/// mark byte-identical between deliberate compaction points, so the cached
/// prefix stays valid turn-to-turn.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheHints {
    /// Request a cache breakpoint covering the system preamble + tool
    /// definitions block (the static prefix that never changes within a run).
    pub cache_system: bool,

    /// Indices into `CompletionRequest.messages` after which a cache
    /// breakpoint is requested. Providers translate each index to the last
    /// content block of the corresponding (post-merge) message. Implicit-cache
    /// providers ignore this field entirely. Indices that fall out of range
    /// after provider-side message merging are skipped, not errors.
    pub breakpoints: Vec<usize>,
}

impl CacheHints {
    /// True when no caching is requested (the default / disabled state).
    pub fn is_empty(&self) -> bool {
        !self.cache_system && self.breakpoints.is_empty()
    }
}

/// Completion request sent to provider
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// System prompt / preamble
    pub preamble: Option<String>,

    /// Conversation messages
    pub messages: Vec<Message>,

    /// Available tool definitions
    pub tools: Vec<ToolDefinition>,

    /// Sampling temperature (0.0 - 2.0)
    pub temperature: Option<f64>,

    /// Maximum tokens in response
    pub max_tokens: Option<u64>,

    /// Provider-specific parameters
    pub additional_params: Option<serde_json::Value>,

    /// Provider-independent prompt-cache hints. Default is empty (no caching).
    pub cache: CacheHints,
}

impl CompletionRequest {
    /// Create a new completion request with a single message
    pub fn new(message: impl Into<Message>) -> Self {
        Self {
            messages: vec![message.into()],
            ..Default::default()
        }
    }

    /// Set the system preamble
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Add messages to the request
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Add a single message
    pub fn add_message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }

    /// Set the available tools
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the max tokens
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set additional provider-specific parameters
    pub fn with_additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

/// Completion response from provider
#[derive(Debug, Clone)]
pub struct CompletionResponse<R = serde_json::Value> {
    /// The assistant's response message
    pub message: Message,

    /// Token usage statistics
    pub usage: Usage,

    /// Raw provider-specific response (for debugging/logging)
    pub raw: R,

    /// Optional reasoning/thinking content (for models that support extended thinking)
    ///
    /// This is populated by providers that support thinking mode (e.g., ZAI with GLM-4,
    /// MlxLm with MiniMax M2.1). The content contains the model's internal reasoning
    /// extracted from `<think>...</think>` tags or dedicated response fields.
    pub reasoning_content: Option<String>,

    /// The reason the model stopped generating.
    ///
    /// Common values: "stop" / "end_turn" (normal completion), "length" (max_tokens hit),
    /// "tool_use" / "tool_calls" (model wants to call tools).
    /// `None` indicates the provider didn't report a reason (possible streaming disconnect).
    pub finish_reason: Option<String>,
}

impl<R> CompletionResponse<R> {
    /// Create a new completion response
    pub fn new(message: Message, usage: Usage, raw: R) -> Self {
        Self {
            message,
            usage,
            raw,
            reasoning_content: None,
            finish_reason: None,
        }
    }

    /// Create a new completion response with reasoning content
    pub fn with_reasoning(
        message: Message,
        usage: Usage,
        raw: R,
        reasoning: Option<String>,
    ) -> Self {
        Self {
            message,
            usage,
            raw,
            reasoning_content: reasoning,
            finish_reason: None,
        }
    }

    /// Check if this response was truncated due to hitting max_tokens.
    ///
    /// Returns `true` if `finish_reason` indicates the response was cut off:
    /// - `"length"` — OpenAI-style providers (openai, zai, cerebras, openrouter, mlxlm)
    /// - `"max_tokens"` — Anthropic-style providers (anthropic, minimax)
    ///
    /// Note: `None` finish_reason is ambiguous (could be a streaming disconnect or a mock/test
    /// response), so it's handled reactively by the JSON parse error check in the agent loop.
    pub fn is_truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("length") | Some("max_tokens")
        )
    }

    /// Get the text content of the response
    pub fn content(&self) -> String {
        self.message.text()
    }

    /// Check if the response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.message.has_tool_calls()
    }

    /// Get tool calls from the response
    pub fn tool_calls(&self) -> Vec<&crate::message::ToolCall> {
        self.message.tool_calls()
    }
}

/// LLM Provider abstraction (FR-018)
///
/// Implementors provide adapters for specific LLM providers (OpenAI, Anthropic, etc.).
/// Responses are fully buffered (FR-002a) - no streaming support.
///
/// # Example
///
/// ```ignore
/// use sombrax_agentic_core::providers::{OpenAIClientBuilder, OpenAIClientExt};
///
/// let client = OpenAIClientBuilder::new("api-key").build();
/// let model = client.completion_model_adapter("gpt-4");
///
/// let response = model.completion(CompletionRequest::new("Hello!")).await?;
/// println!("{}", response.content());
/// ```
pub trait CompletionModel: Clone + Send + Sync + 'static {
    /// The raw response type from the provider
    type Response: Send + Sync + Serialize + DeserializeOwned + 'static;

    /// Send a completion request and receive a buffered response
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + Send;

    /// Returns the model identifier (e.g., "gpt-4", "claude-3-opus")
    fn model_id(&self) -> &str;

    /// Returns the provider name (e.g., "openai", "anthropic")
    fn provider(&self) -> &str;
}

/// Builder for constructing completion requests
pub struct CompletionRequestBuilder<M: CompletionModel> {
    model: M,
    request: CompletionRequest,
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    /// Create a new builder with the given model and initial message
    pub fn new(model: M, prompt: impl Into<Message>) -> Self {
        Self {
            model,
            request: CompletionRequest::new(prompt),
        }
    }

    /// Set the system preamble
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.request.preamble = Some(preamble.into());
        self
    }

    /// Set the conversation messages
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.request.messages = messages;
        self
    }

    /// Add a message to the conversation
    pub fn add_message(mut self, message: impl Into<Message>) -> Self {
        self.request.messages.push(message.into());
        self
    }

    /// Set the available tools
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.request.tools = tools;
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.request.temperature = Some(temp);
        self
    }

    /// Set the max tokens
    pub fn max_tokens(mut self, max: u64) -> Self {
        self.request.max_tokens = Some(max);
        self
    }

    /// Set additional provider-specific parameters
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.request.additional_params = Some(params);
        self
    }

    /// Send the completion request
    pub async fn send(self) -> Result<CompletionResponse<M::Response>, CompletionError> {
        self.model.completion(self.request).await
    }
}

// =============================================================================
// Metrics Wrapper
// =============================================================================

/// Wrapper that adds automatic metrics collection to any [`CompletionModel`]
///
/// Wraps an inner model and records timing, token counts, and success/failure
/// metrics for every completion request. Metrics are recorded to both
/// OpenTelemetry (histograms/counters) and tracing (debug events).
///
/// # Usage
///
/// Use the [`CompletionModelExt::with_metrics`] extension method for convenience:
///
/// ```ignore
/// use sombrax_agentic_core::provider::CompletionModelExt;
///
/// let model = client.completion_model("gpt-4o").with_metrics();
/// ```
///
/// Or construct directly:
///
/// ```ignore
/// use sombrax_agentic_core::provider::MetricsCompletionModel;
///
/// let inner = client.completion_model("gpt-4o");
/// let model = MetricsCompletionModel::new(inner);
/// ```
#[derive(Clone)]
pub struct MetricsCompletionModel<M: CompletionModel> {
    inner: M,
    metrics: LlmMetrics,
}

impl<M: CompletionModel> MetricsCompletionModel<M> {
    /// Create a new metrics wrapper using global metrics
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            metrics: LlmMetrics::global(),
        }
    }

    /// Create a new metrics wrapper with custom LlmMetrics instance
    pub fn with_custom_metrics(inner: M, metrics: LlmMetrics) -> Self {
        Self { inner, metrics }
    }

    /// Get a reference to the inner model
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Unwrap and return the inner model
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: CompletionModel> CompletionModel for MetricsCompletionModel<M> {
    type Response = M::Response;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let start = Instant::now();

        let result = self.inner.completion(request).await;

        let latency = start.elapsed();

        let timing = match &result {
            Ok(response) => CompletionTiming {
                latency_ms: latency.as_secs_f64() * 1000.0,
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
                provider: self.inner.provider().to_string(),
                model: self.inner.model_id().to_string(),
                success: true,
                error_type: None,
            },
            Err(e) => CompletionTiming {
                latency_ms: latency.as_secs_f64() * 1000.0,
                input_tokens: 0,
                output_tokens: 0,
                provider: self.inner.provider().to_string(),
                model: self.inner.model_id().to_string(),
                success: false,
                error_type: Some(classify_completion_error(e)),
            },
        };

        self.metrics.record(&timing);

        result
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }
}

/// Extension trait for easily wrapping any [`CompletionModel`] with metrics
///
/// # Example
///
/// ```ignore
/// use sombrax_agentic_core::provider::CompletionModelExt;
///
/// let model = client.completion_model("gpt-4o").with_metrics();
/// let response = model.completion(request).await?;
/// // Metrics automatically recorded
/// ```
pub trait CompletionModelExt: CompletionModel + Sized {
    /// Wrap this model with automatic metrics collection
    ///
    /// Returns a [`MetricsCompletionModel`] that records timing, token counts,
    /// and success/failure metrics for every completion request.
    fn with_metrics(self) -> MetricsCompletionModel<Self> {
        MetricsCompletionModel::new(self)
    }

    /// Wrap this model with text obfuscation.
    ///
    /// The obfuscator is applied to all text content in requests (obfuscate)
    /// and responses (de-obfuscate), ensuring sensitive text never reaches
    /// the LLM provider.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use sombrax_agentic_core::provider::{CompletionModelExt, MapObfuscator};
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("0xdead...beef".into(), "CONTRACT_A".into());
    ///
    /// let model = client
    ///     .completion_model_adapter("gpt-4o")
    ///     .with_obfuscator(MapObfuscator::new(map))
    ///     .with_metrics();
    /// ```
    fn with_obfuscator<O: Obfuscator>(self, obfuscator: O) -> ObfuscatingCompletionModel<Self, O> {
        ObfuscatingCompletionModel::new(self, obfuscator)
    }
}

/// Blanket implementation for all CompletionModel types
impl<M: CompletionModel> CompletionModelExt for M {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_operations() {
        let u1 = Usage::new(100, 50);
        let u2 = Usage::new(200, 100);

        assert_eq!(u1.total(), 150);
        assert_eq!((u1 + u2).total(), 450);

        let mut u3 = Usage::new(10, 5);
        u3 += Usage::new(20, 10);
        assert_eq!(u3.total(), 45);
    }

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::new("Hello")
            .with_preamble("You are helpful")
            .with_temperature(0.7)
            .with_max_tokens(100);

        assert_eq!(request.preamble, Some("You are helpful".to_string()));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.messages.len(), 1);
    }
}
