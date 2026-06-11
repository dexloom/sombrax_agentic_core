//! Telemetry module for observability
//!
//! Provides OpenTelemetry integration for tracing and metrics (FR-020, FR-021).
//!
//! ## LLM Performance Metrics
//!
//! The [`LlmMetrics`] struct provides comprehensive metrics for LLM completion requests:
//! - End-to-end latency (histogram)
//! - Throughput: tokens per second (output, input, total)
//! - Request counters (total, success, failed by error type)
//! - Token counters (cumulative input/output)
//!
//! Use the [`CompletionTiming`] struct to capture timing data and record it via
//! [`LlmMetrics::record`], which logs to both OpenTelemetry metrics and tracing events.

use crate::error::CompletionError;
use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};
use std::time::Duration;
use tracing::debug;

/// Metrics for the agent hook library
pub struct Metrics {
    /// Request latency histogram
    request_latency: Histogram<f64>,
    /// Hook execution duration histogram
    hook_duration: Histogram<f64>,
    /// Tool call counter
    tool_calls: Counter<u64>,
    /// Tool error counter
    tool_errors: Counter<u64>,
    /// Completion request counter
    completion_requests: Counter<u64>,
    /// Completion error counter
    completion_errors: Counter<u64>,
    /// Context optimization events counter
    optimization_events: Counter<u64>,
}

impl Metrics {
    /// Create a new Metrics instance with the given meter
    pub fn new(meter: &Meter) -> Self {
        Self {
            request_latency: meter
                .f64_histogram("sac.request.latency")
                .with_description("Request latency in milliseconds")
                .with_unit("ms")
                .build(),
            hook_duration: meter
                .f64_histogram("sac.hook.duration")
                .with_description("Hook execution duration in milliseconds")
                .with_unit("ms")
                .build(),
            tool_calls: meter
                .u64_counter("sac.tool.calls")
                .with_description("Number of tool calls")
                .build(),
            tool_errors: meter
                .u64_counter("sac.tool.errors")
                .with_description("Number of tool call errors")
                .build(),
            completion_requests: meter
                .u64_counter("sac.completion.requests")
                .with_description("Number of completion requests")
                .build(),
            completion_errors: meter
                .u64_counter("sac.completion.errors")
                .with_description("Number of completion errors")
                .build(),
            optimization_events: meter
                .u64_counter("sac.context.optimizations")
                .with_description("Number of context optimization events")
                .build(),
        }
    }

    /// Create a new Metrics instance using the global meter provider
    pub fn global() -> Self {
        let meter = global::meter("sac");
        Self::new(&meter)
    }

    /// Record request latency
    pub fn record_request_latency(&self, duration: Duration, attributes: &[KeyValue]) {
        self.request_latency
            .record(duration.as_secs_f64() * 1000.0, attributes);
    }

    /// Record hook execution duration
    pub fn record_hook_duration(&self, hook_name: &str, stage: &str, duration: Duration) {
        self.hook_duration.record(
            duration.as_secs_f64() * 1000.0,
            &[
                KeyValue::new("hook_name", hook_name.to_string()),
                KeyValue::new("stage", stage.to_string()),
            ],
        );
    }

    /// Increment tool call counter
    pub fn record_tool_call(&self, tool_name: &str, success: bool) {
        let attributes = &[
            KeyValue::new("tool_name", tool_name.to_string()),
            KeyValue::new("success", success),
        ];
        self.tool_calls.add(1, attributes);
        if !success {
            self.tool_errors.add(1, attributes);
        }
    }

    /// Increment completion request counter
    pub fn record_completion_request(&self, provider: &str, model: &str, success: bool) {
        let attributes = &[
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("success", success),
        ];
        self.completion_requests.add(1, attributes);
        if !success {
            self.completion_errors.add(1, attributes);
        }
    }

    /// Record context optimization event
    pub fn record_optimization(
        &self,
        strategy: &str,
        messages_before: usize,
        messages_after: usize,
    ) {
        self.optimization_events.add(
            1,
            &[
                KeyValue::new("strategy", strategy.to_string()),
                KeyValue::new(
                    "messages_removed",
                    (messages_before - messages_after) as i64,
                ),
            ],
        );
    }
}

/// Initialize a default metrics instance
///
/// This uses the global meter provider. Make sure to configure OpenTelemetry
/// before calling this function if you want metrics to be exported.
pub fn init_metrics() -> Metrics {
    Metrics::global()
}

/// Create a tracing span for a hook chain execution
#[macro_export]
macro_rules! hook_span {
    ($request_id:expr) => {
        tracing::info_span!("hook_chain", request_id = %$request_id)
    };
}

/// Create a tracing span for a hook execution
#[macro_export]
macro_rules! hook_execution_span {
    ($hook_name:expr, $stage:expr) => {
        tracing::info_span!("hook_execution", hook_name = %$hook_name, stage = %$stage)
    };
}

/// Create a tracing span for a tool call
#[macro_export]
macro_rules! tool_call_span {
    ($tool_name:expr, $call_id:expr) => {
        tracing::info_span!("tool_call", tool_name = %$tool_name, call_id = %$call_id)
    };
}

/// Create a tracing span for a completion request
#[macro_export]
macro_rules! completion_span {
    ($provider:expr, $model:expr) => {
        tracing::info_span!("completion", provider = %$provider, model = %$model)
    };
}

// =============================================================================
// LLM Performance Metrics
// =============================================================================

/// Timing data captured around LLM completion requests
///
/// Captures latency, token counts, and metadata for recording metrics.
/// Use with [`LlmMetrics::record`] to log to both OpenTelemetry and tracing.
///
/// # Example
///
/// ```ignore
/// use std::time::Instant;
/// use sombrax_agentic_core::telemetry::{CompletionTiming, LlmMetrics};
///
/// let start = Instant::now();
/// let result = model.completion(request).await;
/// let latency = start.elapsed();
///
/// let timing = CompletionTiming {
///     latency_ms: latency.as_secs_f64() * 1000.0,
///     input_tokens: response.usage.input_tokens,
///     output_tokens: response.usage.output_tokens,
///     provider: "openai".to_string(),
///     model: "gpt-4o".to_string(),
///     success: true,
///     error_type: None,
/// };
///
/// let metrics = LlmMetrics::global();
/// metrics.record(&timing);
/// ```
#[derive(Debug, Clone)]
pub struct CompletionTiming {
    /// End-to-end latency in milliseconds
    pub latency_ms: f64,
    /// Number of input/prompt tokens
    pub input_tokens: u64,
    /// Number of output/completion tokens
    pub output_tokens: u64,
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: String,
    /// Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
    pub model: String,
    /// Whether the request succeeded
    pub success: bool,
    /// Error type for failed requests (e.g., "rate_limited", "auth_error")
    pub error_type: Option<String>,
}

impl CompletionTiming {
    /// Calculate output tokens per second (generation throughput)
    ///
    /// Returns 0.0 if latency is zero to avoid division by zero.
    pub fn output_tps(&self) -> f64 {
        if self.latency_ms > 0.0 {
            (self.output_tokens as f64) / (self.latency_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Calculate input tokens per second (prefill throughput proxy)
    ///
    /// Returns 0.0 if latency is zero to avoid division by zero.
    pub fn input_tps(&self) -> f64 {
        if self.latency_ms > 0.0 {
            (self.input_tokens as f64) / (self.latency_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Calculate total tokens per second (overall throughput)
    ///
    /// Returns 0.0 if latency is zero to avoid division by zero.
    pub fn total_tps(&self) -> f64 {
        if self.latency_ms > 0.0 {
            ((self.input_tokens + self.output_tokens) as f64) / (self.latency_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Get total token count
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// LLM performance metrics collector
///
/// Records completion request metrics to both OpenTelemetry (histograms/counters)
/// and tracing (debug events). Use [`LlmMetrics::global`] to get a shared instance
/// or [`LlmMetrics::new`] with a custom meter.
///
/// # Metrics Recorded
///
/// - `sac.llm.latency_ms` - End-to-end latency histogram
/// - `sac.llm.output_tokens_per_second` - Output generation throughput
/// - `sac.llm.input_tokens_per_second` - Input processing throughput
/// - `sac.llm.total_tokens_per_second` - Overall throughput
/// - `sac.llm.requests_total` - Total request counter
/// - `sac.llm.requests_success` - Successful request counter
/// - `sac.llm.requests_failed` - Failed request counter (by error type)
/// - `sac.llm.tokens_input_total` - Cumulative input tokens
/// - `sac.llm.tokens_output_total` - Cumulative output tokens
#[derive(Clone)]
pub struct LlmMetrics {
    /// End-to-end latency histogram (milliseconds)
    e2e_latency: Histogram<f64>,
    /// Output tokens per second histogram
    output_tps: Histogram<f64>,
    /// Input tokens per second histogram
    input_tps: Histogram<f64>,
    /// Total tokens per second histogram
    total_tps: Histogram<f64>,
    /// Total requests counter
    requests_total: Counter<u64>,
    /// Successful requests counter
    requests_success: Counter<u64>,
    /// Failed requests counter
    requests_failed: Counter<u64>,
    /// Cumulative input tokens counter
    tokens_input_total: Counter<u64>,
    /// Cumulative output tokens counter
    tokens_output_total: Counter<u64>,
}

impl LlmMetrics {
    /// Create new LlmMetrics with the given meter
    pub fn new(meter: &Meter) -> Self {
        Self {
            e2e_latency: meter
                .f64_histogram("sac.llm.latency_ms")
                .with_description("End-to-end completion latency in milliseconds")
                .with_unit("ms")
                .build(),
            output_tps: meter
                .f64_histogram("sac.llm.output_tokens_per_second")
                .with_description("Output token generation throughput")
                .with_unit("tokens/s")
                .build(),
            input_tps: meter
                .f64_histogram("sac.llm.input_tokens_per_second")
                .with_description("Input token processing throughput")
                .with_unit("tokens/s")
                .build(),
            total_tps: meter
                .f64_histogram("sac.llm.total_tokens_per_second")
                .with_description("Total token throughput")
                .with_unit("tokens/s")
                .build(),
            requests_total: meter
                .u64_counter("sac.llm.requests_total")
                .with_description("Total LLM completion requests")
                .build(),
            requests_success: meter
                .u64_counter("sac.llm.requests_success")
                .with_description("Successful LLM completion requests")
                .build(),
            requests_failed: meter
                .u64_counter("sac.llm.requests_failed")
                .with_description("Failed LLM completion requests")
                .build(),
            tokens_input_total: meter
                .u64_counter("sac.llm.tokens_input_total")
                .with_description("Total input tokens processed")
                .build(),
            tokens_output_total: meter
                .u64_counter("sac.llm.tokens_output_total")
                .with_description("Total output tokens generated")
                .build(),
        }
    }

    /// Create LlmMetrics using the global meter provider
    pub fn global() -> Self {
        let meter = global::meter("sac.llm");
        Self::new(&meter)
    }

    /// Record completion metrics to OpenTelemetry and tracing
    ///
    /// This method:
    /// 1. Records histograms for latency and throughput metrics
    /// 2. Increments counters for requests and tokens
    /// 3. Emits a tracing::debug event with all metrics
    pub fn record(&self, timing: &CompletionTiming) {
        let attributes = &[
            KeyValue::new("provider", timing.provider.clone()),
            KeyValue::new("model", timing.model.clone()),
        ];

        // Record latency histogram
        self.e2e_latency.record(timing.latency_ms, attributes);

        // Record throughput histograms (only for successful requests with output)
        if timing.success && timing.output_tokens > 0 {
            self.output_tps.record(timing.output_tps(), attributes);
            self.input_tps.record(timing.input_tps(), attributes);
            self.total_tps.record(timing.total_tps(), attributes);
        }

        // Record request counters
        self.requests_total.add(1, attributes);
        if timing.success {
            self.requests_success.add(1, attributes);
        } else {
            let error_attrs = &[
                KeyValue::new("provider", timing.provider.clone()),
                KeyValue::new("model", timing.model.clone()),
                KeyValue::new("error_type", timing.error_type.clone().unwrap_or_default()),
            ];
            self.requests_failed.add(1, error_attrs);
        }

        // Record token counters
        self.tokens_input_total.add(timing.input_tokens, attributes);
        self.tokens_output_total
            .add(timing.output_tokens, attributes);

        // Emit tracing debug event with all metrics
        debug!(
            target: "sombrax_agentic_core::llm_metrics",
            provider = %timing.provider,
            model = %timing.model,
            latency_ms = timing.latency_ms,
            input_tokens = timing.input_tokens,
            output_tokens = timing.output_tokens,
            total_tokens = timing.total_tokens(),
            output_tps = timing.output_tps(),
            input_tps = timing.input_tps(),
            total_tps = timing.total_tps(),
            success = timing.success,
            error_type = timing.error_type.as_deref().unwrap_or(""),
            "llm_completion_metrics"
        );
    }
}

/// Classify a completion error into a string category for metrics
///
/// Returns a short string identifier for the error type, suitable for use
/// as an OpenTelemetry attribute value.
///
/// # Error Categories
///
/// - `"rate_limited"` - Provider rate limit exceeded
/// - `"auth_error"` - Authentication failure
/// - `"http_error"` - HTTP/network error
/// - `"json_error"` - JSON serialization/deserialization error
/// - `"provider_error"` - Provider-specific error
/// - `"invalid_request"` - Invalid request configuration
/// - `"cancelled"` - Request was cancelled
/// - `"hook_error"` - Hook processing error
/// - `"tool_error"` - Tool execution error
pub fn classify_completion_error(error: &CompletionError) -> String {
    match error {
        CompletionError::RateLimited { .. } => "rate_limited".to_string(),
        CompletionError::AuthenticationFailed => "auth_error".to_string(),
        CompletionError::HttpError(_) => "http_error".to_string(),
        CompletionError::JsonError(_) => "json_error".to_string(),
        CompletionError::ProviderError(_) => "provider_error".to_string(),
        CompletionError::InvalidRequest(_) => "invalid_request".to_string(),
        CompletionError::Cancelled => "cancelled".to_string(),
        CompletionError::HookError(_) => "hook_error".to_string(),
        CompletionError::ToolError(_) => "tool_error".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        // Just test that we can create metrics without panicking
        let _metrics = Metrics::global();
    }

    #[test]
    fn test_record_operations() {
        let metrics = Metrics::global();

        // These should not panic
        metrics.record_request_latency(Duration::from_millis(100), &[]);
        metrics.record_hook_duration("TestHook", "pre_completion", Duration::from_millis(5));
        metrics.record_tool_call("get_weather", true);
        metrics.record_tool_call("get_weather", false);
        metrics.record_completion_request("openai", "gpt-4", true);
        metrics.record_optimization("recency", 100, 50);
    }

    // =========================================================================
    // LLM Metrics Tests
    // =========================================================================

    #[test]
    fn test_completion_timing_tps_calculations() {
        let timing = CompletionTiming {
            latency_ms: 1000.0, // 1 second
            input_tokens: 100,
            output_tokens: 500,
            provider: "test".to_string(),
            model: "test-model".to_string(),
            success: true,
            error_type: None,
        };

        // 500 output tokens / 1 second = 500 tokens/sec
        assert!((timing.output_tps() - 500.0).abs() < 0.001);
        // 100 input tokens / 1 second = 100 tokens/sec
        assert!((timing.input_tps() - 100.0).abs() < 0.001);
        // 600 total tokens / 1 second = 600 tokens/sec
        assert!((timing.total_tps() - 600.0).abs() < 0.001);
        // Total tokens = 600
        assert_eq!(timing.total_tokens(), 600);
    }

    #[test]
    fn test_completion_timing_tps_with_fractional_latency() {
        let timing = CompletionTiming {
            latency_ms: 500.0, // 0.5 seconds
            input_tokens: 50,
            output_tokens: 250,
            provider: "test".to_string(),
            model: "test-model".to_string(),
            success: true,
            error_type: None,
        };

        // 250 output tokens / 0.5 seconds = 500 tokens/sec
        assert!((timing.output_tps() - 500.0).abs() < 0.001);
        // 50 input tokens / 0.5 seconds = 100 tokens/sec
        assert!((timing.input_tps() - 100.0).abs() < 0.001);
        // 300 total tokens / 0.5 seconds = 600 tokens/sec
        assert!((timing.total_tps() - 600.0).abs() < 0.001);
    }

    #[test]
    fn test_completion_timing_zero_latency() {
        let timing = CompletionTiming {
            latency_ms: 0.0,
            input_tokens: 100,
            output_tokens: 500,
            provider: "test".to_string(),
            model: "test-model".to_string(),
            success: true,
            error_type: None,
        };

        // Should return 0.0 to avoid division by zero
        assert_eq!(timing.output_tps(), 0.0);
        assert_eq!(timing.input_tps(), 0.0);
        assert_eq!(timing.total_tps(), 0.0);
    }

    #[test]
    fn test_completion_timing_zero_tokens() {
        let timing = CompletionTiming {
            latency_ms: 1000.0,
            input_tokens: 0,
            output_tokens: 0,
            provider: "test".to_string(),
            model: "test-model".to_string(),
            success: false,
            error_type: Some("rate_limited".to_string()),
        };

        assert_eq!(timing.output_tps(), 0.0);
        assert_eq!(timing.input_tps(), 0.0);
        assert_eq!(timing.total_tps(), 0.0);
        assert_eq!(timing.total_tokens(), 0);
    }

    #[test]
    fn test_llm_metrics_creation() {
        // Test that we can create LlmMetrics without panicking
        let _metrics = LlmMetrics::global();
    }

    #[test]
    fn test_llm_metrics_record_success() {
        let metrics = LlmMetrics::global();

        let timing = CompletionTiming {
            latency_ms: 1234.5,
            input_tokens: 150,
            output_tokens: 500,
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            success: true,
            error_type: None,
        };

        // Should not panic
        metrics.record(&timing);
    }

    #[test]
    fn test_llm_metrics_record_failure() {
        let metrics = LlmMetrics::global();

        let timing = CompletionTiming {
            latency_ms: 100.0,
            input_tokens: 0,
            output_tokens: 0,
            provider: "anthropic".to_string(),
            model: "claude-3-5-sonnet".to_string(),
            success: false,
            error_type: Some("rate_limited".to_string()),
        };

        // Should not panic
        metrics.record(&timing);
    }

    #[test]
    fn test_classify_completion_error_rate_limited() {
        let error = CompletionError::RateLimited {
            retry_after_secs: Some(60),
        };
        assert_eq!(classify_completion_error(&error), "rate_limited");
    }

    #[test]
    fn test_classify_completion_error_auth() {
        let error = CompletionError::AuthenticationFailed;
        assert_eq!(classify_completion_error(&error), "auth_error");
    }

    #[test]
    fn test_classify_completion_error_http() {
        let error = CompletionError::HttpError("connection refused".to_string());
        assert_eq!(classify_completion_error(&error), "http_error");
    }

    #[test]
    fn test_classify_completion_error_invalid_request() {
        let error = CompletionError::InvalidRequest("missing field".to_string());
        assert_eq!(classify_completion_error(&error), "invalid_request");
    }

    #[test]
    fn test_classify_completion_error_cancelled() {
        let error = CompletionError::Cancelled;
        assert_eq!(classify_completion_error(&error), "cancelled");
    }

    #[test]
    fn test_classify_completion_error_provider() {
        let error = CompletionError::ProviderError("model not found".to_string());
        assert_eq!(classify_completion_error(&error), "provider_error");
    }
}
