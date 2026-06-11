//! Agent module
//!
//! Provides the Agent struct and AgentBuilder for creating LLM agents with hooks.

mod completion;
mod registry;
pub mod wrapper;

pub use completion::AgentCompletion;
pub use registry::{AgentHandle, AgentRegistry, AgentRequest, AgentResponse};
pub use wrapper::AgentWrapper;

use crate::context::{ContextOptimizer, HookContext, OptimizationConfig};
use crate::error::{CompletionError, RegistryError};
use crate::hook::{Hook, HookChain, ToolCallDecision};
use crate::message::Message;
use crate::provider::{CompletionModel, CompletionRequest, CompletionResponse};
use crate::retry::{ResponseValidation, RetryConfig};
use crate::skill::{default_search_paths, SkillRegistry};
use crate::telemetry::Metrics;
use crate::tool::{ToolDefinition, ToolDyn};
use crate::tools::into_tool_dyn;
use crate::tools::skill::SkillTool;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::sleep;

/// Check if a JSON parse error indicates a truncated response.
///
/// These errors occur when the LLM response was cut off mid-stream,
/// resulting in incomplete JSON. Immediate retry from the last state
/// is appropriate since the issue is transient.
fn is_truncated_json_error(err: &serde_json::Error) -> bool {
    let err_str = err.to_string().to_lowercase();
    // Common patterns for truncated/incomplete JSON
    err_str.contains("eof while parsing")
        || err_str.contains("unexpected end of input")
        || err_str.contains("unexpected end of json")
        || err_str.contains("unexpected end of file")
        || err_str.contains("premature end")
}

/// Check if an error message indicates a tool argument error.
///
/// These errors suggest the LLM provided wrong/invalid arguments to a tool,
/// and retrying from before the tool call might help.
fn is_tool_argument_error(err: &str) -> bool {
    let err_lower = err.to_lowercase();
    // Common patterns for argument/parameter errors
    err_lower.contains("missing field")
        || err_lower.contains("invalid type")
        || err_lower.contains("invalid value")
        || err_lower.contains("expected ")
        || err_lower.contains("unknown field")
        || err_lower.contains("invalid argument")
        || err_lower.contains("parameter")
        || err_lower.contains("required field")
        || err_lower.contains("deserialization")
        || err_lower.contains("validation failed")
        || err_lower.contains("invalid pattern")
        || err_lower.contains("invalid regex")
        || err_lower.contains("invalid path")
        || err_lower.contains("file not found")
        || err_lower.contains("no such file")
}

/// Execution statistics for an agent prompt execution.
///
/// Tracks cumulative metrics across all LLM calls and tool executions
/// within a single `prompt()` or `prompt_with_history()` invocation.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Input tokens consumed (cumulative across all LLM calls)
    pub input_tokens: u64,
    /// Output tokens generated (cumulative across all LLM calls)
    pub output_tokens: u64,
    /// Cache read tokens (tokens served from cache, cumulative)
    pub cache_read_tokens: u64,
    /// Cache creation tokens (tokens written to cache, cumulative)
    pub cache_creation_tokens: u64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of tool calls made
    pub tool_calls: usize,
    /// Number of LLM completions (turns in the agent loop)
    pub message_count: usize,
    /// Number of retries (transient error retries)
    pub retries_count: usize,
    /// Number of tool errors encountered
    pub tool_error_count: usize,
}

impl ExecutionStats {
    /// Accumulate stats from another ExecutionStats instance.
    ///
    /// This is useful for aggregating stats from multiple agent executions,
    /// such as in parallel audit pipelines.
    pub fn accumulate(&mut self, other: &ExecutionStats) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
        self.execution_time_ms += other.execution_time_ms;
        self.tool_calls += other.tool_calls;
        self.message_count += other.message_count;
        self.retries_count += other.retries_count;
        self.tool_error_count += other.tool_error_count;
    }

    /// Get total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Get total cached tokens (read + creation).
    pub fn total_cached_tokens(&self) -> u64 {
        self.cache_read_tokens + self.cache_creation_tokens
    }

    /// Format execution stats as markdown to append after investigation block.
    pub fn format_markdown(&self) -> String {
        // Only show cache rows if there are cached tokens
        let cache_rows = if self.cache_read_tokens > 0 || self.cache_creation_tokens > 0 {
            format!(
                "| Cache Read Tokens | {} |\n| Cache Creation Tokens | {} |\n",
                self.cache_read_tokens, self.cache_creation_tokens
            )
        } else {
            String::new()
        };

        format!(
            r#"
---

<details>
<summary>📊 Execution Statistics</summary>

| Metric | Value |
|--------|-------|
| Input Tokens | {} |
| Output Tokens | {} |
{}| Total Tokens | {} |
| Execution Time | {:.2}s |
| Tool Calls | {} |
| Dialog Messages | {} |

</details>
"#,
            self.input_tokens,
            self.output_tokens,
            cache_rows,
            self.total_tokens(),
            self.execution_time_ms as f64 / 1000.0,
            self.tool_calls,
            self.message_count
        )
    }

    /// Format aggregated pipeline stats as markdown summary.
    pub fn format_pipeline_summary(
        &self,
        task_count: usize,
        successful_tasks: usize,
        failed_tasks: usize,
    ) -> String {
        let avg_tokens = if successful_tasks > 0 {
            self.total_tokens() / successful_tasks as u64
        } else {
            0
        };
        let avg_time_ms = if successful_tasks > 0 {
            self.execution_time_ms / successful_tasks as u64
        } else {
            0
        };
        let avg_tool_calls = if successful_tasks > 0 {
            self.tool_calls / successful_tasks
        } else {
            0
        };

        // Only show cache section if there are cached tokens
        let cache_section = if self.cache_read_tokens > 0 || self.cache_creation_tokens > 0 {
            format!(
                r#"
### Cache Usage
| Metric | Total | Average per Task |
|--------|-------|------------------|
| Cache Read Tokens | {} | {} |
| Cache Creation Tokens | {} | {} |
"#,
                self.cache_read_tokens,
                if successful_tasks > 0 {
                    self.cache_read_tokens / successful_tasks as u64
                } else {
                    0
                },
                self.cache_creation_tokens,
                if successful_tasks > 0 {
                    self.cache_creation_tokens / successful_tasks as u64
                } else {
                    0
                }
            )
        } else {
            String::new()
        };

        format!(
            r#"
---

## 📊 Pipeline Execution Summary

<details open>
<summary>Aggregated Statistics for All Investigations</summary>

### Task Summary
| Metric | Value |
|--------|-------|
| Total Tasks | {} |
| Successful | {} |
| Failed | {} |

### Token Usage
| Metric | Total | Average per Task |
|--------|-------|------------------|
| Input Tokens | {} | {} |
| Output Tokens | {} | {} |
| **Total Tokens** | **{}** | **{}** |
{}
### Performance
| Metric | Total | Average per Task |
|--------|-------|------------------|
| Execution Time | {:.2}s | {:.2}s |
| Tool Calls | {} | {} |
| Dialog Messages | {} | {} |

</details>
"#,
            task_count,
            successful_tasks,
            failed_tasks,
            self.input_tokens,
            if successful_tasks > 0 {
                self.input_tokens / successful_tasks as u64
            } else {
                0
            },
            self.output_tokens,
            if successful_tasks > 0 {
                self.output_tokens / successful_tasks as u64
            } else {
                0
            },
            self.total_tokens(),
            avg_tokens,
            cache_section,
            self.execution_time_ms as f64 / 1000.0,
            avg_time_ms as f64 / 1000.0,
            self.tool_calls,
            avg_tool_calls,
            self.message_count,
            if successful_tasks > 0 {
                self.message_count / successful_tasks
            } else {
                0
            }
        )
    }
}

/// Response from an agent prompt execution.
///
/// Contains both the LLM completion response, execution statistics, and the full message history.
#[derive(Debug, Clone)]
pub struct PromptResponse<R> {
    /// The completion response from the LLM
    pub response: CompletionResponse<R>,
    /// Execution statistics for the entire agent loop
    pub stats: ExecutionStats,
    /// Full message history from the execution (for multi-stage pipelines)
    pub messages: Vec<Message>,
}

impl<R> PromptResponse<R> {
    /// Create a new prompt response
    pub fn new(
        response: CompletionResponse<R>,
        stats: ExecutionStats,
        messages: Vec<Message>,
    ) -> Self {
        Self {
            response,
            stats,
            messages,
        }
    }

    /// Get the text content of the response
    pub fn content(&self) -> String {
        self.response.content()
    }

    /// Check if the response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.response.has_tool_calls()
    }

    /// Get tool calls from the response
    pub fn tool_calls(&self) -> Vec<&crate::message::ToolCall> {
        self.response.tool_calls()
    }

    /// Get the underlying message
    pub fn message(&self) -> &Message {
        &self.response.message
    }

    /// Get token usage from the final response
    pub fn usage(&self) -> &crate::provider::Usage {
        &self.response.usage
    }

    /// Get the full message history from the execution
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

/// Checkpoint for resuming agent execution after validation failure.
///
/// When response validation fails (e.g., response too short), the agent can
/// resume from this checkpoint instead of re-executing all tool calls.
#[derive(Debug, Clone)]
struct ExecutionCheckpoint {
    /// Messages accumulated up to this point (including tool results)
    messages: Vec<Message>,
    /// Tool definitions to use
    tools: Vec<ToolDefinition>,
    /// Number of tool errors encountered so far
    tool_error_count: usize,
    /// Current turn count
    turn_count: usize,
    /// Rollback checkpoint: state BEFORE the last tool call batch
    /// Used when we want to retry from before a failed tool call
    rollback_messages: Option<Vec<Message>>,
    /// Whether to rollback on next validation failure (set when tool args error detected)
    should_rollback: bool,
}

/// An LLM agent with hooks, tools, and context optimization
pub struct Agent<M: CompletionModel> {
    /// Agent name (optional)
    name: Option<String>,
    /// LLM model
    model: M,
    /// System preamble
    preamble: Option<String>,
    /// Hook chain
    hook_chain: HookChain,
    /// Available tools
    tools: Vec<Arc<dyn ToolDyn>>,
    /// Context optimizer
    optimizer: Option<Arc<dyn ContextOptimizer>>,
    /// Optimization config
    optimization_config: OptimizationConfig,
    /// Advertised capabilities
    capabilities: Vec<String>,
    /// Maximum turns for tool execution loop (None = unlimited)
    max_turns: Option<usize>,
    /// Retry configuration for handling transient errors (default: 10 retries)
    retry_config: RetryConfig,
    /// Response validation configuration (optional)
    response_validation: Option<ResponseValidation>,
    /// Skill registry (optional)
    skill_registry: Option<Arc<SkillRegistry>>,
}

impl<M: CompletionModel> Agent<M> {
    /// Create a new agent builder
    pub fn builder(model: M) -> AgentBuilder<M> {
        AgentBuilder::new(model)
    }

    /// Get the agent name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the preamble
    pub fn preamble(&self) -> Option<&str> {
        self.preamble.as_deref()
    }

    /// Get the capabilities
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Check if the agent has tools
    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    /// Get tool definitions
    pub async fn tool_definitions(&self, prompt: &str) -> Vec<ToolDefinition> {
        let mut defs = Vec::with_capacity(self.tools.len());
        for tool in &self.tools {
            defs.push(tool.definition(prompt.to_string()).await);
        }
        defs
    }

    /// Find a tool by name
    pub fn find_tool(&self, name: &str) -> Option<&Arc<dyn ToolDyn>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    /// Execute a completion request with retry logic for transient errors.
    ///
    /// This method will retry the completion request on transient errors
    /// (rate limits, network issues) using exponential backoff.
    ///
    /// Returns a tuple of (response, retry_count) where retry_count is the
    /// number of retries that were attempted before success.
    async fn completion_with_retry(
        &self,
        request: CompletionRequest,
        metrics: &Metrics,
    ) -> Result<(CompletionResponse<M::Response>, usize), CompletionError> {
        let mut retry_count = 0;
        let mut had_retryable_error = false;
        let mut step_back_attempted = false;
        let mut request = request;

        loop {
            match self.model.completion(request.clone()).await {
                Ok(response) => {
                    metrics.record_completion_request(
                        self.model.provider(),
                        self.model.model_id(),
                        true,
                    );
                    return Ok((response, retry_count));
                }
                Err(e) => {
                    // Check if error is retryable
                    if !e.is_retryable() || !self.retry_config.retries_enabled() {
                        // Step-back recovery: if we previously had a retryable error (e.g. 502)
                        // and now get a non-retryable error (e.g. 400), the provider may have
                        // corrupted state from the failed request. Try removing the last
                        // assistant+user turn and retrying once.
                        if had_retryable_error
                            && !step_back_attempted
                            && request.messages.len() >= 4
                        {
                            let len = request.messages.len();
                            let last_is_user = request.messages[len - 1].is_user();
                            let second_last_is_assistant = request.messages[len - 2].is_assistant();

                            if last_is_user && second_last_is_assistant {
                                step_back_attempted = true;
                                metrics.record_completion_request(
                                    self.model.provider(),
                                    self.model.model_id(),
                                    false,
                                );
                                tracing::warn!(
                                    "Non-retryable error after prior retryable error, \
                                     stepping back one turn ({} messages -> {}) and retrying: {}",
                                    len,
                                    len - 2,
                                    e
                                );
                                request.messages.truncate(len - 2);
                                retry_count += 1;

                                let delay = self.retry_config.delay_for_attempt(retry_count);
                                sleep(delay).await;
                                continue;
                            }
                        }

                        metrics.record_completion_request(
                            self.model.provider(),
                            self.model.model_id(),
                            false,
                        );
                        tracing::debug!("Completion failed with non-retryable error: {}", e);
                        return Err(e);
                    }

                    had_retryable_error = true;

                    // Check if we've exhausted retries
                    if retry_count >= self.retry_config.max_retries {
                        metrics.record_completion_request(
                            self.model.provider(),
                            self.model.model_id(),
                            false,
                        );
                        tracing::warn!("Completion failed after {} retries: {}", retry_count, e);
                        return Err(e);
                    }

                    // Get delay - use provider's suggested delay if available
                    let delay = e
                        .retry_after_secs()
                        .map(std::time::Duration::from_secs)
                        .unwrap_or_else(|| self.retry_config.delay_for_attempt(retry_count));

                    retry_count += 1;
                    tracing::warn!(
                        "Completion failed with retryable error, retry {}/{} after {:?}: {}",
                        retry_count,
                        self.retry_config.max_retries,
                        delay,
                        e
                    );

                    sleep(delay).await;
                }
            }
        }
    }

    /// Send a prompt and get a response with execution statistics
    pub async fn prompt(
        &self,
        message: impl Into<Message>,
    ) -> Result<PromptResponse<M::Response>, CompletionError> {
        self.prompt_with_history(message, &[]).await
    }

    /// Send a prompt with conversation history and get response with execution statistics
    pub async fn prompt_with_history(
        &self,
        message: impl Into<Message>,
        history: &[Message],
    ) -> Result<PromptResponse<M::Response>, CompletionError> {
        let message = message.into();
        let mut validation_retry_count = 0;
        let mut checkpoint: Option<ExecutionCheckpoint> = None;

        // Track cumulative stats across validation retries
        let mut cumulative_stats = ExecutionStats::default();

        // Validation retry loop
        loop {
            let (response, new_checkpoint, stats, final_messages) = self
                .execute_prompt_inner(&message, history, checkpoint.clone())
                .await?;

            // Accumulate stats from this attempt
            cumulative_stats.input_tokens += stats.input_tokens;
            cumulative_stats.output_tokens += stats.output_tokens;
            cumulative_stats.cache_read_tokens += stats.cache_read_tokens;
            cumulative_stats.cache_creation_tokens += stats.cache_creation_tokens;
            cumulative_stats.execution_time_ms += stats.execution_time_ms;
            cumulative_stats.tool_calls += stats.tool_calls;
            cumulative_stats.message_count += stats.message_count;
            cumulative_stats.retries_count += stats.retries_count;
            cumulative_stats.tool_error_count += stats.tool_error_count;

            // Check response validation if configured
            // Skip min_length validation if response has tool calls (tool call responses may have empty content)
            if let Some(ref validation) = self.response_validation {
                let result = if response.has_tool_calls() {
                    // Only check tool errors, skip length check for tool call responses
                    validation.validate_skip_length(stats.tool_error_count)
                } else {
                    let content = response.content();
                    validation.validate(&content, stats.tool_error_count)
                };

                if result.should_retry()
                    && validation_retry_count < validation.max_validation_retries
                {
                    validation_retry_count += 1;
                    let delay = self.retry_config.delay_for_attempt(validation_retry_count);

                    // Check if we should rollback to before the failed tool call
                    let (use_checkpoint, msg_count) = if let Some(ref cp) = new_checkpoint {
                        if cp.should_rollback {
                            if let Some(ref rollback_msgs) = cp.rollback_messages {
                                // Create a rollback checkpoint with the pre-tool-call state
                                let rollback_cp = ExecutionCheckpoint {
                                    messages: rollback_msgs.clone(),
                                    tools: cp.tools.clone(),
                                    tool_error_count: 0, // Reset error count for fresh retry
                                    turn_count: cp.turn_count.saturating_sub(1), // Go back one turn
                                    rollback_messages: None,
                                    should_rollback: false,
                                };
                                tracing::info!(
                                    "Rewinding agent: rolling back to {} messages before failed tool call (turn {} -> {})",
                                    rollback_msgs.len(),
                                    cp.turn_count,
                                    cp.turn_count.saturating_sub(1)
                                );
                                tracing::warn!(
                                    "Response validation failed ({}), retry {}/{} after {:?} (ROLLBACK to {} messages, before failed tool call)",
                                    result,
                                    validation_retry_count,
                                    validation.max_validation_retries,
                                    delay,
                                    rollback_msgs.len()
                                );
                                (Some(rollback_cp), rollback_msgs.len())
                            } else {
                                (new_checkpoint.clone(), cp.messages.len())
                            }
                        } else {
                            (new_checkpoint.clone(), cp.messages.len())
                        }
                    } else {
                        (None, 0)
                    };

                    if !new_checkpoint
                        .as_ref()
                        .map(|c| c.should_rollback)
                        .unwrap_or(false)
                    {
                        tracing::warn!(
                            "Response validation failed ({}), retry {}/{} after {:?} (resuming from checkpoint with {} messages)",
                            result,
                            validation_retry_count,
                            validation.max_validation_retries,
                            delay,
                            msg_count
                        );
                    }

                    // Save checkpoint for next retry
                    checkpoint = use_checkpoint;
                    sleep(delay).await;
                    continue;
                }

                // Log if validation still fails but we've exhausted retries
                if result.should_retry() {
                    tracing::warn!(
                        "Response validation failed ({}) after {} retries, returning anyway",
                        result,
                        validation_retry_count
                    );
                }
            }

            return Ok(PromptResponse::new(
                response,
                cumulative_stats,
                final_messages,
            ));
        }
    }

    /// Internal method that executes the prompt and returns the response.
    ///
    /// If a checkpoint is provided, resumes from that point instead of starting fresh.
    /// Returns the response, checkpoint for potential retry, execution stats, and final message history.
    async fn execute_prompt_inner(
        &self,
        message: &Message,
        history: &[Message],
        checkpoint: Option<ExecutionCheckpoint>,
    ) -> Result<
        (
            CompletionResponse<M::Response>,
            Option<ExecutionCheckpoint>,
            ExecutionStats,
            Vec<Message>,
        ),
        CompletionError,
    > {
        let request_start = Instant::now();
        let metrics = Metrics::global();
        let mut ctx = HookContext::new_with_uuid();

        // Initialize execution stats
        let mut stats = ExecutionStats::default();

        // Resume from checkpoint or start fresh
        let (mut messages, tools, mut turn_count) = if let Some(cp) = checkpoint {
            tracing::info!(
                "Resuming agent from checkpoint: {} messages, {} tool errors, turn {}",
                cp.messages.len(),
                cp.tool_error_count,
                cp.turn_count
            );
            stats.tool_error_count = cp.tool_error_count;
            (cp.messages, cp.tools, cp.turn_count)
        } else {
            // Execute pre-completion hooks
            let message = self
                .hook_chain
                .execute_pre_completion(message.clone(), history, &mut ctx)
                .await?;

            // Get tool definitions with filter hooks applied
            let tools = if self.has_tools() {
                let defs = self.tool_definitions(&message.text()).await;
                self.hook_chain.execute_filter_tools(defs, &mut ctx).await?
            } else {
                vec![]
            };

            // Optimize context if needed
            let mut messages = history.to_vec();
            messages.push(message);

            if let Some(optimizer) = &self.optimizer {
                messages = optimizer
                    .optimize(messages, &self.optimization_config)
                    .await;
            }

            (messages, tools, 0)
        };

        // Build completion request
        let request = CompletionRequest {
            preamble: self.preamble.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            temperature: None,
            max_tokens: None,
            additional_params: None,
        };

        // Send to model with retry logic for transient errors
        let (mut response, retry_count) = self.completion_with_retry(request, &metrics).await?;

        // Remap tool call IDs to simple incremental format (e.g., tool_call_1, tool_call_2)
        response
            .message
            .remap_tool_call_ids(|_old_id| ctx.next_tool_call_id());

        // Accumulate stats from first completion
        stats.message_count += 1;
        stats.retries_count += retry_count;
        stats.input_tokens += response.usage.input_tokens;
        stats.output_tokens += response.usage.output_tokens;
        stats.cache_read_tokens += response.usage.cache_read_tokens;
        stats.cache_creation_tokens += response.usage.cache_creation_tokens;

        // Notify hooks of assistant message (for display/logging)
        self.hook_chain
            .execute_on_assistant_message(&response.message, &mut ctx)
            .await?;

        // Track if we encounter tool argument errors (for rollback)
        let mut had_tool_arg_error = false;
        let mut rollback_messages: Option<Vec<Message>> = None;
        // Track consecutive truncation retries to avoid infinite loops
        let mut truncation_retry_count: usize = 0;
        const MAX_TRUNCATION_RETRIES: usize = 3;

        // Tool execution loop: continue while the response contains tool calls
        while response.has_tool_calls() {
            // Check max_turns limit
            if let Some(max) = self.max_turns {
                if turn_count >= max {
                    tracing::debug!(
                        "Agent reached max_turns limit ({}) with pending tool calls",
                        max
                    );
                    break;
                }
            }
            turn_count += 1;

            // Save rollback checkpoint BEFORE processing this batch of tool calls
            // This allows us to "step back" if tool arguments are wrong
            rollback_messages = Some(messages.clone());

            // Proactive truncation check: detect truncated responses via finish_reason
            // BEFORE attempting to parse tool arguments (avoids wasted effort on broken JSON)
            if response.is_truncated() && truncation_retry_count < MAX_TRUNCATION_RETRIES {
                truncation_retry_count += 1;
                let reason = response.finish_reason.as_deref().unwrap_or("none");
                tracing::warn!(
                    finish_reason = reason,
                    "Response truncated (finish_reason='{}'), retrying immediately (attempt {}/{})",
                    reason,
                    truncation_retry_count,
                    MAX_TRUNCATION_RETRIES
                );

                // Brief delay before retry
                let delay = self.retry_config.delay_for_attempt(truncation_retry_count);
                sleep(delay).await;

                // Re-send the completion request from the current state
                let request = CompletionRequest {
                    preamble: self.preamble.clone(),
                    messages: messages.clone(),
                    tools: tools.clone(),
                    temperature: None,
                    max_tokens: None,
                    additional_params: None,
                };

                let (mut new_response, retry_count) =
                    self.completion_with_retry(request, &metrics).await?;

                // Remap tool call IDs
                new_response
                    .message
                    .remap_tool_call_ids(|_old_id| ctx.next_tool_call_id());

                response = new_response;
                stats.retries_count += retry_count + 1;
                turn_count -= 1; // Don't count the truncated attempt as a turn
                continue;
            }

            // Append the assistant message with tool calls to history
            // Include reasoning content if present (some models return tool calls inside reasoning)
            let message_for_history = if let Some(ref reasoning) = response.reasoning_content {
                // Set reasoning on the message so providers can send it back properly
                response.message.with_reasoning(reasoning)
            } else {
                response.message.clone()
            };
            messages.push(message_for_history);

            // Track if this batch has truncation errors (for immediate retry)
            let mut has_truncation_error = false;

            // Process each tool call
            for tool_call in response.tool_calls() {
                // Count this tool call
                stats.tool_calls += 1;

                let tool_name = &tool_call.function.name;
                let tool_args_str = &tool_call.function.arguments;

                // Parse arguments for hooks - detect JSON parse errors
                let args_parse_result: Result<serde_json::Value, _> =
                    serde_json::from_str(tool_args_str);

                let (args, json_parse_error) = match args_parse_result {
                    Ok(v) => (v, false),
                    Err(e) => {
                        // Check if this is a truncation error (immediate retry candidate)
                        if is_truncated_json_error(&e) {
                            let reason = response.finish_reason.as_deref().unwrap_or("none");
                            tracing::warn!(
                                finish_reason = reason,
                                "Tool '{}' has truncated JSON arguments (finish_reason='{}', will retry): {} (args: {})",
                                tool_name,
                                reason,
                                e,
                                tool_args_str
                            );
                            has_truncation_error = true;
                        } else {
                            tracing::warn!(
                                "Tool '{}' received invalid JSON arguments: {} (args: {})",
                                tool_name,
                                e,
                                tool_args_str
                            );
                        }
                        had_tool_arg_error = true;
                        (serde_json::json!({}), true)
                    }
                };

                // Execute pre-tool-call hooks
                let decision = self
                    .hook_chain
                    .execute_pre_tool_call(tool_name, args, &mut ctx)
                    .await?;

                let tool_result = match decision {
                    ToolCallDecision::Block(reason) => {
                        // Return the block reason as the tool result
                        format!("Tool call blocked: {}", reason)
                    }
                    ToolCallDecision::Proceed(modified_args) => {
                        if json_parse_error {
                            // Don't even try to call the tool if args couldn't be parsed
                            stats.tool_error_count += 1;
                            format!(
                                "Tool argument error: Invalid JSON in arguments for '{}'. Raw args: {}",
                                tool_name, tool_args_str
                            )
                        } else {
                            // Find and execute the tool
                            let result = if let Some(tool) = self.find_tool(tool_name) {
                                // Convert modified args back to string for the tool call
                                let args_str = serde_json::to_string(&modified_args)?;
                                match tool.call(args_str).await {
                                    Ok(output) => output,
                                    Err(e) => {
                                        let err_str = e.to_string();
                                        stats.tool_error_count += 1;
                                        // Detect common tool argument errors
                                        if is_tool_argument_error(&err_str) {
                                            tracing::warn!(
                                                "Tool '{}' argument error detected: {}",
                                                tool_name,
                                                err_str
                                            );
                                            had_tool_arg_error = true;
                                        }
                                        format!("Tool execution error: {}", err_str)
                                    }
                                }
                            } else {
                                // Tool not found error
                                stats.tool_error_count += 1;
                                format!("Tool '{}' not found", tool_name)
                            };

                            // Execute post-tool-call hooks
                            self.hook_chain
                                .execute_post_tool_call(tool_name, result, &mut ctx)
                                .await?
                        }
                    }
                };

                // Add tool result as a user message
                let tool_result_message = Message::tool_result(&tool_call.id, tool_result);
                messages.push(tool_result_message);
            }

            // Handle truncation errors with immediate retry
            // This happens when the LLM response was cut off mid-stream
            if has_truncation_error && truncation_retry_count < MAX_TRUNCATION_RETRIES {
                truncation_retry_count += 1;

                // Restore to rollback state (before this truncated response)
                if let Some(ref rollback_msgs) = rollback_messages {
                    tracing::info!(
                        "Truncated JSON detected, rolling back to {} messages and retrying (attempt {}/{})",
                        rollback_msgs.len(),
                        truncation_retry_count,
                        MAX_TRUNCATION_RETRIES
                    );
                    messages = rollback_msgs.clone();

                    // Brief delay before retry
                    let delay = self.retry_config.delay_for_attempt(truncation_retry_count);
                    sleep(delay).await;

                    // Re-send the completion request from the rollback state
                    let request = CompletionRequest {
                        preamble: self.preamble.clone(),
                        messages: messages.clone(),
                        tools: tools.clone(),
                        temperature: None,
                        max_tokens: None,
                        additional_params: None,
                    };

                    let (mut new_response, retry_count) =
                        self.completion_with_retry(request, &metrics).await?;

                    // Remap tool call IDs
                    new_response
                        .message
                        .remap_tool_call_ids(|_old_id| ctx.next_tool_call_id());

                    response = new_response;
                    stats.retries_count += retry_count + 1; // +1 for the truncation retry

                    // Continue the tool execution loop with the new response
                    continue;
                }
            } else if has_truncation_error {
                tracing::warn!(
                    "Truncated JSON detected but max retries ({}) exceeded, proceeding with error response",
                    MAX_TRUNCATION_RETRIES
                );
            }

            // Reset truncation retry counter on successful (non-truncated) response
            if !has_truncation_error {
                truncation_retry_count = 0;
            }

            // Re-optimize context if needed
            if let Some(optimizer) = &self.optimizer {
                messages = optimizer
                    .optimize(messages, &self.optimization_config)
                    .await;
            }

            // Build the next completion request
            let request = CompletionRequest {
                preamble: self.preamble.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                temperature: None,
                max_tokens: None,
                additional_params: None,
            };

            // Send to model with retry logic for transient errors
            let (mut new_response, retry_count) =
                self.completion_with_retry(request, &metrics).await?;

            // Remap tool call IDs to simple incremental format (e.g., tool_call_1, tool_call_2)
            new_response
                .message
                .remap_tool_call_ids(|_old_id| ctx.next_tool_call_id());

            response = new_response;

            // Accumulate stats from this completion
            stats.message_count += 1;
            stats.retries_count += retry_count;
            stats.input_tokens += response.usage.input_tokens;
            stats.output_tokens += response.usage.output_tokens;
            stats.cache_read_tokens += response.usage.cache_read_tokens;
            stats.cache_creation_tokens += response.usage.cache_creation_tokens;

            // Notify hooks of assistant message (for display/logging)
            self.hook_chain
                .execute_on_assistant_message(&response.message, &mut ctx)
                .await?;
        }

        // Execute post-completion hooks on the final response
        let response = self
            .hook_chain
            .execute_post_completion(response, &mut ctx)
            .await?;

        // Record total request latency (FR-021)
        let elapsed = request_start.elapsed();
        metrics.record_request_latency(elapsed, &[]);

        // Finalize execution stats
        stats.execution_time_ms = elapsed.as_millis() as u64;

        // Create checkpoint for potential retry (resume from current state)
        let checkpoint = ExecutionCheckpoint {
            messages: messages.clone(),
            tools,
            tool_error_count: stats.tool_error_count,
            turn_count,
            rollback_messages,
            should_rollback: had_tool_arg_error,
        };

        Ok((response, Some(checkpoint), stats, messages))
    }

    /// Create a completion builder for more control
    pub fn completion(&self, message: impl Into<Message>) -> AgentCompletion<'_, M> {
        AgentCompletion::new(self, message.into())
    }
}

impl<M: CompletionModel> Clone for Agent<M> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            model: self.model.clone(),
            preamble: self.preamble.clone(),
            hook_chain: self.hook_chain.clone(),
            tools: self.tools.clone(),
            optimizer: self.optimizer.clone(),
            optimization_config: self.optimization_config.clone(),
            capabilities: self.capabilities.clone(),
            max_turns: self.max_turns,
            retry_config: self.retry_config.clone(),
            response_validation: self.response_validation.clone(),
            skill_registry: self.skill_registry.clone(),
        }
    }
}

/// Builder for creating agents
pub struct AgentBuilder<M: CompletionModel> {
    name: Option<String>,
    model: M,
    preamble: Option<String>,
    hook_chain: HookChain,
    tools: Vec<Arc<dyn ToolDyn>>,
    optimizer: Option<Arc<dyn ContextOptimizer>>,
    optimization_config: OptimizationConfig,
    capabilities: Vec<String>,
    max_turns: Option<usize>,
    retry_config: RetryConfig,
    response_validation: Option<ResponseValidation>,
    skill_registry: Option<Arc<SkillRegistry>>,
    enable_skills: bool,
}

impl<M: CompletionModel> AgentBuilder<M> {
    /// Create a new agent builder with the given model
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            model,
            preamble: None,
            hook_chain: HookChain::new(),
            tools: Vec::new(),
            optimizer: None,
            optimization_config: OptimizationConfig::default(),
            capabilities: Vec::new(),
            max_turns: None,
            retry_config: RetryConfig::default(),
            response_validation: None,
            skill_registry: None,
            enable_skills: false,
        }
    }

    /// Set the agent name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the system preamble
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Add a hook to the chain
    pub fn hook<H: Hook>(mut self, hook: H) -> Self {
        self.hook_chain.add(hook);
        self
    }

    /// Add a tool
    pub fn tool(mut self, tool: Arc<dyn ToolDyn>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools
    pub fn tools(mut self, tools: Vec<Arc<dyn ToolDyn>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add MCP tools from a source
    pub async fn mcp_tools(mut self, source: crate::tool::McpToolSource) -> Self {
        let mcp_tools = source.as_tools().await;
        for tool in mcp_tools {
            self.tools.push(Arc::new(tool));
        }
        self
    }

    /// Set the context optimizer
    pub fn context_optimizer<O: ContextOptimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Some(Arc::new(optimizer));
        self
    }

    /// Set a context optimizer from an existing Arc.
    ///
    /// This is useful when you already have an `Arc<dyn ContextOptimizer>`,
    /// such as when passing through options structs.
    pub fn context_optimizer_arc(mut self, optimizer: Arc<dyn ContextOptimizer>) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set the optimization config
    pub fn optimization_config(mut self, config: OptimizationConfig) -> Self {
        self.optimization_config = config;
        self
    }

    /// Set the advertised capabilities
    pub fn capabilities(mut self, capabilities: Vec<impl Into<String>>) -> Self {
        self.capabilities = capabilities.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Add a capability
    pub fn capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Set the maximum number of turns for the tool execution loop
    ///
    /// When set, the agent will stop executing tools after this many turns,
    /// even if the model is still producing tool calls. This prevents infinite
    /// loops and provides a safety limit for long-running agents.
    ///
    /// A "turn" is one iteration of the tool execution loop (processing all
    /// tool calls from one model response and sending results back).
    pub fn max_turns(mut self, max: usize) -> Self {
        self.max_turns = Some(max);
        self
    }

    /// Set the retry configuration for handling transient errors.
    ///
    /// By default, agents will retry up to 5 times with exponential backoff
    /// when encountering transient errors like rate limits or network issues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use sombrax_agentic_core::retry::RetryConfig;
    ///
    /// // Custom retry configuration with 3 retries
    /// let agent = Agent::builder(model)
    ///     .retry_config(RetryConfig::with_max_retries(3))
    ///     .build();
    ///
    /// // Disable retries entirely
    /// let agent = Agent::builder(model)
    ///     .retry_config(RetryConfig::no_retries())
    ///     .build();
    /// ```
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Set response validation to retry on empty or low-quality responses.
    ///
    /// When enabled, the agent will retry the entire execution if the response
    /// fails validation (e.g., too short, too many tool errors).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use sombrax_agentic_core::retry::ResponseValidation;
    ///
    /// // Retry if response is less than 100 characters
    /// let agent = Agent::builder(model)
    ///     .response_validation(ResponseValidation::min_length(100))
    ///     .build();
    ///
    /// // Retry if response < 100 chars OR more than 3 tool errors
    /// let agent = Agent::builder(model)
    ///     .response_validation(ResponseValidation::new(100, 3))
    ///     .build();
    /// ```
    pub fn response_validation(mut self, validation: ResponseValidation) -> Self {
        self.response_validation = Some(validation);
        self
    }

    /// Enable skills with auto-discovery from default paths
    ///
    /// This will:
    /// - Discover skills from `./.sac/skills/` and `~/.sac/skills/`
    /// - Add skill metadata to the system preamble
    /// - Auto-register the SkillTool for loading skills
    pub fn with_skills(mut self) -> Self {
        self.enable_skills = true;
        self
    }

    /// Use a custom skill registry
    ///
    /// This will:
    /// - Use the provided registry instead of auto-discovery
    /// - Add skill metadata to the system preamble
    /// - Auto-register the SkillTool for loading skills
    pub fn skill_registry(mut self, registry: Arc<SkillRegistry>) -> Self {
        self.skill_registry = Some(registry);
        self.enable_skills = true;
        self
    }

    /// Build the agent
    pub fn build(self) -> Agent<M> {
        let mut tools = self.tools;
        let mut preamble = self.preamble;

        // Handle skill integration
        let skill_registry = if self.enable_skills {
            let registry = if let Some(reg) = self.skill_registry {
                reg
            } else {
                // Try to discover skills, but don't panic if no runtime
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        // We have a runtime - use it
                        Arc::new(tokio::task::block_in_place(|| {
                            handle.block_on(async {
                                SkillRegistry::discover(default_search_paths())
                                    .await
                                    .unwrap_or_else(|_| SkillRegistry::new())
                            })
                        }))
                    }
                    Err(_) => {
                        // No runtime available - create empty registry
                        // Skills can still be added via skill_registry() method
                        Arc::new(SkillRegistry::new())
                    }
                }
            };

            // Only add SkillTool and modify preamble if registry is non-empty
            if !registry.is_empty() {
                tools.push(into_tool_dyn(SkillTool::new(Arc::clone(&registry))));

                let skill_metadata = registry
                    .all_metadata()
                    .iter()
                    .map(|meta| format!("- {}: {}", meta.name, meta.description))
                    .collect::<Vec<_>>()
                    .join("\n");

                let skill_section = format!("\n\nAvailable Skills:\n{}", skill_metadata);

                preamble = Some(match preamble {
                    Some(existing) => format!("{}{}", existing, skill_section),
                    None => skill_section,
                });
            }

            Some(registry)
        } else {
            self.skill_registry
        };

        Agent {
            name: self.name,
            model: self.model,
            preamble,
            hook_chain: self.hook_chain,
            tools,
            optimizer: self.optimizer,
            optimization_config: self.optimization_config,
            capabilities: self.capabilities,
            max_turns: self.max_turns,
            retry_config: self.retry_config,
            response_validation: self.response_validation,
            skill_registry,
        }
    }
}

/// Implement AgentHandle for Agent to support registry
impl<M: CompletionModel> AgentHandle for Agent<M> {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("unnamed")
    }

    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    fn invoke<'a>(
        &'a self,
        request: AgentRequest,
    ) -> Pin<Box<dyn Future<Output = Result<registry::AgentResponse, RegistryError>> + Send + 'a>>
    {
        Box::pin(async move {
            let agent_response = self
                .prompt_with_history(request.message, &request.history)
                .await
                .map_err(|e| RegistryError::InvocationFailed(e.to_string()))?;

            Ok(registry::AgentResponse {
                content: agent_response.content(),
                agent_name: AgentHandle::name(self).to_string(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CompletionError;
    use crate::message::{AssistantContent, ToolCall};
    use crate::provider::Usage;
    use crate::tool::{Tool, ToolDefinition};
    use serde::Deserialize;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct MockModel;

    impl CompletionModel for MockModel {
        type Response = serde_json::Value;

        async fn completion(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            let last_message = request
                .messages
                .last()
                .map(|m| m.text())
                .unwrap_or_default();
            Ok(CompletionResponse::new(
                Message::assistant(format!("Echo: {}", last_message)),
                Usage::new(10, 20),
                serde_json::json!({}),
            ))
        }

        fn model_id(&self) -> &str {
            "mock-model"
        }

        fn provider(&self) -> &str {
            "mock"
        }
    }

    /// Mock model that returns tool calls on first invocation, then final answer
    #[derive(Clone)]
    struct ToolCallingMockModel {
        call_count: Arc<AtomicUsize>,
    }

    impl ToolCallingMockModel {
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl CompletionModel for ToolCallingMockModel {
        type Response = serde_json::Value;

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);

            if count == 0 {
                // First call: return a tool call
                let message = Message::Assistant {
                    content: vec![AssistantContent::ToolCall(ToolCall::new(
                        "call-123",
                        "add_numbers",
                        r#"{"a": 5, "b": 3}"#,
                    ))],
                    id: None,
                    reasoning: None,
                };
                Ok(CompletionResponse::new(
                    message,
                    Usage::new(10, 20),
                    serde_json::json!({}),
                ))
            } else {
                // Subsequent calls: return final answer
                Ok(CompletionResponse::new(
                    Message::assistant("The result is 8"),
                    Usage::new(10, 20),
                    serde_json::json!({}),
                ))
            }
        }

        fn model_id(&self) -> &str {
            "tool-calling-mock"
        }

        fn provider(&self) -> &str {
            "mock"
        }
    }

    /// Mock tool for testing
    #[derive(Clone)]
    struct AddNumbersTool;

    #[derive(Deserialize)]
    struct AddArgs {
        a: i32,
        b: i32,
    }

    impl Tool for AddNumbersTool {
        const NAME: &'static str = "add_numbers";
        type Args = AddArgs;
        type Output = i32;
        type Error = std::convert::Infallible;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition::new(
                "add_numbers",
                "Add two numbers together",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }),
            )
        }

        async fn call(&self, args: AddArgs) -> Result<i32, Self::Error> {
            Ok(args.a + args.b)
        }
    }

    #[tokio::test]
    async fn test_agent_builder() {
        let agent = AgentBuilder::new(MockModel)
            .name("test-agent")
            .preamble("You are a helpful assistant.")
            .capability("testing")
            .build();

        assert_eq!(agent.name(), Some("test-agent"));
        assert_eq!(agent.preamble(), Some("You are a helpful assistant."));
        assert_eq!(agent.capabilities(), &["testing"]);
    }

    #[tokio::test]
    async fn test_agent_prompt() {
        let agent = AgentBuilder::new(MockModel)
            .preamble("You are helpful.")
            .build();

        let response = agent.prompt("Hello").await.unwrap();
        assert!(response.content().contains("Echo: Hello"));
    }

    #[tokio::test]
    async fn test_agent_with_hook() {
        use crate::hook::builtin::PrefixHook;

        let agent = AgentBuilder::new(MockModel)
            .hook(PrefixHook::new("[TEST] "))
            .build();

        let response = agent.prompt("Hello").await.unwrap();
        assert!(response.content().contains("[TEST] Hello"));
    }

    #[tokio::test]
    async fn test_agent_tool_execution_loop() {
        let model = ToolCallingMockModel::new();
        let call_count = model.call_count.clone();

        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .build();

        let response = agent.prompt("What is 5 + 3?").await.unwrap();

        // Verify the model was called twice (once with tool call, once after tool result)
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        // Verify final response
        assert!(response.content().contains("The result is 8"));

        // Verify no more tool calls in final response
        assert!(!response.has_tool_calls());
    }

    #[tokio::test]
    async fn test_agent_tool_not_found() {
        use crate::message::UserContent;

        /// Model that calls a non-existent tool
        #[derive(Clone)]
        struct NonExistentToolModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for NonExistentToolModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call for non-existent tool
                    let message = Message::Assistant {
                        content: vec![AssistantContent::ToolCall(ToolCall::new(
                            "call-456",
                            "non_existent_tool",
                            "{}",
                        ))],
                        id: None,
                        reasoning: None,
                    };
                    Ok(CompletionResponse::new(
                        message,
                        Usage::new(10, 20),
                        serde_json::json!({}),
                    ))
                } else {
                    // Check if we got tool result with error by looking at ToolResult content
                    let has_error = request.messages.iter().any(|m| {
                        if let Message::User { content, .. } = m {
                            content.iter().any(|c| {
                                if let UserContent::ToolResult { content, .. } = c {
                                    content.contains("not found")
                                } else {
                                    false
                                }
                            })
                        } else {
                            false
                        }
                    });
                    let final_msg = if has_error {
                        "Tool was not found"
                    } else {
                        "Unexpected"
                    };
                    Ok(CompletionResponse::new(
                        Message::assistant(final_msg),
                        Usage::new(10, 20),
                        serde_json::json!({}),
                    ))
                }
            }

            fn model_id(&self) -> &str {
                "non-existent-tool-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = NonExistentToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };

        // Agent with a tool, but model calls a different tool
        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .build();

        let response = agent.prompt("Call unknown tool").await.unwrap();

        // The response should indicate tool was not found
        assert!(
            response.content().contains("Tool was not found"),
            "Expected 'Tool was not found', got: {}",
            response.content()
        );
    }

    #[tokio::test]
    async fn test_agent_with_tool_hook() {
        use crate::error::HookResult;
        use crate::hook::Hook;
        use crate::message::UserContent;

        /// Hook that blocks certain tools
        #[derive(Clone)]
        struct ToolBlockerHook;

        impl Hook for ToolBlockerHook {
            async fn pre_tool_call(
                &self,
                tool_name: &str,
                _args: serde_json::Value,
                _ctx: &mut HookContext,
            ) -> HookResult<ToolCallDecision> {
                if tool_name == "blocked_tool" {
                    Ok(ToolCallDecision::Block("This tool is blocked".to_string()))
                } else {
                    Ok(ToolCallDecision::Proceed(_args))
                }
            }
        }

        /// Model that calls a blocked tool
        #[derive(Clone)]
        struct BlockedToolModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for BlockedToolModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);

                if count == 0 {
                    let message = Message::Assistant {
                        content: vec![AssistantContent::ToolCall(ToolCall::new(
                            "call-789",
                            "blocked_tool",
                            "{}",
                        ))],
                        id: None,
                        reasoning: None,
                    };
                    Ok(CompletionResponse::new(
                        message,
                        Usage::new(10, 20),
                        serde_json::json!({}),
                    ))
                } else {
                    // Check if tool was blocked by looking at ToolResult content
                    let was_blocked = request.messages.iter().any(|m| {
                        if let Message::User { content, .. } = m {
                            content.iter().any(|c| {
                                if let UserContent::ToolResult { content, .. } = c {
                                    content.contains("blocked")
                                } else {
                                    false
                                }
                            })
                        } else {
                            false
                        }
                    });
                    let final_msg = if was_blocked {
                        "Tool was blocked by hook"
                    } else {
                        "Unexpected"
                    };
                    Ok(CompletionResponse::new(
                        Message::assistant(final_msg),
                        Usage::new(10, 20),
                        serde_json::json!({}),
                    ))
                }
            }

            fn model_id(&self) -> &str {
                "blocked-tool-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = BlockedToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };

        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .hook(ToolBlockerHook)
            .build();

        let response = agent.prompt("Use blocked tool").await.unwrap();

        // The response should indicate tool was blocked
        assert!(response.content().contains("Tool was blocked by hook"));
    }

    #[tokio::test]
    async fn test_agent_max_turns_limit() {
        /// Model that always returns tool calls
        #[derive(Clone)]
        struct InfiniteToolCallingModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for InfiniteToolCallingModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);
                // Always return a tool call
                let message = Message::Assistant {
                    content: vec![AssistantContent::ToolCall(ToolCall::new(
                        format!("call-{}", count),
                        "add_numbers",
                        r#"{"a": 1, "b": 2}"#,
                    ))],
                    id: None,
                    reasoning: None,
                };
                Ok(CompletionResponse::new(
                    message,
                    Usage::new(10, 20),
                    serde_json::json!({}),
                ))
            }

            fn model_id(&self) -> &str {
                "infinite-tool-calling-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = InfiniteToolCallingModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };
        let call_count = model.call_count.clone();

        // Agent with max_turns = 3
        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .max_turns(3)
            .build();

        let response = agent.prompt("What is 1 + 2?").await.unwrap();

        // Model should have been called 4 times:
        // 1 initial + 3 turns (max_turns limit)
        // After 3 turns, the loop breaks even though response still has tool calls
        assert_eq!(call_count.load(Ordering::SeqCst), 4);

        // Response should still have tool calls since we hit the limit
        assert!(response.has_tool_calls());
    }

    #[tokio::test]
    async fn test_agent_retries_on_transient_error() {
        /// Model that fails with retryable error N times, then succeeds
        #[derive(Clone)]
        struct RetryingMockModel {
            call_count: Arc<AtomicUsize>,
            fail_count: usize,
        }

        impl CompletionModel for RetryingMockModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);

                if count < self.fail_count {
                    // Return a retryable error
                    Err(CompletionError::RateLimited {
                        retry_after_secs: Some(0), // No actual delay in test
                    })
                } else {
                    // Success after retries
                    Ok(CompletionResponse::new(
                        Message::assistant("Success after retries!"),
                        Usage::new(10, 20),
                        serde_json::json!({}),
                    ))
                }
            }

            fn model_id(&self) -> &str {
                "retrying-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = RetryingMockModel {
            call_count: Arc::new(AtomicUsize::new(0)),
            fail_count: 2, // Fail twice, then succeed
        };
        let call_count = model.call_count.clone();

        // Use custom retry config with 0-second delays for fast test
        let agent = AgentBuilder::new(model)
            .retry_config(RetryConfig::with_delays(5, vec![0, 0, 0, 0, 0]))
            .build();

        let response = agent.prompt("Test retry").await.unwrap();

        // Model should have been called 3 times (2 failures + 1 success)
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        assert!(response.content().contains("Success after retries"));
    }

    #[tokio::test]
    async fn test_agent_exhausts_retries_on_persistent_error() {
        /// Model that always fails with retryable error
        #[derive(Clone)]
        struct AlwaysFailingModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for AlwaysFailingModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Err(CompletionError::RateLimited {
                    retry_after_secs: Some(0),
                })
            }

            fn model_id(&self) -> &str {
                "always-failing-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = AlwaysFailingModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };
        let call_count = model.call_count.clone();

        // Only allow 3 retries
        let agent = AgentBuilder::new(model)
            .retry_config(RetryConfig::with_delays(3, vec![0, 0, 0]))
            .build();

        let result = agent.prompt("Test exhausted retries").await;

        // Should fail after exhausting retries
        assert!(result.is_err());

        // Model should have been called 4 times (1 initial + 3 retries)
        assert_eq!(call_count.load(Ordering::SeqCst), 4);

        // Error should be rate limited
        let err = result.unwrap_err();
        assert!(matches!(err, CompletionError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn test_agent_no_retry_on_non_retryable_error() {
        /// Model that fails with non-retryable error
        #[derive(Clone)]
        struct NonRetryableErrorModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for NonRetryableErrorModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                // Authentication errors are not retryable
                Err(CompletionError::AuthenticationFailed)
            }

            fn model_id(&self) -> &str {
                "non-retryable-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = NonRetryableErrorModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };
        let call_count = model.call_count.clone();

        let agent = AgentBuilder::new(model)
            .retry_config(RetryConfig::with_delays(5, vec![0, 0, 0, 0, 0]))
            .build();

        let result = agent.prompt("Test non-retryable").await;

        // Should fail immediately without retrying
        assert!(result.is_err());

        // Model should have been called only once (no retries)
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Error should be authentication failed
        let err = result.unwrap_err();
        assert!(matches!(err, CompletionError::AuthenticationFailed));
    }

    #[tokio::test]
    async fn test_execution_stats_accumulation() {
        // Use the ToolCallingMockModel which makes 2 LLM calls
        let model = ToolCallingMockModel::new();
        let call_count = model.call_count.clone();

        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .build();

        let response = agent.prompt("What is 5 + 3?").await.unwrap();

        // Verify the model was called twice (once with tool call, once after tool result)
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        // Verify stats are accumulated correctly
        let stats = &response.stats;

        // 2 LLM completions
        assert_eq!(stats.message_count, 2, "Expected 2 LLM completions");

        // 1 tool call
        assert_eq!(stats.tool_calls, 1, "Expected 1 tool call");

        // Tokens: 10+10 input, 20+20 output (from MockModel Usage::new(10, 20))
        assert_eq!(stats.input_tokens, 20, "Expected 20 input tokens (10+10)");
        assert_eq!(stats.output_tokens, 40, "Expected 40 output tokens (20+20)");

        // No retries
        assert_eq!(stats.retries_count, 0, "Expected 0 retries");

        // No tool errors
        assert_eq!(stats.tool_error_count, 0, "Expected 0 tool errors");

        // Execution time is recorded (u64 millis; may be 0 on a fast mock run).
        let _ = stats.execution_time_ms;
    }

    #[tokio::test]
    async fn test_execution_stats_tool_errors() {
        /// Model that calls a non-existent tool, resulting in a tool error
        #[derive(Clone)]
        struct ToolErrorModel {
            call_count: Arc<AtomicUsize>,
        }

        impl CompletionModel for ToolErrorModel {
            type Response = serde_json::Value;

            async fn completion(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call for non-existent tool
                    let message = Message::Assistant {
                        content: vec![AssistantContent::ToolCall(ToolCall::new(
                            "call-err",
                            "nonexistent_tool",
                            "{}",
                        ))],
                        id: None,
                        reasoning: None,
                    };
                    Ok(CompletionResponse::new(
                        message,
                        Usage::new(15, 25),
                        serde_json::json!({}),
                    ))
                } else {
                    // Second call: final response
                    Ok(CompletionResponse::new(
                        Message::assistant("Tool not found, but done"),
                        Usage::new(5, 10),
                        serde_json::json!({}),
                    ))
                }
            }

            fn model_id(&self) -> &str {
                "tool-error-mock"
            }

            fn provider(&self) -> &str {
                "mock"
            }
        }

        let model = ToolErrorModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        };

        // Agent with AddNumbersTool but model calls nonexistent_tool
        let agent = AgentBuilder::new(model)
            .tool(crate::tool::into_arc_dyn(AddNumbersTool))
            .build();

        let response = agent.prompt("Call a tool").await.unwrap();

        let stats = &response.stats;

        // 2 LLM completions
        assert_eq!(stats.message_count, 2);

        // 1 tool call attempted
        assert_eq!(stats.tool_calls, 1);

        // 1 tool error (tool not found)
        assert_eq!(stats.tool_error_count, 1, "Expected 1 tool error");

        // Tokens accumulated from both calls
        assert_eq!(stats.input_tokens, 20, "Expected 15+5 input tokens");
        assert_eq!(stats.output_tokens, 35, "Expected 25+10 output tokens");
    }

    #[test]
    fn test_is_truncated_json_error() {
        // Test EOF while parsing (the specific error from the user's log)
        let truncated_json = r#"{"file_path": "/some/path", "new_string": "#;
        let err = serde_json::from_str::<serde_json::Value>(truncated_json).unwrap_err();
        assert!(
            is_truncated_json_error(&err),
            "Expected 'EOF while parsing' to be detected as truncation: {}",
            err
        );

        // Test truncated string
        let truncated_string = r#"{"key": "incomplete"#;
        let err = serde_json::from_str::<serde_json::Value>(truncated_string).unwrap_err();
        assert!(
            is_truncated_json_error(&err),
            "Expected truncated string to be detected as truncation: {}",
            err
        );

        // Test truncated object
        let truncated_object = r#"{"key": {"nested":"#;
        let err = serde_json::from_str::<serde_json::Value>(truncated_object).unwrap_err();
        assert!(
            is_truncated_json_error(&err),
            "Expected truncated object to be detected as truncation: {}",
            err
        );

        // Non-truncation errors should NOT be detected as truncation
        let invalid_json = r#"{"key": invalid}"#;
        let err = serde_json::from_str::<serde_json::Value>(invalid_json).unwrap_err();
        assert!(
            !is_truncated_json_error(&err),
            "Invalid JSON (not truncation) should not be detected as truncation: {}",
            err
        );

        // Type mismatch errors should NOT be detected as truncation
        let type_mismatch = r#"{"key": 123}"#;
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct ExpectsString {
            key: String,
        }
        let err = serde_json::from_str::<ExpectsString>(type_mismatch).unwrap_err();
        assert!(
            !is_truncated_json_error(&err),
            "Type mismatch should not be detected as truncation: {}",
            err
        );
    }

    #[test]
    fn test_is_tool_argument_error() {
        // Positive cases - these should be detected as argument errors
        assert!(is_tool_argument_error("missing field `name`"));
        assert!(is_tool_argument_error("invalid type: expected string"));
        assert!(is_tool_argument_error("unknown field `foo`"));
        assert!(is_tool_argument_error("invalid argument provided"));
        assert!(is_tool_argument_error("required field missing"));
        assert!(is_tool_argument_error("file not found: /some/path"));
        assert!(is_tool_argument_error("no such file or directory"));

        // Negative cases - these should NOT be detected as argument errors
        assert!(!is_tool_argument_error("connection refused"));
        assert!(!is_tool_argument_error("timeout occurred"));
        assert!(!is_tool_argument_error("internal server error"));
    }

    #[test] // Not #[tokio::test] - this is a sync test!
    fn test_agent_build_without_runtime() {
        // This test verifies that with_skills() doesn't panic when called
        // from a synchronous context without an active Tokio runtime

        let model = MockModel;

        // This should NOT panic - it should create an empty registry instead
        let agent = Agent::builder(model).with_skills().build();

        // Registry should exist but be empty (no runtime to discover skills)
        assert!(agent.skill_registry.is_some());
        let registry = agent.skill_registry.unwrap();
        assert!(registry.is_empty());
    }
}
