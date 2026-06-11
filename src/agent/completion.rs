//! Agent completion builder
//!
//! Provides a fluent API for building and sending completion requests.

use crate::agent::Agent;
use crate::context::HookContext;
use crate::error::CompletionError;
use crate::hook::ToolCallDecision;
use crate::message::Message;
use crate::provider::{CompletionModel, CompletionRequest, CompletionResponse};
use crate::telemetry::Metrics;
use std::time::Instant;

/// Builder for agent completion requests
pub struct AgentCompletion<'a, M: CompletionModel> {
    agent: &'a Agent<M>,
    message: Message,
    history: Vec<Message>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl<'a, M: CompletionModel> AgentCompletion<'a, M> {
    /// Create a new completion builder
    pub fn new(agent: &'a Agent<M>, message: Message) -> Self {
        Self {
            agent,
            message,
            history: Vec::new(),
            temperature: None,
            max_tokens: None,
            additional_params: None,
        }
    }

    /// Set the conversation history
    pub fn history(mut self, history: &[Message]) -> Self {
        self.history = history.to_vec();
        self
    }

    /// Add a message to the history
    pub fn add_history(mut self, message: Message) -> Self {
        self.history.push(message);
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the max tokens
    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set additional provider-specific parameters
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Send the completion request
    pub async fn send(self) -> Result<CompletionResponse<M::Response>, CompletionError> {
        let request_start = Instant::now();
        let metrics = Metrics::global();
        let mut ctx = HookContext::new_with_uuid();

        // Execute pre-completion hooks
        let message = self
            .agent
            .hook_chain
            .execute_pre_completion(self.message, &self.history, &mut ctx)
            .await?;

        // Get tool definitions with filter hooks applied
        let tools = if self.agent.has_tools() {
            let defs = self.agent.tool_definitions(&message.text()).await;
            self.agent
                .hook_chain
                .execute_filter_tools(defs, &mut ctx)
                .await?
        } else {
            vec![]
        };

        // Optimize context if needed
        let mut messages = self.history.clone();
        messages.push(message);

        if let Some(optimizer) = &self.agent.optimizer {
            messages = optimizer
                .optimize(messages, &self.agent.optimization_config)
                .await;
        }

        // Build completion request
        let request = CompletionRequest {
            preamble: self.agent.preamble.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params.clone(),
        };

        // Send to model and record metrics (FR-021)
        let mut response = match self.agent.model.completion(request).await {
            Ok(response) => {
                metrics.record_completion_request(
                    self.agent.model.provider(),
                    self.agent.model.model_id(),
                    true,
                );
                response
            }
            Err(e) => {
                metrics.record_completion_request(
                    self.agent.model.provider(),
                    self.agent.model.model_id(),
                    false,
                );
                return Err(e);
            }
        };

        // Notify hooks of assistant message (for display/logging)
        self.agent
            .hook_chain
            .execute_on_assistant_message(&response.message, &mut ctx)
            .await?;

        // Tool execution loop: continue while the response contains tool calls
        while response.has_tool_calls() {
            // Append the assistant message with tool calls to history
            messages.push(response.message.clone());

            // Process each tool call
            for tool_call in response.tool_calls() {
                let tool_name = &tool_call.function.name;
                let tool_args_str = &tool_call.function.arguments;

                // Parse arguments for hooks
                let args: serde_json::Value =
                    serde_json::from_str(tool_args_str).unwrap_or_else(|_| serde_json::json!({}));

                // Execute pre-tool-call hooks
                let decision = self
                    .agent
                    .hook_chain
                    .execute_pre_tool_call(tool_name, args, &mut ctx)
                    .await?;

                let tool_result = match decision {
                    ToolCallDecision::Block(reason) => {
                        // Return the block reason as the tool result
                        format!("Tool call blocked: {}", reason)
                    }
                    ToolCallDecision::Proceed(modified_args) => {
                        // Find and execute the tool
                        let result = if let Some(tool) = self.agent.find_tool(tool_name) {
                            // Convert modified args back to string for the tool call
                            let args_str = serde_json::to_string(&modified_args)?;
                            match tool.call(args_str).await {
                                Ok(output) => output,
                                Err(e) => format!("Tool execution error: {}", e),
                            }
                        } else {
                            // Tool not found error
                            format!("Tool '{}' not found", tool_name)
                        };

                        // Execute post-tool-call hooks
                        self.agent
                            .hook_chain
                            .execute_post_tool_call(tool_name, result, &mut ctx)
                            .await?
                    }
                };

                // Add tool result as a user message
                let tool_result_message = Message::tool_result(&tool_call.id, tool_result);
                messages.push(tool_result_message);
            }

            // Re-optimize context if needed
            if let Some(optimizer) = &self.agent.optimizer {
                messages = optimizer
                    .optimize(messages, &self.agent.optimization_config)
                    .await;
            }

            // Build the next completion request
            let request = CompletionRequest {
                preamble: self.agent.preamble.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                additional_params: self.additional_params.clone(),
            };

            // Send to model again and record metrics
            response = match self.agent.model.completion(request).await {
                Ok(response) => {
                    metrics.record_completion_request(
                        self.agent.model.provider(),
                        self.agent.model.model_id(),
                        true,
                    );
                    response
                }
                Err(e) => {
                    metrics.record_completion_request(
                        self.agent.model.provider(),
                        self.agent.model.model_id(),
                        false,
                    );
                    return Err(e);
                }
            };

            // Notify hooks of assistant message (for display/logging)
            self.agent
                .hook_chain
                .execute_on_assistant_message(&response.message, &mut ctx)
                .await?;
        }

        // Execute post-completion hooks on the final response
        let response = self
            .agent
            .hook_chain
            .execute_post_completion(response, &mut ctx)
            .await?;

        // Record total request latency (FR-021)
        metrics.record_request_latency(request_start.elapsed(), &[]);

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::error::CompletionError;
    use crate::provider::Usage;

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
                Message::assistant(format!("Temp={:?}: {}", request.temperature, last_message)),
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

    #[tokio::test]
    async fn test_completion_builder() {
        let agent = AgentBuilder::new(MockModel).build();

        let response = agent
            .completion("Hello")
            .temperature(0.7)
            .max_tokens(100)
            .send()
            .await
            .unwrap();

        assert!(response.content().contains("Temp=Some(0.7)"));
    }

    #[tokio::test]
    async fn test_completion_with_history() {
        let agent = AgentBuilder::new(MockModel).build();

        let response = agent
            .completion("Follow up question")
            .history(&[
                Message::user("First message"),
                Message::assistant("First response"),
            ])
            .send()
            .await
            .unwrap();

        assert!(response.content().contains("Follow up question"));
    }
}
