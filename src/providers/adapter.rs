//! CompletionModel trait implementations for sac providers
//!
//! This module provides adapter implementations that bridge the sac providers
//! completion models with the `CompletionModel` trait from the core library.
//!
//! # Usage
//!
//! ```rust,no_run
//! use sombrax_agentic_core::providers::{ZaiClientBuilder, ZaiCompletionModelAdapter};
//! use sombrax_agentic_core::{AgentBuilder, CompletionModel};
//!
//! let client = ZaiClientBuilder::new("api-key").build();
//! let model = client.completion_model("glm-4.6");
//! let adapter = ZaiCompletionModelAdapter::new(model);
//!
//! // Now `adapter` implements CompletionModel and can be used with Agent
//! let agent = AgentBuilder::new(adapter)
//!     .preamble("You are a helpful assistant.")
//!     .build();
//! ```

use crate::{
    message::{AssistantContent, Message, ToolCall, ToolCallFunction, UserContent},
    provider::{CompletionRequest, CompletionResponse, Usage},
    tool::ToolDefinition,
    CompletionError, CompletionModel,
};

use super::zai::client as zai_types;

// ============================================================================
// Type Conversion Helpers
// ============================================================================

/// Convert sac Message to provider-local Message format
fn convert_message_to_provider(msg: &Message) -> zai_types::Message {
    match msg {
        Message::User { content, .. } => {
            // Extract text and tool results from user content
            let mut text_parts = Vec::new();
            let mut tool_call_id = None;

            for item in content {
                match item {
                    UserContent::Text { text } => text_parts.push(text.clone()),
                    UserContent::ToolResult { id, content } => {
                        tool_call_id = Some(id.clone());
                        text_parts.push(content.clone());
                    }
                    UserContent::Image { .. } | UserContent::Document { .. } => {
                        // Images/documents not supported by these providers
                    }
                }
            }

            // Tool results should use role="tool" not "user"
            // This ensures proper formatting in ChatML and other templates
            let role = if tool_call_id.is_some() {
                "tool".to_string()
            } else {
                "user".to_string()
            };

            zai_types::Message {
                role,
                content: text_parts.join("\n"),
                tool_calls: None,
                tool_call_id,
                reasoning: None,
            }
        }
        Message::Assistant {
            content, reasoning, ..
        } => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for item in content {
                match item {
                    AssistantContent::Text { text } => text_parts.push(text.clone()),
                    AssistantContent::ToolCall(tc) => {
                        tool_calls.push(zai_types::ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        });
                    }
                    AssistantContent::Reasoning { reasoning } => {
                        // Include reasoning as text
                        text_parts.extend(reasoning.clone());
                    }
                }
            }

            zai_types::Message {
                role: "assistant".to_string(),
                content: text_parts.join("\n"),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
                reasoning: reasoning.clone(),
            }
        }
    }
}

/// Convert sac ToolDefinition to provider-local format
fn convert_tool_definition(tool: &ToolDefinition) -> zai_types::ToolDefinition {
    zai_types::ToolDefinition {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
    }
}

/// Convert provider-local Message to sac Message
fn convert_message_from_provider(msg: &zai_types::Message) -> Message {
    if msg.role == "assistant" {
        let mut content = Vec::new();

        if !msg.content.is_empty() {
            content.push(AssistantContent::Text {
                text: msg.content.clone(),
            });
        }

        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                content.push(AssistantContent::ToolCall(ToolCall {
                    id: tc.id.clone(),
                    function: ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                }));
            }
        }

        Message::Assistant {
            content,
            id: None,
            // Carry reasoning forward so the next request can echo it back
            // as `reasoning_content`. Required by providers that strict-check
            // it on assistant tool-call turns when thinking is enabled
            // (e.g., Moonshot AI: HTTP 400 "thinking is enabled but
            // reasoning_content is missing in assistant tool call message
            // at index N"). The reverse mapping in `convert_message_to_provider`
            // already passes this field through, so the round-trip is
            // symmetric.
            reasoning: msg.reasoning.clone(),
        }
    } else {
        Message::user(&msg.content)
    }
}

/// Convert sac CompletionRequest to provider-local format
fn convert_request_to_provider(request: &CompletionRequest) -> zai_types::CompletionRequest {
    zai_types::CompletionRequest {
        preamble: request.preamble.clone(),
        messages: request
            .messages
            .iter()
            .map(convert_message_to_provider)
            .collect(),
        tools: request.tools.iter().map(convert_tool_definition).collect(),
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        additional_params: request.additional_params.clone(),
        cache: request.cache.clone(),
    }
}

/// Convert provider error to CompletionError
fn convert_error(err: super::error::CompletionError) -> CompletionError {
    match err {
        super::error::CompletionError::Provider(pe) => match pe {
            super::error::ProviderError::Http { status, message } => {
                if status == 429 {
                    CompletionError::RateLimited {
                        retry_after_secs: None,
                    }
                } else if status == 401 {
                    CompletionError::AuthenticationFailed
                } else {
                    CompletionError::HttpError(format!("HTTP {}: {}", status, message))
                }
            }
            super::error::ProviderError::Authentication(msg) => {
                CompletionError::ProviderError(format!("Authentication failed: {}", msg))
            }
            super::error::ProviderError::RateLimited { retry_after_ms } => {
                CompletionError::RateLimited {
                    retry_after_secs: retry_after_ms.map(|ms| ms / 1000),
                }
            }
            super::error::ProviderError::InvalidResponse(msg) => {
                CompletionError::ProviderError(format!("Invalid response: {}", msg))
            }
            super::error::ProviderError::Request(e) => CompletionError::HttpError(e.to_string()),
            super::error::ProviderError::Json(e) => CompletionError::JsonError(e),
            super::error::ProviderError::EnvVarNotSet(var) => {
                CompletionError::ProviderError(format!("Environment variable not set: {}", var))
            }
        },
        super::error::CompletionError::InvalidRequest(msg) => CompletionError::InvalidRequest(msg),
    }
}

// ============================================================================
// ZAI Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `ZaiCompletionModel`
#[derive(Clone)]
pub struct ZaiCompletionModelAdapter {
    inner: super::zai::ZaiCompletionModel,
}

impl ZaiCompletionModelAdapter {
    /// Create a new adapter wrapping a ZaiCompletionModel
    pub fn new(inner: super::zai::ZaiCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for ZaiCompletionModelAdapter {
    type Response = super::zai::ZaiResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// Anthropic Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `AnthropicCompletionModel`
#[derive(Clone)]
pub struct AnthropicCompletionModelAdapter {
    inner: super::anthropic::AnthropicCompletionModel,
}

impl AnthropicCompletionModelAdapter {
    /// Create a new adapter wrapping an AnthropicCompletionModel
    pub fn new(inner: super::anthropic::AnthropicCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for AnthropicCompletionModelAdapter {
    type Response = super::anthropic::AnthropicResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// MiniMax Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `MinimaxCompletionModel`
#[derive(Clone)]
pub struct MinimaxCompletionModelAdapter {
    inner: super::minimax::MinimaxCompletionModel,
}

impl MinimaxCompletionModelAdapter {
    /// Create a new adapter wrapping a MinimaxCompletionModel
    pub fn new(inner: super::minimax::MinimaxCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for MinimaxCompletionModelAdapter {
    type Response = super::minimax::MinimaxResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// OpenAI Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `OpenAICompletionModel`
#[derive(Clone)]
pub struct OpenAICompletionModelAdapter {
    inner: super::openai::OpenAICompletionModel,
}

impl OpenAICompletionModelAdapter {
    /// Create a new adapter wrapping an OpenAICompletionModel
    pub fn new(inner: super::openai::OpenAICompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for OpenAICompletionModelAdapter {
    type Response = super::openai::OpenAIResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// Cerebras Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `CerebrasCompletionModel`
#[derive(Clone)]
pub struct CerebrasCompletionModelAdapter {
    inner: super::cerebras::CerebrasCompletionModel,
}

impl CerebrasCompletionModelAdapter {
    /// Create a new adapter wrapping a CerebrasCompletionModel
    pub fn new(inner: super::cerebras::CerebrasCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for CerebrasCompletionModelAdapter {
    type Response = super::cerebras::CerebrasResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// Ollama Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `OllamaCompletionModel`
#[derive(Clone)]
pub struct OllamaCompletionModelAdapter {
    inner: super::ollama::OllamaCompletionModel,
}

impl OllamaCompletionModelAdapter {
    /// Create a new adapter wrapping an OllamaCompletionModel
    pub fn new(inner: super::ollama::OllamaCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for OllamaCompletionModelAdapter {
    type Response = super::ollama::OllamaResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// OpenRouter Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `OpenRouterCompletionModel`
#[derive(Clone)]
pub struct OpenRouterCompletionModelAdapter {
    inner: super::openrouter::OpenRouterCompletionModel,
}

impl OpenRouterCompletionModelAdapter {
    /// Create a new adapter wrapping an OpenRouterCompletionModel
    pub fn new(inner: super::openrouter::OpenRouterCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for OpenRouterCompletionModelAdapter {
    type Response = super::openrouter::OpenRouterResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

// ============================================================================
// Extension traits for ergonomic usage
// ============================================================================

/// Extension trait for ZaiClient to get an adapter directly
pub trait ZaiClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> ZaiCompletionModelAdapter;
}

impl ZaiClientExt for super::zai::ZaiClient {
    fn completion_model_adapter(&self, model_id: &str) -> ZaiCompletionModelAdapter {
        ZaiCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for AnthropicClient to get an adapter directly
pub trait AnthropicClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> AnthropicCompletionModelAdapter;
}

impl AnthropicClientExt for super::anthropic::AnthropicClient {
    fn completion_model_adapter(&self, model_id: &str) -> AnthropicCompletionModelAdapter {
        AnthropicCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for MinimaxClient to get an adapter directly
pub trait MinimaxClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> MinimaxCompletionModelAdapter;
}

impl MinimaxClientExt for super::minimax::MinimaxClient {
    fn completion_model_adapter(&self, model_id: &str) -> MinimaxCompletionModelAdapter {
        MinimaxCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for OpenAIClient to get an adapter directly
pub trait OpenAIClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> OpenAICompletionModelAdapter;
}

impl OpenAIClientExt for super::openai::OpenAIClient {
    fn completion_model_adapter(&self, model_id: &str) -> OpenAICompletionModelAdapter {
        OpenAICompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for CerebrasClient to get an adapter directly
pub trait CerebrasClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> CerebrasCompletionModelAdapter;
}

impl CerebrasClientExt for super::cerebras::CerebrasClient {
    fn completion_model_adapter(&self, model_id: &str) -> CerebrasCompletionModelAdapter {
        CerebrasCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for OllamaClient to get an adapter directly
pub trait OllamaClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> OllamaCompletionModelAdapter;
}

impl OllamaClientExt for super::ollama::OllamaClient {
    fn completion_model_adapter(&self, model_id: &str) -> OllamaCompletionModelAdapter {
        OllamaCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

/// Extension trait for OpenRouterClient to get an adapter directly
pub trait OpenRouterClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> OpenRouterCompletionModelAdapter;
}

impl OpenRouterClientExt for super::openrouter::OpenRouterClient {
    fn completion_model_adapter(&self, model_id: &str) -> OpenRouterCompletionModelAdapter {
        OpenRouterCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

// ============================================================================
// LMStudio Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `LmStudioCompletionModel`
#[derive(Clone)]
pub struct LmStudioCompletionModelAdapter {
    inner: super::lmstudio::LmStudioCompletionModel,
}

impl LmStudioCompletionModelAdapter {
    /// Create a new adapter wrapping an LmStudioCompletionModel
    pub fn new(inner: super::lmstudio::LmStudioCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for LmStudioCompletionModelAdapter {
    type Response = super::lmstudio::LmStudioResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

/// Extension trait for LmStudioClient to get an adapter directly
pub trait LmStudioClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> LmStudioCompletionModelAdapter;
}

impl LmStudioClientExt for super::lmstudio::LmStudioClient {
    fn completion_model_adapter(&self, model_id: &str) -> LmStudioCompletionModelAdapter {
        LmStudioCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

// ============================================================================
// MLX-LM Adapter
// ============================================================================

/// Adapter that implements `CompletionModel` for `MlxLmCompletionModel`
#[derive(Clone)]
pub struct MlxLmCompletionModelAdapter {
    inner: super::mlxlm::MlxLmCompletionModel,
}

impl MlxLmCompletionModelAdapter {
    /// Create a new adapter wrapping an MlxLmCompletionModel
    pub fn new(inner: super::mlxlm::MlxLmCompletionModel) -> Self {
        Self { inner }
    }
}

impl CompletionModel for MlxLmCompletionModelAdapter {
    type Response = super::mlxlm::MlxLmResponse;

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let provider_request = convert_request_to_provider(&request);
        let provider_response = self
            .inner
            .completion(provider_request)
            .await
            .map_err(convert_error)?;

        let mut resp = CompletionResponse::with_reasoning(
            convert_message_from_provider(&provider_response.message),
            Usage::with_cache(
                provider_response.usage.prompt_tokens,
                provider_response.usage.completion_tokens,
                provider_response.usage.cache_read_tokens,
                provider_response.usage.cache_creation_tokens,
            ),
            provider_response.raw,
            provider_response.reasoning_content,
        );
        resp.finish_reason = provider_response.finish_reason;
        Ok(resp)
    }
}

/// Extension trait for MlxLmClient to get an adapter directly
pub trait MlxLmClientExt {
    /// Create a completion model adapter for use with Agent
    fn completion_model_adapter(&self, model_id: &str) -> MlxLmCompletionModelAdapter;
}

impl MlxLmClientExt for super::mlxlm::MlxLmClient {
    fn completion_model_adapter(&self, model_id: &str) -> MlxLmCompletionModelAdapter {
        MlxLmCompletionModelAdapter::new(self.completion_model(model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_user_message() {
        let msg = Message::user("Hello, world!");
        let converted = convert_message_to_provider(&msg);

        assert_eq!(converted.role, "user");
        assert_eq!(converted.content, "Hello, world!");
        assert!(converted.tool_calls.is_none());
    }

    #[test]
    fn test_convert_assistant_message() {
        let msg = Message::assistant("Hello back!");
        let converted = convert_message_to_provider(&msg);

        assert_eq!(converted.role, "assistant");
        assert_eq!(converted.content, "Hello back!");
        assert!(converted.tool_calls.is_none());
    }

    #[test]
    fn test_convert_tool_definition() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Get the weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
        );

        let converted = convert_tool_definition(&tool);

        assert_eq!(converted.name, "get_weather");
        assert_eq!(converted.description, "Get the weather for a location");
    }

    #[test]
    fn test_convert_message_roundtrip() {
        let original = zai_types::Message {
            role: "assistant".to_string(),
            content: "Test response".to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        };

        let converted = convert_message_from_provider(&original);
        assert!(converted.is_assistant());
        assert_eq!(converted.text(), "Test response");
    }
}
