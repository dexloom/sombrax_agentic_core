//! ExtractorWrapper enum for unified LLM extraction across providers.

use crate::error::CompletionError;
use crate::providers::provider_type::ProviderType;
use crate::providers::zai::{CompletionRequest, Message, ToolDefinition};
use crate::providers::{
    AnthropicClient, CerebrasClient, LmStudioClient, MinimaxClient, MlxLmClient, OllamaClient,
    OpenAIClient, OpenRouterClient, ZaiClient,
};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Error type for extraction operations.
#[derive(Error, Debug)]
pub enum ExtractorError {
    /// Provider/completion error
    #[error("Provider error: {0}")]
    Provider(#[from] CompletionError),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// No tool call or valid JSON found in response
    #[error("No tool call or valid JSON found in response")]
    NoToolCall,
}

/// Wrapper enum for different LLM client types used for extraction.
///
/// This is necessary because different providers return different concrete client types.
/// By storing the client (not the extractor), we can create extractors for different
/// response types on demand.
///
/// ## Design Note
///
/// Unlike `AgentWrapper` which stores `Agent<M>`, we store the underlying `Client`
/// because extraction is generic over the response type `T`.
/// This allows using the same `ExtractorWrapper` instance for multiple extraction
/// operations with different response types.
///
/// Extraction is implemented by using the completion API with a tool definition
/// that describes the expected JSON schema for the response type.
pub enum ExtractorWrapper {
    /// OpenRouter client for multi-model aggregator
    OpenRouter(OpenRouterClient),
    /// OpenAI client
    OpenAI(OpenAIClient),
    /// Anthropic (Claude) client
    Anthropic(AnthropicClient),
    /// MiniMax client (Anthropic-compatible API)
    Minimax(MinimaxClient),
    /// Cerebras client with custom tool content serialization
    Cerebras(CerebrasClient),
    /// Ollama client using native `/api/chat` (cloud + local)
    Ollama(OllamaClient),
    /// ZAI client with thinking mode support
    Zai(ZaiClient),
    /// MLX-LM client for Apple Silicon local models
    MlxLm(MlxLmClient),
    /// LMStudio client with anti-repetition controls
    LmStudio(LmStudioClient),
}

impl ExtractorWrapper {
    /// Extract structured data from text using the underlying provider.
    ///
    /// This method uses the completion API with a tool definition to extract
    /// structured data. The type `T` is converted to a JSON schema using schemars,
    /// and the LLM is asked to call a tool with that schema as parameters.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The response type to extract. Must implement `JsonSchema` for schema
    ///   generation and `DeserializeOwned` for deserialization from JSON.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name to use for extraction
    /// * `preamble` - System prompt/preamble for the extractor
    /// * `prompt` - The user prompt containing the data to extract from
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[derive(Debug, JsonSchema, Deserialize)]
    /// struct ReviewResponse {
    ///     rating: u8,
    ///     note: String,
    /// }
    ///
    /// let extractor = ExtractorWrapper::OpenAI(client);
    /// let review: ReviewResponse = extractor
    ///     .extract("gpt-4", "You are a reviewer.", "Analyze this code...")
    ///     .await?;
    /// ```
    pub async fn extract<T>(
        &self,
        model: &str,
        preamble: &str,
        prompt: &str,
    ) -> Result<T, ExtractorError>
    where
        T: JsonSchema + DeserializeOwned + Serialize + Send + Sync + 'static,
    {
        // Generate JSON schema for the type
        let schema = schemars::schema_for!(T);
        let schema_json = serde_json::to_value(&schema).map_err(|e| {
            ExtractorError::Serialization(format!("Failed to serialize schema: {}", e))
        })?;

        // Create a tool definition for structured extraction
        let tool = ToolDefinition {
            name: "extract".to_string(),
            description: "Extract structured data from the provided text".to_string(),
            parameters: schema_json,
        };

        // Build the completion request
        let request = CompletionRequest {
            preamble: Some(preamble.to_string()),
            messages: vec![Message::user(prompt)],
            tools: vec![tool],
            temperature: Some(0.0), // Use low temperature for extraction
            max_tokens: None,
            additional_params: None,
            cache: Default::default(),
        };

        // Call the appropriate provider and extract the tool arguments
        let message = match self {
            ExtractorWrapper::OpenRouter(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::OpenAI(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::Anthropic(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::Minimax(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::Cerebras(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::Ollama(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::Zai(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::MlxLm(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
            ExtractorWrapper::LmStudio(client) => {
                let completion_model = client.completion_model(model);
                let response = completion_model
                    .completion(request)
                    .await
                    .map_err(|e| ExtractorError::Serialization(e.to_string()))?;
                response.message
            }
        };

        let arguments = Self::extract_tool_arguments(&message)?;

        // Deserialize the tool arguments into the target type
        serde_json::from_str(&arguments).map_err(|e| {
            ExtractorError::Serialization(format!(
                "Failed to deserialize extracted data: {}. Raw arguments: {}",
                e, arguments
            ))
        })
    }

    /// Extract tool call arguments from the response message.
    fn extract_tool_arguments(message: &Message) -> Result<String, ExtractorError> {
        // Check for tool calls
        if let Some(tool_call) = message.tool_calls.as_ref().and_then(|calls| calls.first()) {
            return Ok(tool_call.arguments.clone());
        }

        // Fallback: try to parse the content as JSON directly
        // Some models may return JSON in the content field instead of tool calls
        if !message.content.is_empty() {
            // Try to extract JSON from the content
            if let Some(json) = Self::extract_json_from_content(&message.content) {
                return Ok(json);
            }
        }

        Err(ExtractorError::NoToolCall)
    }

    /// Try to extract JSON from content that may have markdown or other formatting.
    fn extract_json_from_content(content: &str) -> Option<String> {
        // Try direct parse first
        if serde_json::from_str::<serde_json::Value>(content).is_ok() {
            return Some(content.to_string());
        }

        // Try to extract JSON from markdown code blocks
        let trimmed = content.trim();
        if trimmed.starts_with("```json") {
            if let Some(end) = trimmed.rfind("```") {
                let json_start = trimmed.find('\n').map(|i| i + 1).unwrap_or(7);
                if json_start < end {
                    let json_str = trimmed[json_start..end].trim();
                    if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                        return Some(json_str.to_string());
                    }
                }
            }
        }

        // Try to find JSON object in content
        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                if start < end {
                    let json_str = &content[start..=end];
                    if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                        return Some(json_str.to_string());
                    }
                }
            }
        }

        None
    }

    /// Get the provider type for this extractor.
    pub fn provider_type(&self) -> ProviderType {
        match self {
            ExtractorWrapper::OpenRouter(_) => ProviderType::OpenRouter,
            ExtractorWrapper::OpenAI(_) => ProviderType::OpenAI,
            ExtractorWrapper::Anthropic(_) => ProviderType::Anthropic,
            ExtractorWrapper::Minimax(_) => ProviderType::Minimax,
            ExtractorWrapper::Cerebras(_) => ProviderType::Cerebras,
            ExtractorWrapper::Ollama(_) => ProviderType::Ollama,
            ExtractorWrapper::Zai(_) => ProviderType::Zai,
            ExtractorWrapper::MlxLm(_) => ProviderType::MlxLm,
            ExtractorWrapper::LmStudio(_) => ProviderType::LmStudio,
        }
    }
}
