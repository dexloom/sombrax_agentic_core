//! Fetch tool for HTTP requests

use std::collections::HashMap;
use std::time::{Duration, Instant};

use reqwest::Client;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info_span, instrument, Instrument};

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

use super::env_expand::expand_env_vars;

/// Default timeout in milliseconds
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Maximum timeout in milliseconds
const MAX_TIMEOUT_MS: u64 = 300_000;

/// Maximum response body size in characters
const MAX_RESPONSE_BODY_CHARS: usize = 100_000;

/// Fetch HTTP resources
#[derive(Clone)]
pub struct FetchTool {
    #[allow(dead_code)]
    context: ToolContext,
    http_client: Client,
}

impl FetchTool {
    /// Create a new fetch tool
    pub fn new(context: ToolContext) -> Self {
        Self {
            context,
            http_client: Client::new(),
        }
    }
}

/// Arguments for the fetch tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchArgs {
    /// URL to fetch
    pub url: String,
    /// HTTP method (GET, POST, PUT, DELETE, PATCH)
    pub method: String,
    /// JSON body for POST/PUT/PATCH
    #[serde(default)]
    pub body: Option<serde_json::Value>,
    /// Custom headers
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
    /// Timeout in milliseconds (default: 30000, max: 300000)
    #[serde(
        default,
        deserialize_with = "crate::tools::serde_flexible::deserialize_flexible_optional_u64"
    )]
    pub timeout: Option<u64>,
}

/// Output of the fetch tool
#[derive(Debug, Serialize)]
pub struct FetchOutput {
    /// HTTP status code
    pub status_code: u16,
    /// Response body (as JSON if possible, truncated if exceeding size limit)
    pub body: serde_json::Value,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Whether request was successful (2xx status)
    pub success: bool,
    /// URL that was fetched
    pub url: String,
    /// HTTP method used
    pub method: String,
    /// Whether the response body was truncated due to size limits
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Original body size in characters (only present when truncated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_size: Option<usize>,
}

impl Tool for FetchTool {
    const NAME: &'static str = "fetch";
    type Args = FetchArgs;
    type Output = FetchOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(FetchArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Fetch HTTP resources (APIs, web pages, etc.).

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What URL do I need to fetch?
2. What HTTP method should I use (GET, POST, etc.)?
3. Do I need to send a request body or custom headers?

## PARAMETERS

- `url` (REQUIRED, STRING): Full URL to fetch
  Environment variables are supported using ${VAR_NAME} syntax
  CORRECT: "https://api.example.com/data"
  CORRECT: "https://api.etherscan.io/api?apikey=${ETHERSCAN_API_KEY}"
  CORRECT: "https://httpbin.org/get"
  WRONG: {"url": "..."} <-- Do NOT pass JSON objects!
  WRONG: {} <-- Empty object is invalid!

- `method` (REQUIRED, STRING): HTTP method
  CORRECT: "GET"
  CORRECT: "POST"
  Valid methods: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS

- `body` (optional, JSON): Request body for POST/PUT/PATCH requests
  Example: {"name": "value", "count": 42}

- `headers` (optional, OBJECT): Custom HTTP headers
  Header values support environment variables using ${VAR_NAME} syntax
  Example: {"Authorization": "Bearer ${MY_API_TOKEN}"}

- `timeout` (optional, NUMBER): Timeout in milliseconds (default: 30000, max: 300000)

## EXAMPLES

Simple GET request:
  url: "https://api.github.com/repos/owner/repo"
  method: "GET"

POST with JSON body:
  url: "https://api.example.com/submit"
  method: "POST"
  body: {"key": "value"}
  headers: {"Content-Type": "application/json"}

GET with authentication:
  url: "https://api.example.com/protected"
  method: "GET"
  headers: {"Authorization": "Bearer my-token"}

## RESPONSE FORMAT

Returns:
- status_code: HTTP status (200, 404, etc.)
- body: Response body (parsed as JSON if possible)
- headers: Response headers
- success: true if 2xx status code

## COMMON MISTAKES TO AVOID

1. Do NOT pass JSON objects as url or method - use plain strings
2. Do NOT forget to specify the method parameter
3. Do NOT use HTTP (non-secure) URLs for sensitive data
4. Large responses may be truncated
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "fetch", url = %args.url, method = %args.method))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Validate method
        let method = args.method.to_uppercase();
        let valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];
        if !valid_methods.contains(&method.as_str()) {
            return Err(ToolError::Validation(format!(
                "Invalid HTTP method: {}. Valid methods: {:?}",
                method, valid_methods
            )));
        }

        // Expand environment variables in URL
        let url = expand_env_vars(&args.url)?;

        let timeout_ms = args
            .timeout
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(MAX_TIMEOUT_MS);
        let timeout_duration = Duration::from_millis(timeout_ms);

        // Build request
        let mut request = match method.as_str() {
            "GET" => self.http_client.get(&url),
            "POST" => self.http_client.post(&url),
            "PUT" => self.http_client.put(&url),
            "DELETE" => self.http_client.delete(&url),
            "PATCH" => self.http_client.patch(&url),
            "HEAD" => self.http_client.head(&url),
            _ => self.http_client.get(&url),
        };

        // Add headers (with env var expansion in values)
        if let Some(headers) = &args.headers {
            for (key, value) in headers {
                let expanded_value = expand_env_vars(value)?;
                request = request.header(key.as_str(), expanded_value.as_str());
            }
        }

        // Add body
        if let Some(body) = &args.body {
            request = request.json(body);
        }

        let start = Instant::now();

        // Execute with timeout
        let result = timeout(timeout_duration, async {
            request.send().instrument(info_span!("http_request")).await
        })
        .await;

        match result {
            Ok(Ok(response)) => {
                let response_time_ms = start.elapsed().as_millis() as u64;
                let status_code = response.status().as_u16();
                let success = response.status().is_success();

                // Collect headers
                let headers: HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                // Get body and truncate if too large
                let body_text = response.text().await.unwrap_or_default();
                let original_len = body_text.len();
                let truncated = original_len > MAX_RESPONSE_BODY_CHARS;

                let body_text = if truncated {
                    let truncation_msg = format!(
                        "\n... [TRUNCATED: response was {} chars, showing first {}. Use more specific query parameters to reduce response size.]",
                        original_len, MAX_RESPONSE_BODY_CHARS
                    );
                    let end = body_text
                        .char_indices()
                        .nth(MAX_RESPONSE_BODY_CHARS)
                        .map(|(i, _)| i)
                        .unwrap_or(body_text.len());
                    format!("{}{}", &body_text[..end], truncation_msg)
                } else {
                    body_text
                };

                let body: serde_json::Value = serde_json::from_str(&body_text)
                    .unwrap_or(serde_json::Value::String(body_text));

                Ok(FetchOutput {
                    status_code,
                    body,
                    headers,
                    response_time_ms,
                    success,
                    url: args.url,
                    method,
                    truncated,
                    original_size: if truncated { Some(original_len) } else { None },
                })
            }
            Ok(Err(e)) => Err(ToolError::Http(e)),
            Err(_) => Err(ToolError::Timeout(timeout_ms)),
        }
    }
}
