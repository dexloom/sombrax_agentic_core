use reqwest::Client;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

/// Build an HTTP client with a 30-minute request timeout.
///
/// Local models (MLX, ollama) can take several minutes for large prompts
/// (e.g., 26B model processing 7K tokens, or pruned models generating
/// 5k+ token files at 9 tok/s). The generous timeout prevents premature
/// connection drops while still recovering from truly stuck requests.
pub(crate) fn build_http_client() -> Client {
    catch_unwind(AssertUnwindSafe(|| {
        Client::builder()
            .timeout(Duration::from_secs(1800))
            .pool_max_idle_per_host(0) // Disable connection pooling for local models
            .build()
            .expect("failed to build HTTP client")
    }))
    .unwrap_or_else(|_| {
        Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(1800))
            .pool_max_idle_per_host(0)
            .build()
            .expect("failed to build HTTP client")
    })
}
