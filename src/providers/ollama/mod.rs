//! Ollama native provider (cloud + local).
//!
//! Speaks Ollama's own `/api/chat` protocol. A single provider serves both
//! local (`http://localhost:11434`, keyless) and cloud
//! (`https://ollama.com`, `OLLAMA_API_KEY` Bearer) usage — the wire format
//! is identical.

mod client;
mod types;

pub use client::*;
pub use types::*;
