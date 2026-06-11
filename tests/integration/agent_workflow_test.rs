//! Integration test for agent with pre/post completion hooks (T019)
//!
//! Tests the full workflow of an agent with hooks intercepting and modifying
//! message content before/after LLM calls.

use sombrax_agentic_core::agent::AgentBuilder;
use sombrax_agentic_core::context::HookContext;
use sombrax_agentic_core::error::{CompletionError, HookResult};
use sombrax_agentic_core::hook::Hook;
use sombrax_agentic_core::message::Message;
use sombrax_agentic_core::provider::{
    CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
#[allow(unused_imports)]
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Mock completion model for testing
#[derive(Clone)]
struct MockModel {
    responses: Vec<String>,
    call_count: Arc<AtomicUsize>,
    received_messages: Arc<std::sync::Mutex<Vec<String>>>,
}

impl MockModel {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(String::from).collect(),
            call_count: Arc::new(AtomicUsize::new(0)),
            received_messages: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn received_messages(&self) -> Vec<String> {
        self.received_messages.lock().unwrap().clone()
    }
}

impl CompletionModel for MockModel {
    type Response = serde_json::Value;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);

        // Record received message content
        for msg in &request.messages {
            if let Message::User { content, .. } = msg {
                for c in content {
                    if let sombrax_agentic_core::message::UserContent::Text { text } = c {
                        self.received_messages.lock().unwrap().push(text.clone());
                    }
                }
            }
        }

        let response_text = self
            .responses
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "Default response".to_string());
        let message = Message::assistant(&response_text);
        let usage = Usage::new(10, 20);

        Ok(CompletionResponse::new(
            message,
            usage,
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

/// Hook that adds a prefix to incoming messages
#[derive(Clone)]
struct PrefixHook {
    prefix: String,
}

impl Hook for PrefixHook {
    fn pre_completion(
        &self,
        message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        let prefix = self.prefix.clone();
        async move {
            match message {
                Message::User { content, id } => {
                    let modified = content
                        .into_iter()
                        .map(|c| match c {
                            sombrax_agentic_core::message::UserContent::Text { text } => {
                                sombrax_agentic_core::message::UserContent::Text {
                                    text: format!("{}{}", prefix, text),
                                }
                            }
                            other => other,
                        })
                        .collect();
                    Ok(Message::User {
                        content: modified,
                        id,
                    })
                }
                other => Ok(other),
            }
        }
    }
}

/// Hook that adds a suffix to assistant responses
#[derive(Clone)]
struct SuffixHook {
    suffix: String,
}

impl Hook for SuffixHook {
    fn post_completion_message(
        &self,
        mut message: Message,
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        let suffix = self.suffix.clone();
        async move {
            if let Message::Assistant {
                ref mut content, ..
            } = message
            {
                for c in content.iter_mut() {
                    if let sombrax_agentic_core::message::AssistantContent::Text { ref mut text } =
                        c
                    {
                        text.push_str(&suffix);
                    }
                }
            }
            Ok(message)
        }
    }
}

/// Hook that tracks execution order
#[derive(Clone)]
struct TrackingHook {
    name: String,
    pre_calls: Arc<std::sync::Mutex<Vec<String>>>,
    post_calls: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Hook for TrackingHook {
    fn pre_completion(
        &self,
        message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        self.pre_calls.lock().unwrap().push(self.name.clone());
        async move { Ok(message) }
    }

    fn post_completion_message(
        &self,
        message: Message,
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        self.post_calls.lock().unwrap().push(self.name.clone());
        async move { Ok(message) }
    }
}

#[tokio::test]
async fn test_agent_with_prefix_hook() {
    let model = MockModel::new(vec!["Hello!"]);

    let agent = AgentBuilder::new(model.clone())
        .hook(PrefixHook {
            prefix: "[MODIFIED] ".to_string(),
        })
        .build();

    let _response = agent
        .completion(Message::user("Original message"))
        .send()
        .await
        .unwrap();

    // Model should have received the prefixed message
    let received = model.received_messages();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0], "[MODIFIED] Original message");
}

#[tokio::test]
async fn test_agent_with_suffix_hook() {
    let model = MockModel::new(vec!["Response"]);

    let agent = AgentBuilder::new(model)
        .hook(SuffixHook {
            suffix: " [END]".to_string(),
        })
        .build();

    let response = agent
        .completion(Message::user("Test"))
        .send()
        .await
        .unwrap();

    // Response should have the suffix
    if let Message::Assistant { content, .. } = &response.message {
        if let sombrax_agentic_core::message::AssistantContent::Text { text } = &content[0] {
            assert_eq!(text, "Response [END]");
        }
    }
}

#[tokio::test]
async fn test_agent_with_multiple_hooks() {
    let pre_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let post_calls = Arc::new(std::sync::Mutex::new(Vec::new()));

    let model = MockModel::new(vec!["Test"]);

    let agent = AgentBuilder::new(model)
        .hook(TrackingHook {
            name: "hook1".to_string(),
            pre_calls: pre_calls.clone(),
            post_calls: post_calls.clone(),
        })
        .hook(TrackingHook {
            name: "hook2".to_string(),
            pre_calls: pre_calls.clone(),
            post_calls: post_calls.clone(),
        })
        .hook(TrackingHook {
            name: "hook3".to_string(),
            pre_calls: pre_calls.clone(),
            post_calls: post_calls.clone(),
        })
        .build();

    let _ = agent
        .completion(Message::user("Test"))
        .send()
        .await
        .unwrap();

    // Pre-hooks called in order
    assert_eq!(*pre_calls.lock().unwrap(), vec!["hook1", "hook2", "hook3"]);

    // Post-hooks also called in order (each hook receives output of previous hook per FR-003)
    assert_eq!(*post_calls.lock().unwrap(), vec!["hook1", "hook2", "hook3"]);
}

#[tokio::test]
async fn test_agent_with_preamble() {
    let model = MockModel::new(vec!["I am helpful."]);

    let agent = AgentBuilder::new(model)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .completion(Message::user("Hello"))
        .send()
        .await
        .unwrap();

    // Agent should successfully complete with preamble set
    assert_eq!(response.message.role(), "assistant");
}

#[tokio::test]
async fn test_agent_hook_chain_modifications() {
    let model = MockModel::new(vec!["Response"]);

    let agent = AgentBuilder::new(model.clone())
        .hook(PrefixHook {
            prefix: "A-".to_string(),
        })
        .hook(PrefixHook {
            prefix: "B-".to_string(),
        })
        .build();

    let _ = agent.completion(Message::user("X")).send().await.unwrap();

    // Chained prefixes should be applied
    let received = model.received_messages();
    assert_eq!(received[0], "B-A-X");
}

#[tokio::test]
async fn test_agent_with_history() {
    let model = MockModel::new(vec!["Second response"]);

    let agent = AgentBuilder::new(model).build();

    let history = vec![
        Message::user("First message"),
        Message::assistant("First response"),
    ];

    let response = agent
        .completion(Message::user("Second message"))
        .history(&history)
        .send()
        .await
        .unwrap();

    assert_eq!(response.message.role(), "assistant");
}
