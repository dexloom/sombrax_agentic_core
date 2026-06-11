//! Unit tests for HookChain sequential execution (T017)

use sombrax_agentic_core::context::HookContext;
use sombrax_agentic_core::error::HookResult;
use sombrax_agentic_core::hook::{Hook, HookChain};
use sombrax_agentic_core::message::Message;
#[allow(unused_imports)]
use sombrax_agentic_core::provider::CompletionResponse;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A hook that records the order it was called
#[derive(Clone)]
struct OrderTrackingHook {
    id: usize,
    call_order: Arc<AtomicUsize>,
    recorded_order: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl OrderTrackingHook {
    fn new(
        id: usize,
        call_order: Arc<AtomicUsize>,
        recorded_order: Arc<std::sync::Mutex<Vec<usize>>>,
    ) -> Self {
        Self {
            id,
            call_order,
            recorded_order,
        }
    }
}

impl Hook for OrderTrackingHook {
    fn pre_completion(
        &self,
        message: Message,
        _history: &[Message],
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        let _order = self.call_order.fetch_add(1, Ordering::SeqCst);
        self.recorded_order.lock().unwrap().push(self.id);
        async move { Ok(message) }
    }

    fn post_completion_message(
        &self,
        message: Message,
        _ctx: &mut HookContext,
    ) -> impl Future<Output = HookResult<Message>> + Send {
        self.recorded_order.lock().unwrap().push(self.id + 100); // +100 to distinguish post from pre
        async move { Ok(message) }
    }
}

/// A hook that modifies the message content
#[derive(Clone)]
struct PrefixHook {
    prefix: String,
}

impl PrefixHook {
    fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
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
                    let modified_content = content
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
                        content: modified_content,
                        id,
                    })
                }
                other => Ok(other),
            }
        }
    }
}

#[tokio::test]
async fn test_hook_chain_empty() {
    let chain: HookChain = HookChain::new();
    let mut ctx = HookContext::new("test-empty");
    let message = Message::user("Hello");

    let result = chain
        .execute_pre_completion(message.clone(), &[], &mut ctx)
        .await;
    assert!(result.is_ok());

    // Message should be unchanged
    if let Message::User { content, .. } = result.unwrap() {
        if let sombrax_agentic_core::message::UserContent::Text { text } = &content[0] {
            assert_eq!(text, "Hello");
        }
    }
}

#[tokio::test]
async fn test_hook_chain_single_hook() {
    let mut chain = HookChain::new();
    chain.add(PrefixHook::new("[PREFIX] "));

    let mut ctx = HookContext::new("test-single");
    let message = Message::user("Hello");

    let result = chain
        .execute_pre_completion(message, &[], &mut ctx)
        .await
        .unwrap();

    if let Message::User { content, .. } = result {
        if let sombrax_agentic_core::message::UserContent::Text { text } = &content[0] {
            assert_eq!(text, "[PREFIX] Hello");
        }
    }
}

#[tokio::test]
async fn test_hook_chain_multiple_hooks_order() {
    let call_order = Arc::new(AtomicUsize::new(0));
    let recorded_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    let mut chain = HookChain::new();
    chain.add(OrderTrackingHook::new(
        1,
        call_order.clone(),
        recorded_order.clone(),
    ));
    chain.add(OrderTrackingHook::new(
        2,
        call_order.clone(),
        recorded_order.clone(),
    ));
    chain.add(OrderTrackingHook::new(
        3,
        call_order.clone(),
        recorded_order.clone(),
    ));

    let mut ctx = HookContext::new("test-order");
    let message = Message::user("Hello");

    let _ = chain.execute_pre_completion(message, &[], &mut ctx).await;

    // Verify hooks were called in order 1, 2, 3
    let order = recorded_order.lock().unwrap();
    assert_eq!(*order, vec![1, 2, 3]);
}

#[tokio::test]
async fn test_hook_chain_chained_modifications() {
    let mut chain = HookChain::new();
    chain.add(PrefixHook::new("A-"));
    chain.add(PrefixHook::new("B-"));
    chain.add(PrefixHook::new("C-"));

    let mut ctx = HookContext::new("test-chained");
    let message = Message::user("X");

    let result = chain
        .execute_pre_completion(message, &[], &mut ctx)
        .await
        .unwrap();

    // Each hook adds its prefix to the result of the previous
    if let Message::User { content, .. } = result {
        if let sombrax_agentic_core::message::UserContent::Text { text } = &content[0] {
            assert_eq!(text, "C-B-A-X");
        }
    }
}

#[tokio::test]
async fn test_hook_chain_context_sharing() {
    /// Hook that sets a value in context
    #[derive(Clone)]
    struct SetterHook {
        key: String,
        value: String,
    }

    impl Hook for SetterHook {
        fn pre_completion(
            &self,
            message: Message,
            _history: &[Message],
            ctx: &mut HookContext,
        ) -> impl Future<Output = HookResult<Message>> + Send {
            let _ = ctx.set(&self.key, &self.value);
            async move { Ok(message) }
        }
    }

    /// Hook that reads a value from context
    #[derive(Clone)]
    struct ReaderHook {
        key: String,
        found: Arc<std::sync::Mutex<Option<String>>>,
    }

    impl Hook for ReaderHook {
        fn pre_completion(
            &self,
            message: Message,
            _history: &[Message],
            ctx: &mut HookContext,
        ) -> impl Future<Output = HookResult<Message>> + Send {
            if let Some(Ok(value)) = ctx.get::<String>(&self.key) {
                *self.found.lock().unwrap() = Some(value);
            }
            async move { Ok(message) }
        }
    }

    let found = Arc::new(std::sync::Mutex::new(None));

    let mut chain = HookChain::new();
    chain.add(SetterHook {
        key: "shared".to_string(),
        value: "passed!".to_string(),
    });
    chain.add(ReaderHook {
        key: "shared".to_string(),
        found: found.clone(),
    });

    let mut ctx = HookContext::new("test-context-sharing");
    let message = Message::user("Hello");

    let _ = chain.execute_pre_completion(message, &[], &mut ctx).await;

    // Second hook should have read the value set by first hook
    assert_eq!(*found.lock().unwrap(), Some("passed!".to_string()));
}
