//! Unit tests for HookContext get/set/remove operations (T016)

use sombrax_agentic_core::context::HookContext;
use std::thread;
use std::time::Duration;

#[test]
fn test_hook_context_creation() {
    let ctx = HookContext::new("test-request-123");
    assert_eq!(ctx.request_id, "test-request-123");
    assert!(!ctx.is_cancelled());
}

#[test]
fn test_hook_context_set_and_get() {
    let mut ctx = HookContext::new("test-1");

    // Set string value
    ctx.set("key1", "value1").unwrap();
    let result: String = ctx.get("key1").unwrap().unwrap();
    assert_eq!(result, "value1");

    // Set numeric value
    ctx.set("count", 42i32).unwrap();
    let count: i32 = ctx.get("count").unwrap().unwrap();
    assert_eq!(count, 42);

    // Set complex value
    ctx.set("data", serde_json::json!({"nested": true}))
        .unwrap();
    let data: serde_json::Value = ctx.get("data").unwrap().unwrap();
    assert_eq!(data["nested"], true);
}

#[test]
fn test_hook_context_get_nonexistent() {
    let ctx = HookContext::new("test-2");
    let result: Option<Result<String, _>> = ctx.get("nonexistent");
    assert!(result.is_none());
}

#[test]
fn test_hook_context_remove() {
    let mut ctx = HookContext::new("test-3");

    ctx.set("to_remove", "value").unwrap();
    assert!(ctx.contains("to_remove"));

    let removed = ctx.remove("to_remove");
    assert!(removed.is_some());
    assert!(!ctx.contains("to_remove"));
}

#[test]
fn test_hook_context_contains() {
    let mut ctx = HookContext::new("test-4");

    assert!(!ctx.contains("key"));
    ctx.set("key", "value").unwrap();
    assert!(ctx.contains("key"));
}

#[test]
fn test_hook_context_elapsed() {
    let ctx = HookContext::new("test-5");

    // Sleep briefly to ensure elapsed time > 0
    thread::sleep(Duration::from_millis(10));

    let elapsed = ctx.elapsed();
    assert!(elapsed >= Duration::from_millis(10));
}

#[test]
fn test_hook_context_cancellation() {
    let ctx = HookContext::new("test-6");

    assert!(!ctx.is_cancelled());

    ctx.cancel_signal.cancel();
    assert!(ctx.is_cancelled());

    ctx.cancel_signal.reset();
    assert!(!ctx.is_cancelled());
}

#[test]
fn test_hook_context_keys() {
    let mut ctx = HookContext::new("test-7");

    ctx.set("a", 1).unwrap();
    ctx.set("b", 2).unwrap();
    ctx.set("c", 3).unwrap();

    let keys: Vec<&str> = ctx.keys().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"a"));
    assert!(keys.contains(&"b"));
    assert!(keys.contains(&"c"));
}

#[test]
fn test_hook_context_overwrite() {
    let mut ctx = HookContext::new("test-8");

    ctx.set("key", "first").unwrap();
    let val1: String = ctx.get("key").unwrap().unwrap();
    assert_eq!(val1, "first");

    ctx.set("key", "second").unwrap();
    let val2: String = ctx.get("key").unwrap().unwrap();
    assert_eq!(val2, "second");
}

#[test]
fn test_hook_context_uuid_generation() {
    let ctx1 = HookContext::new_with_uuid();
    let ctx2 = HookContext::new_with_uuid();

    // UUIDs should be unique
    assert_ne!(ctx1.request_id, ctx2.request_id);

    // UUID format validation (36 chars with hyphens)
    assert_eq!(ctx1.request_id.len(), 36);
}
