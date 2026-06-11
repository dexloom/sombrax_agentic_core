//! Test Transcripts for File-History Context Management
//!
//! This module contains example conversation transcripts that validate the
//! ordering and deduplication behavior of the classification system.
//!
//! # Transcript Format
//!
//! Each transcript shows:
//! 1. Initial messages in the conversation
//! 2. Expected classification of each message
//! 3. Expected supersedence decisions
//! 4. Final context order after optimization
//!
//! # Test Scenarios
//!
//! - Read → Patch → Read sequences
//! - Multiple file operations interleaved
//! - Partial read handling
//! - Write supersedence
//! - Edge cases (renames, deletes, large files)

#[cfg(test)]
mod tests {
    use crate::context::classification::*;
    use std::path::PathBuf;

    /// Helper to create a test context manager with default rules
    fn test_manager() -> ContextManager {
        ContextManager::new()
    }

    /// Helper to create a manager with custom rules
    fn test_manager_with_rules(rules: SupersedenceRules) -> ContextManager {
        ContextManager::with_rules(rules)
    }

    // ========================================================================
    // Transcript 1: Simple Read → Edit → Read
    // ========================================================================

    /// Tests the basic read-edit-read pattern where:
    /// - Initial read should be dropped (superseded by final read)
    /// - Edit should be dropped (superseded by final read)
    /// - Final read should be kept and moved to end
    #[test]
    fn transcript_read_edit_read() {
        let mut manager = test_manager();

        // === Conversation Transcript ===
        //
        // [msg-1] Assistant: "Let me read the file"
        //         ToolCall(read, {file_path: "/src/main.rs"})
        //
        // [msg-2] User: ToolResult for msg-1
        //         Content: "fn main() { old_code(); }"
        //
        // [msg-3] Assistant: "I'll edit the file"
        //         ToolCall(edit, {file_path: "/src/main.rs", old: "old_code", new: "new_code"})
        //
        // [msg-4] User: ToolResult for msg-3
        //         Content: "Edit successful"
        //
        // [msg-5] Assistant: "Let me verify the change"
        //         ToolCall(read, {file_path: "/src/main.rs"})
        //
        // [msg-6] User: ToolResult for msg-5
        //         Content: "fn main() { new_code(); }"

        // Classify each operation
        let read_args_1 = serde_json::json!({"file_path": "/src/main.rs"});
        manager
            .classifier_mut()
            .classify_tool_call("read", &read_args_1, "msg-1", "call-1");
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/main.rs","content":"fn main() { old_code(); }"}"#,
            "msg-2",
            "call-1",
        );

        let edit_args = serde_json::json!({
            "file_path": "/src/main.rs",
            "old_string": "old_code",
            "new_string": "new_code"
        });
        manager
            .classifier_mut()
            .classify_tool_call("edit", &edit_args, "msg-3", "call-2");
        manager.classifier_mut().classify_tool_result(
            "edit",
            r#"{"file_path":"/src/main.rs","replacements":1}"#,
            "msg-4",
            "call-2",
        );

        let read_args_2 = serde_json::json!({"file_path": "/src/main.rs"});
        manager
            .classifier_mut()
            .classify_tool_call("read", &read_args_2, "msg-5", "call-3");
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/main.rs","content":"fn main() { new_code(); }"}"#,
            "msg-6",
            "call-3",
        );

        // Compute optimization
        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // Dropped messages:
        // - msg-2: First read result (superseded by msg-6)
        // - msg-4: Edit result (superseded by msg-6 read)
        //
        // Kept messages:
        // - msg-1, msg-3, msg-5: Tool calls (always kept for context)
        // - msg-6: Latest read (moved to end)

        assert!(
            result.should_drop("msg-2"),
            "First read result should be dropped"
        );
        assert!(result.should_drop("msg-4"), "Edit result should be dropped");
        assert!(
            result.should_include("msg-6"),
            "Final read should be included"
        );

        // Verify final order includes latest read at end
        let final_order = result.final_order();
        assert!(
            final_order.last().map(|k| k.message_id.as_str()) == Some("msg-6")
                || result.move_to_end.iter().any(|k| k.message_id == "msg-6"),
            "Latest read should be at end of context"
        );
    }

    // ========================================================================
    // Transcript 2: Multiple Files Interleaved
    // ========================================================================

    /// Tests handling of operations on multiple files:
    /// - Operations on different files don't affect each other
    /// - Each file tracks its own supersedence independently
    #[test]
    fn transcript_multiple_files() {
        let mut manager = test_manager();

        // === Conversation Transcript ===
        //
        // [msg-1] Read file_a.rs
        // [msg-2] Result: content of file_a.rs v1
        // [msg-3] Read file_b.rs
        // [msg-4] Result: content of file_b.rs v1
        // [msg-5] Edit file_a.rs
        // [msg-6] Result: edit success
        // [msg-7] Read file_a.rs (verification)
        // [msg-8] Result: content of file_a.rs v2

        // File A operations
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file_a.rs"}),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file_a.rs","content":"v1"}"#,
            "msg-2",
            "call-1",
        );

        // File B operations
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file_b.rs"}),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file_b.rs","content":"v1"}"#,
            "msg-4",
            "call-2",
        );

        // Edit file A
        manager.classifier_mut().classify_tool_call(
            "edit",
            &serde_json::json!({
                "file_path": "/src/file_a.rs",
                "old_string": "old",
                "new_string": "new"
            }),
            "msg-5",
            "call-3",
        );
        manager.classifier_mut().classify_tool_result(
            "edit",
            r#"{"file_path":"/src/file_a.rs"}"#,
            "msg-6",
            "call-3",
        );

        // Read file A again
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file_a.rs"}),
            "msg-7",
            "call-4",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file_a.rs","content":"v2"}"#,
            "msg-8",
            "call-4",
        );

        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // File A: msg-2 (old read) dropped, msg-8 (new read) kept
        // File B: msg-4 kept (no superseding operation)

        assert!(
            result.should_drop("msg-2"),
            "Old read of file_a should be dropped"
        );
        assert!(
            result.should_include("msg-4"),
            "Read of file_b should be kept (no newer read)"
        );
        assert!(
            result.should_include("msg-8"),
            "New read of file_a should be kept"
        );
    }

    // ========================================================================
    // Transcript 3: Write Supersedes All
    // ========================================================================

    /// Tests that a write operation supersedes all prior reads and edits
    #[test]
    fn transcript_write_supersedes() {
        let mut manager = test_manager();

        // === Conversation Transcript ===
        //
        // [msg-1] Read file.rs
        // [msg-2] Result: content v1
        // [msg-3] Edit file.rs
        // [msg-4] Result: edit success
        // [msg-5] Write file.rs (complete rewrite)
        // [msg-6] Result: write success

        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"v1"}"#,
            "msg-2",
            "call-1",
        );

        manager.classifier_mut().classify_tool_call(
            "edit",
            &serde_json::json!({
                "file_path": "/src/file.rs",
                "old_string": "a",
                "new_string": "b"
            }),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "edit",
            r#"{"file_path":"/src/file.rs"}"#,
            "msg-4",
            "call-2",
        );

        manager.classifier_mut().classify_tool_call(
            "write",
            &serde_json::json!({
                "file_path": "/src/file.rs",
                "content": "completely new content"
            }),
            "msg-5",
            "call-3",
        );
        manager.classifier_mut().classify_tool_result(
            "write",
            r#"{"file_path":"/src/file.rs","bytes_written":22}"#,
            "msg-6",
            "call-3",
        );

        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // All prior operations on the file should be dropped
        // Only the write result should be kept (as latest snapshot)

        assert!(
            result.should_drop("msg-2"),
            "Read before write should be dropped"
        );
        assert!(
            result.should_drop("msg-4"),
            "Edit before write should be dropped"
        );
        assert!(
            result.should_include("msg-6"),
            "Write result should be kept"
        );
    }

    // ========================================================================
    // Transcript 4: Partial Reads
    // ========================================================================

    /// Tests handling of partial reads with offset/limit
    #[test]
    fn transcript_partial_reads() {
        let mut manager = test_manager();

        // === Conversation Transcript ===
        //
        // [msg-1] Read file.rs lines 0-50
        // [msg-2] Result: partial content
        // [msg-3] Read file.rs lines 100-150
        // [msg-4] Result: partial content
        // [msg-5] Read file.rs (full)
        // [msg-6] Result: full content

        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({
                "file_path": "/src/file.rs",
                "offset": 0,
                "limit": 50
            }),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"lines 0-50"}"#,
            "msg-2",
            "call-1",
        );

        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({
                "file_path": "/src/file.rs",
                "offset": 100,
                "limit": 50
            }),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"lines 100-150"}"#,
            "msg-4",
            "call-2",
        );

        // Full read supersedes partial reads
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-5",
            "call-3",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"full file content"}"#,
            "msg-6",
            "call-3",
        );

        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // Partial reads should be dropped when full read exists

        assert!(
            result.should_drop("msg-2"),
            "Partial read 0-50 should be dropped"
        );
        assert!(
            result.should_drop("msg-4"),
            "Partial read 100-150 should be dropped"
        );
        assert!(result.should_include("msg-6"), "Full read should be kept");
    }

    // ========================================================================
    // Transcript 5: Read-After-Write Enforcement
    // ========================================================================

    /// Tests that the system identifies when read-after-write is needed
    #[test]
    fn transcript_read_after_write_needed() {
        let mut manager = test_manager();

        // === Conversation Transcript ===
        //
        // [msg-1] Write file.rs
        // [msg-2] Result: write success
        // (No subsequent read)

        manager.classifier_mut().classify_tool_call(
            "write",
            &serde_json::json!({
                "file_path": "/src/file.rs",
                "content": "new content"
            }),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "write",
            r#"{"file_path":"/src/file.rs"}"#,
            "msg-2",
            "call-1",
        );

        let path = PathBuf::from("/src/file.rs");

        // === Expected Results ===
        //
        // File should be flagged as needing read-after-write

        assert!(
            manager.needs_read_after_write(&path),
            "Write without read should need read-after-write"
        );

        // Now add a read
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"new content"}"#,
            "msg-4",
            "call-2",
        );

        assert!(
            !manager.needs_read_after_write(&path),
            "After read, should not need read-after-write"
        );
    }

    // ========================================================================
    // Transcript 6: Preserve Edit History Mode
    // ========================================================================

    /// Tests behavior when edit history preservation is enabled
    #[test]
    fn transcript_preserve_edit_history() {
        let rules = SupersedenceRules {
            preserve_edit_history: true,
            max_edits_per_file: 3,
            ..Default::default()
        };
        let mut manager = test_manager_with_rules(rules);

        // === Conversation Transcript ===
        //
        // Multiple edits without a final read

        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs","content":"v1"}"#,
            "msg-2",
            "call-1",
        );

        // Multiple edits
        for i in 0..4 {
            let msg_call = format!("msg-{}", 3 + i * 2);
            let msg_result = format!("msg-{}", 4 + i * 2);
            let call_id = format!("call-{}", 2 + i);

            manager.classifier_mut().classify_tool_call(
                "edit",
                &serde_json::json!({
                    "file_path": "/src/file.rs",
                    "old_string": format!("v{}", i + 1),
                    "new_string": format!("v{}", i + 2)
                }),
                &msg_call,
                &call_id,
            );
            manager.classifier_mut().classify_tool_result(
                "edit",
                &format!(r#"{{"file_path":"/src/file.rs","version":"v{}"}}"#, i + 2),
                &msg_result,
                &call_id,
            );
        }

        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // With preserve_edit_history=true and max_edits=3:
        // - Original read should be kept (no superseding read)
        // - First edit (msg-4) should be dropped (older than max_edits)
        // - Last 3 edits should be kept

        // Count how many edit results are kept
        let kept_edits: Vec<_> = ["msg-4", "msg-6", "msg-8", "msg-10"]
            .iter()
            .filter(|id| result.should_include(id))
            .collect();

        assert!(
            kept_edits.len() <= 3,
            "Should keep at most 3 edits, got {}",
            kept_edits.len()
        );
    }

    // ========================================================================
    // Transcript 7: Context Optimization Result Structure
    // ========================================================================

    /// Tests the structure of the optimization result
    #[test]
    fn transcript_optimization_result_structure() {
        let mut manager = test_manager();

        // Simple read-edit-read
        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs"}"#,
            "msg-2",
            "call-1",
        );

        manager.classifier_mut().classify_tool_call(
            "read",
            &serde_json::json!({"file_path": "/src/file.rs"}),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "read",
            r#"{"file_path":"/src/file.rs"}"#,
            "msg-4",
            "call-2",
        );

        let result = manager.compute_optimized_order();

        // Verify result structure
        assert!(
            !result.keep.is_empty() || !result.move_to_end.is_empty(),
            "Should have some messages to keep"
        );

        // Final order should not contain duplicates
        let final_order = result.final_order();
        let unique: std::collections::HashSet<_> = final_order.iter().collect();
        assert_eq!(
            final_order.len(),
            unique.len(),
            "Final order should not have duplicates"
        );

        // Dropped classification keys should not appear in final order
        for dropped in &result.drop {
            assert!(
                !final_order.contains(dropped),
                "Dropped classification key {:?} should not be in final order",
                dropped
            );
        }
    }

    // ========================================================================
    // Transcript 8: Non-File Operations Unaffected
    // ========================================================================

    /// Tests that non-file operations are not affected by optimization
    #[test]
    fn transcript_non_file_operations() {
        let mut manager = test_manager();

        // Non-file tool calls
        manager.classifier_mut().classify_tool_call(
            "web_search",
            &serde_json::json!({"query": "rust tutorials"}),
            "msg-1",
            "call-1",
        );
        manager.classifier_mut().classify_tool_result(
            "web_search",
            r#"{"results": ["result1", "result2"]}"#,
            "msg-2",
            "call-1",
        );

        manager.classifier_mut().classify_tool_call(
            "bash",
            &serde_json::json!({"command": "ls -la"}),
            "msg-3",
            "call-2",
        );
        manager.classifier_mut().classify_tool_result(
            "bash",
            r#"total 100\ndrwxr-xr-x..."#,
            "msg-4",
            "call-2",
        );

        let result = manager.compute_optimized_order();

        // === Expected Results ===
        //
        // All non-file operations should be kept

        assert!(
            result.should_include("msg-1"),
            "Non-file tool calls should be kept"
        );
        assert!(
            result.should_include("msg-2"),
            "Non-file tool results should be kept"
        );
        assert!(
            result.should_include("msg-3"),
            "Bash tool calls should be kept"
        );
        assert!(
            result.should_include("msg-4"),
            "Bash tool results should be kept"
        );
    }

    // ========================================================================
    // Final Order Visualization
    // ========================================================================

    /// Helper to print the optimization result for debugging
    #[allow(dead_code)]
    fn visualize_result(result: &ContextOptimizationResult) -> String {
        let mut output = String::new();
        output.push_str("=== Context Optimization Result ===\n\n");

        output.push_str("KEPT (in place):\n");
        for key in &result.keep {
            output.push_str(&format!("  - {}\n", key));
        }

        output.push_str("\nDROPPED:\n");
        for key in &result.drop {
            output.push_str(&format!("  - {}\n", key));
        }

        output.push_str("\nMOVED TO END:\n");
        for key in &result.move_to_end {
            output.push_str(&format!("  - {}\n", key));
        }

        output.push_str("\nFINAL ORDER:\n");
        for (i, key) in result.final_order().iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", i + 1, key));
        }

        output
    }
}
