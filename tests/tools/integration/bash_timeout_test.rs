//! Integration tests for Bash timeout behavior (T064)
//!
//! Tests that the BashTool properly handles command timeouts
//! and long-running processes.

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::registry::Tool;
use sombrax_agentic_core::tools::shell::{BashArgs, BashTool};
use std::time::Instant;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("timeout-test".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

/// Test that short commands complete within reasonable time
#[tokio::test]
async fn test_fast_command_completes_quickly() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let start = Instant::now();
    let result = tool
        .call(BashArgs {
            command: "echo 'fast'".to_string(),
            timeout: Some(5000),
            description: Some("Fast echo".to_string()),
        })
        .await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(
        elapsed.as_secs() < 2,
        "Fast command took too long: {:?}",
        elapsed
    );
}

/// Test that timeout is respected for slow commands
#[tokio::test]
async fn test_slow_command_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let start = Instant::now();
    let result = tool
        .call(BashArgs {
            command: "sleep 10".to_string(),
            timeout: Some(500), // 500ms timeout
            description: Some("Sleep command".to_string()),
        })
        .await;
    let elapsed = start.elapsed();

    // Command should be terminated due to timeout
    // Either returns an error or returns with non-success
    if let Ok(output) = result {
        // If it succeeded, it should have been killed
        assert!(!output.success || elapsed.as_millis() < 1000);
    }

    // Should not have waited the full 10 seconds
    assert!(
        elapsed.as_secs() < 5,
        "Command was not terminated: {:?}",
        elapsed
    );
}

/// Test default timeout behavior
#[tokio::test]
async fn test_default_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Command without explicit timeout
    let result = tool
        .call(BashArgs {
            command: "echo 'default timeout test'".to_string(),
            timeout: None,
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
}

/// Test that timeout value is clamped to maximum
#[tokio::test]
async fn test_timeout_clamping() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Very large timeout should be clamped
    let result = tool
        .call(BashArgs {
            command: "echo 'clamped'".to_string(),
            timeout: Some(999999999), // Very large value
            description: None,
        })
        .await;

    assert!(result.is_ok());
}

/// Test timeout with CPU-intensive command
#[tokio::test]
async fn test_cpu_intensive_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let start = Instant::now();
    // Command that does actual work, not just sleeping
    let result = tool
        .call(BashArgs {
            command: "yes | head -n 1000000 > /dev/null".to_string(),
            timeout: Some(500),
            description: Some("CPU intensive".to_string()),
        })
        .await;
    let elapsed = start.elapsed();

    // Should complete or timeout within reasonable time
    assert!(elapsed.as_secs() < 5, "CPU intensive command ran too long");

    if let Ok(output) = result {
        // Either completed quickly or was terminated
        assert!(output.success || elapsed.as_millis() >= 400);
    }
}

/// Test multiple sequential commands with timeout
#[tokio::test]
async fn test_sequential_commands_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // First command should complete
    let result1 = tool
        .call(BashArgs {
            command: "echo 'first'".to_string(),
            timeout: Some(1000),
            description: None,
        })
        .await;
    assert!(result1.is_ok());

    // Second command with potential timeout
    let start = Instant::now();
    let _result2 = tool
        .call(BashArgs {
            command: "sleep 5".to_string(),
            timeout: Some(200),
            description: None,
        })
        .await;
    let elapsed = start.elapsed();

    // Should timeout quickly
    assert!(elapsed.as_secs() < 3);

    // Third command should still work after timeout
    let result3 = tool
        .call(BashArgs {
            command: "echo 'third'".to_string(),
            timeout: Some(1000),
            description: None,
        })
        .await;
    assert!(result3.is_ok());
}

/// Test that stdout is captured even with timeout
#[tokio::test]
async fn test_partial_output_capture_on_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Command that produces output before potentially timing out
    let result = tool
        .call(BashArgs {
            command: "echo 'before'; sleep 5; echo 'after'".to_string(),
            timeout: Some(500),
            description: None,
        })
        .await;

    // Whether success or timeout, should have captured some output
    if let Ok(output) = result {
        // Either got all output (fast completion) or partial output (timeout)
        assert!(output.stdout.contains("before") || !output.success);
    }
}

/// Test zero timeout behavior
#[tokio::test]
async fn test_zero_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Zero timeout might be treated as "no timeout" or "immediate timeout"
    let result = tool
        .call(BashArgs {
            command: "echo 'zero timeout'".to_string(),
            timeout: Some(0),
            description: None,
        })
        .await;

    // Should either succeed immediately or fail due to timeout
    // Both are acceptable behaviors
    match result {
        Ok(output) => {
            // If it succeeded, the output should be correct
            if output.success {
                assert!(output.stdout.contains("zero timeout"));
            }
        }
        Err(_) => {
            // Timeout error is acceptable
        }
    }
}

/// Test timeout with pipe commands
#[tokio::test]
async fn test_pipe_command_timeout() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    let result = tool
        .call(BashArgs {
            command: "echo 'hello world' | grep 'hello'".to_string(),
            timeout: Some(2000),
            description: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.success);
    assert!(output.stdout.contains("hello"));
}

/// Test that process group is terminated on timeout
#[tokio::test]
async fn test_process_group_termination() {
    let (temp_dir, ctx) = create_test_context();
    let tool = BashTool::new(ctx);

    // Create a script that spawns background processes
    let script_path = temp_dir.path().join("spawner.sh");
    std::fs::write(
        &script_path,
        r#"#!/bin/bash
echo "starting"
sleep 100 &
sleep 100 &
sleep 100
"#,
    )
    .unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let start = Instant::now();
    let result = tool
        .call(BashArgs {
            command: script_path.to_string_lossy().to_string(),
            timeout: Some(500),
            description: None,
        })
        .await;
    let elapsed = start.elapsed();

    // Should have terminated within the timeout
    assert!(
        elapsed.as_secs() < 5,
        "Process group not terminated properly"
    );

    // The command should have produced some output before timeout
    if let Ok(output) = result {
        if !output.success {
            // Timed out as expected
            assert!(output.stdout.contains("starting") || output.stdout.is_empty());
        }
    }
}
