//! Unit tests for GlobTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{GlobArgs, GlobTool};
use sombrax_agentic_core::tools::registry::Tool;
use std::fs;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_glob_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = GlobTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "glob");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_glob_tool_simple_pattern() {
    let (temp_dir, ctx) = create_test_context();

    // Create test files
    fs::write(temp_dir.path().join("file1.txt"), "content").unwrap();
    fs::write(temp_dir.path().join("file2.txt"), "content").unwrap();
    fs::write(temp_dir.path().join("file.rs"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "*.txt".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.matches.len(), 2);
}

#[tokio::test]
async fn test_glob_tool_recursive_pattern() {
    let (temp_dir, ctx) = create_test_context();

    // Create nested structure
    let subdir = temp_dir.path().join("src");
    fs::create_dir(&subdir).unwrap();
    fs::write(temp_dir.path().join("root.rs"), "content").unwrap();
    fs::write(subdir.join("lib.rs"), "content").unwrap();
    fs::write(subdir.join("main.rs"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "**/*.rs".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.matches.len(), 3);
}

#[tokio::test]
async fn test_glob_tool_no_matches() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("file.txt"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "*.rs".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.matches.is_empty());
}

#[tokio::test]
async fn test_glob_tool_default_path() {
    let (temp_dir, ctx) = create_test_context();

    fs::write(temp_dir.path().join("test.txt"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "*.txt".to_string(),
            description: None,
            path: None, // Uses workspace directory
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_glob_tool_max_results() {
    let (temp_dir, ctx) = create_test_context();

    // Create many files
    for i in 0..10 {
        fs::write(temp_dir.path().join(format!("file{}.txt", i)), "content").unwrap();
    }

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "*.txt".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: Some(3),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.matches.len(), 3);
    assert!(output.truncated);
}

#[tokio::test]
async fn test_glob_tool_excludes_default_patterns() {
    let (temp_dir, ctx) = create_test_context();

    // Create files in normal and excluded directories
    fs::write(temp_dir.path().join("main.rs"), "content").unwrap();

    let node_modules = temp_dir.path().join("node_modules");
    fs::create_dir(&node_modules).unwrap();
    fs::write(node_modules.join("package.rs"), "content").unwrap();

    let target = temp_dir.path().join("target");
    fs::create_dir(&target).unwrap();
    fs::write(target.join("build.rs"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "**/*.rs".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Should only find main.rs, not files in node_modules or target
    assert_eq!(output.matches.len(), 1);
    assert!(output.matches[0].contains("main.rs"));
}

#[tokio::test]
async fn test_glob_tool_custom_excludes() {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());

    // Add custom exclude
    ctx.add_exclude("secrets");

    // Create files
    fs::write(temp_dir.path().join("config.txt"), "content").unwrap();

    let secrets = temp_dir.path().join("secrets");
    fs::create_dir(&secrets).unwrap();
    fs::write(secrets.join("api_keys.txt"), "content").unwrap();

    let tool = GlobTool::new(ctx);
    let result = tool
        .call(GlobArgs {
            pattern: "**/*.txt".to_string(),
            description: None,
            path: Some(temp_dir.path().to_string_lossy().to_string()),
            max_results: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Should only find config.txt, not files in secrets
    assert_eq!(output.matches.len(), 1);
    assert!(output.matches[0].contains("config.txt"));
}

#[tokio::test]
async fn test_context_exclude_management() {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());

    // Check default excludes include common patterns
    let excludes = ctx.get_excludes();
    assert!(excludes.contains(&"node_modules".to_string()));
    assert!(excludes.contains(&".git".to_string()));
    assert!(excludes.contains(&"target".to_string()));

    // Add custom exclude
    ctx.add_exclude("my_secret_folder");
    assert!(ctx.get_excludes().contains(&"my_secret_folder".to_string()));

    // Remove an exclude
    assert!(ctx.remove_exclude("target"));
    assert!(!ctx.get_excludes().contains(&"target".to_string()));

    // Clear all excludes
    ctx.clear_excludes();
    assert!(ctx.get_excludes().is_empty());

    // Reset to defaults
    ctx.reset_excludes();
    assert!(ctx.get_excludes().contains(&"node_modules".to_string()));
}
