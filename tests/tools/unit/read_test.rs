//! Unit tests for ReadTool

use sombrax_agentic_core::tools::context::ToolContext;
use sombrax_agentic_core::tools::file::{ReadArgs, ReadTool};
use sombrax_agentic_core::tools::registry::Tool;
use std::fs;
use tempfile::TempDir;

fn create_test_context() -> (TempDir, ToolContext) {
    let temp_dir = TempDir::new().unwrap();
    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    (temp_dir, ctx)
}

#[tokio::test]
async fn test_read_tool_definition() {
    let (_temp_dir, ctx) = create_test_context();
    let tool = ReadTool::new(ctx);
    let def = tool.definition("".to_string()).await;

    assert_eq!(def.name, "read");
    assert!(!def.description.is_empty());
}

#[tokio::test]
async fn test_read_tool_simple_file() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "Hello, World!\nLine 2\nLine 3").unwrap();

    let tool = ReadTool::new(ctx);
    let result = tool
        .call(ReadArgs {
            file_path: file_path.to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.content.contains("Hello, World!"));
    assert_eq!(output.total_lines, 3);
}

#[tokio::test]
async fn test_read_tool_with_offset() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "Line 1\nLine 2\nLine 3\nLine 4\nLine 5").unwrap();

    let tool = ReadTool::new(ctx);
    let result = tool
        .call(ReadArgs {
            file_path: file_path.to_string_lossy().to_string(),
            description: None,
            offset: Some(2),
            limit: Some(2),
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.content.contains("Line 3"));
    assert!(output.content.contains("Line 4"));
}

#[tokio::test]
async fn test_read_tool_file_not_found() {
    let (temp_dir, ctx) = create_test_context();
    let tool = ReadTool::new(ctx);

    let result = tool
        .call(ReadArgs {
            file_path: temp_dir
                .path()
                .join("nonexistent.txt")
                .to_string_lossy()
                .to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_tool_empty_file() {
    let (temp_dir, ctx) = create_test_context();
    let file_path = temp_dir.path().join("empty.txt");
    fs::write(&file_path, "").unwrap();

    let tool = ReadTool::new(ctx);
    let result = tool
        .call(ReadArgs {
            file_path: file_path.to_string_lossy().to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.total_lines, 0);
}

#[tokio::test]
async fn test_read_tool_duplicated_workspace_folder() {
    // Simulate: CWD is /tmp/0xABC123/ and path is "0xABC123/src/file.sol"
    // Should resolve to /tmp/0xABC123/src/file.sol
    let temp_dir = TempDir::new().unwrap();
    let workspace_name = "0x46f54d434063e5F1a2b2CC6d9AAa657b1B9ff82c";
    let workspace = temp_dir.path().join(workspace_name);
    fs::create_dir_all(workspace.join("src/cauldrons")).unwrap();

    let file_path = workspace.join("src/cauldrons/CauldronV4.sol");
    fs::write(&file_path, "// Cauldron contract").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), workspace.clone());
    let tool = ReadTool::new(ctx);

    // Path with duplicated workspace folder name
    let malformed_path = format!("{}/src/cauldrons/CauldronV4.sol", workspace_name);

    let result = tool
        .call(ReadArgs {
            file_path: malformed_path,
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(result.is_ok(), "Should resolve duplicated workspace path");
    let output = result.unwrap();
    assert!(output.content.contains("Cauldron contract"));
}

#[tokio::test]
async fn test_read_tool_missing_leading_slash() {
    // This test simulates a path that looks absolute but is missing the leading /
    // e.g., "Users/alice/project/..." instead of "/Users/alice/project/..."
    let temp_dir = TempDir::new().unwrap();
    let subdir = temp_dir.path().join("subdir");
    fs::create_dir(&subdir).unwrap();
    let file_path = subdir.join("test.sol");
    fs::write(&file_path, "// Test content").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    let tool = ReadTool::new(ctx);

    // Get the path without leading /
    let full_path = file_path.to_string_lossy().to_string();
    let path_without_slash = full_path.trim_start_matches('/');

    let result = tool
        .call(ReadArgs {
            file_path: path_without_slash.to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(
        result.is_ok(),
        "Should resolve path missing leading slash: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(output.content.contains("Test content"));
}

#[tokio::test]
async fn test_context_path_normalization() {
    let temp_dir = TempDir::new().unwrap();
    let workspace_name = "my_workspace";
    let workspace = temp_dir.path().join(workspace_name);
    fs::create_dir_all(workspace.join("src")).unwrap();

    let file_path = workspace.join("src/main.rs");
    fs::write(&file_path, "fn main() {}").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), workspace.clone());

    // Test 1: Direct path should work
    let result = ctx.validate_path(&file_path.to_string_lossy());
    assert!(result.is_ok());

    // Test 2: Relative path should work
    let result = ctx.validate_path("src/main.rs");
    assert!(result.is_ok());

    // Test 3: Duplicated workspace name should work
    let result = ctx.validate_path(&format!("{}/src/main.rs", workspace_name));
    assert!(result.is_ok(), "Should handle duplicated workspace name");

    // Test 4: Path with workspace name in middle should work
    let mangled_path = format!("prefix/{}/src/main.rs", workspace_name);
    let result = ctx.validate_path(&mangled_path);
    assert!(result.is_ok(), "Should handle workspace name in path");
}

#[tokio::test]
async fn test_fuzzy_file_search_unique_file() {
    // When there's only one file with that name, it should be found
    let temp_dir = TempDir::new().unwrap();
    fs::create_dir_all(temp_dir.path().join("src/contracts")).unwrap();

    let file_path = temp_dir.path().join("src/contracts/UniqueContract.sol");
    fs::write(&file_path, "// Unique contract").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    let tool = ReadTool::new(ctx);

    // Use a completely wrong path but correct filename
    let result = tool
        .call(ReadArgs {
            file_path: "wrong/path/to/UniqueContract.sol".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(
        result.is_ok(),
        "Should find unique file by name: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(output.content.contains("Unique contract"));
}

#[tokio::test]
async fn test_fuzzy_file_search_multiple_files_uses_hints() {
    // When multiple files have the same name, use directory hints to pick the right one
    let temp_dir = TempDir::new().unwrap();
    fs::create_dir_all(temp_dir.path().join("src/cauldrons")).unwrap();
    fs::create_dir_all(temp_dir.path().join("test/mocks")).unwrap();

    let file1 = temp_dir.path().join("src/cauldrons/CauldronV4.sol");
    let file2 = temp_dir.path().join("test/mocks/CauldronV4.sol");
    fs::write(&file1, "// Real Cauldron").unwrap();
    fs::write(&file2, "// Mock Cauldron").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    let tool = ReadTool::new(ctx);

    // Path with "cauldrons" hint should find the real one
    let result = tool
        .call(ReadArgs {
            file_path: "some/cauldrons/CauldronV4.sol".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(
        result.is_ok(),
        "Should find file using directory hints: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(output.content.contains("Real Cauldron"));
}

#[tokio::test]
async fn test_fuzzy_file_search_with_typo() {
    // Should find file even with minor typo in filename
    let temp_dir = TempDir::new().unwrap();
    fs::create_dir_all(temp_dir.path().join("src")).unwrap();

    let file_path = temp_dir.path().join("src/CauldronV4.sol");
    fs::write(&file_path, "// Cauldron content").unwrap();

    let ctx = ToolContext::new("test-session".to_string(), temp_dir.path().to_path_buf());
    let tool = ReadTool::new(ctx);

    // Typo: CauldornV4 instead of CauldronV4
    let result = tool
        .call(ReadArgs {
            file_path: "src/CauldornV4.sol".to_string(),
            description: None,
            offset: None,
            limit: None,
        })
        .await;

    assert!(
        result.is_ok(),
        "Should find file despite typo: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(output.content.contains("Cauldron content"));
}
