//! Integration tests for `sombrax_agentic_core::prompt`.

use sombrax_agentic_core::prompt::{PromptRegistry, SystemPrompt};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

async fn write_prompt(dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(format!("{name}.md"));
    fs::write(&path, body).await.expect("write prompt");
    path
}

#[tokio::test]
async fn resolve_exact_hit() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "whitehut", "# white\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg.resolve("whitehut").expect("hit");
    assert_eq!(p.name(), "whitehut");
    assert!(p.body().await.unwrap().contains("# white"));
}

#[tokio::test]
async fn resolve_dot_to_underscore() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "whitehut-glm-5_1", "# glm 5.1 body\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg
        .resolve("whitehut-glm-5.1")
        .expect("hit via dot->underscore rung");
    assert_eq!(p.name(), "whitehut-glm-5_1");
}

#[tokio::test]
async fn resolve_single_strip() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "whitehut", "# base\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg
        .resolve("whitehut-claude")
        .expect("falls back to whitehut");
    assert_eq!(p.name(), "whitehut");
}

#[tokio::test]
async fn resolve_multi_strip() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "whitehut", "# base\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg
        .resolve("whitehut-glm-5.1")
        .expect("falls all the way back to base");
    assert_eq!(p.name(), "whitehut");
}

#[tokio::test]
async fn resolve_miss() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "other", "# other\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    assert!(reg.resolve("whitehut").is_none());
    assert!(reg.resolve("whitehut-glm-5.1").is_none());
}

#[tokio::test]
async fn multi_path_last_wins() {
    let low = TempDir::new().unwrap();
    let high = TempDir::new().unwrap();
    write_prompt(low.path(), "whitehut", "low body\n").await;
    write_prompt(high.path(), "whitehut", "high body\n").await;

    let reg = PromptRegistry::discover(vec![low.path().into(), high.path().into()])
        .await
        .unwrap();
    let p = reg.resolve("whitehut").unwrap();
    let body = p.body().await.unwrap();
    assert!(
        body.contains("high body"),
        "expected higher-priority layer to win, got: {body:?}"
    );
}

#[tokio::test]
async fn frontmatter_parsed() {
    let tmp = TempDir::new().unwrap();
    let content = "---\ndescription: short\n---\n\n# body\n";
    write_prompt(tmp.path(), "whitehut", content).await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg.resolve("whitehut").unwrap();
    assert_eq!(p.description(), Some("short"));
    assert!(p.body().await.unwrap().contains("# body"));
}

#[tokio::test]
async fn frontmatter_absent_ok() {
    let tmp = TempDir::new().unwrap();
    write_prompt(tmp.path(), "whitehut", "# body only\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    let p = reg.resolve("whitehut").unwrap();
    assert_eq!(p.description(), None);
}

#[tokio::test]
async fn malformed_frontmatter_skips_file() {
    let tmp = TempDir::new().unwrap();
    // Missing closing --- — SystemPrompt::from_path rejects, discover logs+skips.
    write_prompt(tmp.path(), "bad", "---\ndescription: x\n(no terminator)\n").await;
    write_prompt(tmp.path(), "good", "# ok\n").await;

    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    assert!(reg.resolve("bad").is_none());
    assert!(reg.resolve("good").is_some());
}

#[tokio::test]
async fn reject_uppercase_filename() {
    let tmp = TempDir::new().unwrap();
    // Uppercase stem is rejected at load time.
    fs::write(tmp.path().join("Whitehut.md"), "# body\n")
        .await
        .unwrap();
    let reg = PromptRegistry::discover(vec![tmp.path().to_path_buf()])
        .await
        .unwrap();
    assert!(reg.is_empty());
}

#[tokio::test]
async fn from_path_direct() {
    let tmp = TempDir::new().unwrap();
    let path = write_prompt(tmp.path(), "whitehut", "# body\n").await;
    let p = SystemPrompt::from_path(path).await.unwrap();
    assert_eq!(p.name(), "whitehut");
    assert_eq!(p.body().await.unwrap().trim(), "# body");
}
