use sombrax_agentic_core::skill::SkillRegistry;
use sombrax_agentic_core::tools::registry::Tool;
use sombrax_agentic_core::tools::skill::{SkillArgs, SkillTool};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_skill_tool_load() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create a test skill
    let skill_dir = base_path.join("test-skill");
    fs::create_dir(&skill_dir).await.unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: test-skill
description: A test skill
---

# Test Skill

## Instructions

Follow these steps:
1. Step one
2. Step two
"#,
    )
    .await
    .unwrap();

    // Create a resource file
    fs::write(skill_dir.join("data.txt"), "Sample data")
        .await
        .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    let tool = SkillTool::new(Arc::new(registry));

    let result = tool
        .call(SkillArgs {
            skill_name: "test-skill".to_string(),
        })
        .await
        .unwrap();

    assert_eq!(result.skill_name, "test-skill");
    assert!(result.instructions.contains("Follow these steps"));
    assert!(result.instructions.contains("Step one"));
    assert!(result.resources.contains(&"data.txt".to_string()));
}

#[tokio::test]
async fn test_skill_tool_not_found() {
    let registry = SkillRegistry::new();
    let tool = SkillTool::new(Arc::new(registry));

    let result = tool
        .call(SkillArgs {
            skill_name: "nonexistent".to_string(),
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_skill_tool_definition() {
    let registry = SkillRegistry::new();
    let tool = SkillTool::new(Arc::new(registry));

    let definition = tool.definition("".to_string()).await;

    assert_eq!(definition.name, "load_skill");
    assert!(!definition.description.is_empty());
    assert!(definition.parameters.is_object());
}

#[tokio::test]
async fn test_skill_tool_with_multiple_resources() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    let skill_dir = base_path.join("resource-skill");
    fs::create_dir(&skill_dir).await.unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: resource-skill
description: Skill with multiple resources
---

# Resource Skill
"#,
    )
    .await
    .unwrap();

    // Create multiple resource files
    fs::write(skill_dir.join("readme.md"), "Readme content")
        .await
        .unwrap();
    fs::write(skill_dir.join("config.json"), "{}")
        .await
        .unwrap();

    // Create a subdirectory with a file
    let scripts_dir = skill_dir.join("scripts");
    fs::create_dir(&scripts_dir).await.unwrap();
    fs::write(scripts_dir.join("helper.sh"), "#!/bin/bash")
        .await
        .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    let tool = SkillTool::new(Arc::new(registry));

    let result = tool
        .call(SkillArgs {
            skill_name: "resource-skill".to_string(),
        })
        .await
        .unwrap();

    assert!(result.resources.contains(&"readme.md".to_string()));
    assert!(result.resources.contains(&"config.json".to_string()));
    assert!(result.resources.contains(&"scripts/helper.sh".to_string()));
    // SKILL.md should not be in resources
    assert!(!result.resources.contains(&"SKILL.md".to_string()));
}

#[tokio::test]
async fn test_skill_tool_name() {
    let registry = SkillRegistry::new();
    let tool = SkillTool::new(Arc::new(registry));

    assert_eq!(tool.name(), "load_skill");
}
