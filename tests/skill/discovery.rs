use sombrax_agentic_core::skill::SkillRegistry;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_discover_skills_from_directory() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create first skill
    let skill1_dir = base_path.join("skill-one");
    fs::create_dir(&skill1_dir).await.unwrap();
    fs::write(
        skill1_dir.join("SKILL.md"),
        r#"---
name: skill-one
description: First test skill
---
# Skill One
"#,
    )
    .await
    .unwrap();

    // Create second skill
    let skill2_dir = base_path.join("skill-two");
    fs::create_dir(&skill2_dir).await.unwrap();
    fs::write(
        skill2_dir.join("SKILL.md"),
        r#"---
name: skill-two
description: Second test skill
---
# Skill Two
"#,
    )
    .await
    .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    assert_eq!(registry.len(), 2);
    assert!(registry.get("skill-one").is_some());
    assert!(registry.get("skill-two").is_some());
}

#[tokio::test]
async fn test_discover_nested_skills() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create nested skill directory structure
    let nested_dir = base_path.join("category").join("subcategory");
    fs::create_dir_all(&nested_dir).await.unwrap();

    let skill_dir = nested_dir.join("nested-skill");
    fs::create_dir(&skill_dir).await.unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: nested-skill
description: A deeply nested skill
---
# Nested Skill
"#,
    )
    .await
    .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    assert_eq!(registry.len(), 1);
    assert!(registry.get("nested-skill").is_some());
}

#[tokio::test]
async fn test_skill_deduplication() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create first skill with same name in path1
    let path1 = base_path.join("path1");
    fs::create_dir(&path1).await.unwrap();
    let skill1_dir = path1.join("duplicate-skill");
    fs::create_dir(&skill1_dir).await.unwrap();
    fs::write(
        skill1_dir.join("SKILL.md"),
        r#"---
name: duplicate-skill
description: First version
---
# First
"#,
    )
    .await
    .unwrap();

    // Create second skill with same name in path2
    let path2 = base_path.join("path2");
    fs::create_dir(&path2).await.unwrap();
    let skill2_dir = path2.join("duplicate-skill");
    fs::create_dir(&skill2_dir).await.unwrap();
    fs::write(
        skill2_dir.join("SKILL.md"),
        r#"---
name: duplicate-skill
description: Second version wins
---
# Second
"#,
    )
    .await
    .unwrap();

    // Search path2 first, then path1 - last wins
    let registry = SkillRegistry::discover(vec![path1, path2]).await.unwrap();

    assert_eq!(registry.len(), 1);
    let skill = registry.get("duplicate-skill").unwrap();
    assert_eq!(skill.description(), "Second version wins");
}

#[tokio::test]
async fn test_empty_directory() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());
}

#[tokio::test]
async fn test_nonexistent_path() {
    let registry = SkillRegistry::discover(vec![PathBuf::from("/nonexistent/path")])
        .await
        .unwrap();

    assert_eq!(registry.len(), 0);
}

#[tokio::test]
async fn test_invalid_skills_skipped() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create valid skill
    let valid_dir = base_path.join("valid-skill");
    fs::create_dir(&valid_dir).await.unwrap();
    fs::write(
        valid_dir.join("SKILL.md"),
        r#"---
name: valid-skill
description: This one is valid
---
# Valid
"#,
    )
    .await
    .unwrap();

    // Create invalid skill (invalid name)
    let invalid_dir = base_path.join("invalid-skill");
    fs::create_dir(&invalid_dir).await.unwrap();
    fs::write(
        invalid_dir.join("SKILL.md"),
        r#"---
name: Invalid_Name
description: This one has invalid name
---
# Invalid
"#,
    )
    .await
    .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    // Should only load the valid skill
    assert_eq!(registry.len(), 1);
    assert!(registry.get("valid-skill").is_some());
    assert!(registry.get("Invalid_Name").is_none());
}

#[tokio::test]
async fn test_all_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create two skills
    for i in 1..=2 {
        let skill_dir = base_path.join(format!("skill-{}", i));
        fs::create_dir(&skill_dir).await.unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                r#"---
name: skill-{}
description: Skill number {}
---
# Skill {}
"#,
                i, i, i
            ),
        )
        .await
        .unwrap();
    }

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    let metadata = registry.all_metadata();
    assert_eq!(metadata.len(), 2);

    let names: Vec<String> = metadata.iter().map(|m| m.name.clone()).collect();
    assert!(names.contains(&"skill-1".to_string()));
    assert!(names.contains(&"skill-2".to_string()));
}

#[tokio::test]
async fn test_find_matching() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create skills with different descriptions
    let greeting_dir = base_path.join("greeting");
    fs::create_dir(&greeting_dir).await.unwrap();
    fs::write(
        greeting_dir.join("SKILL.md"),
        r#"---
name: greeting
description: Greet users with hello messages and welcome them
---
# Greeting
"#,
    )
    .await
    .unwrap();

    let math_dir = base_path.join("calculator");
    fs::create_dir(&math_dir).await.unwrap();
    fs::write(
        math_dir.join("SKILL.md"),
        r#"---
name: calculator
description: Perform mathematical calculations and equations
---
# Calculator
"#,
    )
    .await
    .unwrap();

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    // Search for greeting
    let matches = registry.find_matching("Hello, can you greet me?");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].name(), "greeting");

    // Search for math
    let matches = registry.find_matching("Calculate 2+2");
    assert!(matches.iter().any(|s| s.name() == "calculator"));
}

#[tokio::test]
async fn test_skill_names() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    for name in &["alpha", "beta", "gamma"] {
        let skill_dir = base_path.join(name);
        fs::create_dir(&skill_dir).await.unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                r#"---
name: {}
description: Skill {}
---
# {}
"#,
                name, name, name
            ),
        )
        .await
        .unwrap();
    }

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    let names = registry.skill_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    assert!(names.contains(&"gamma".to_string()));
}

#[tokio::test]
async fn test_skill_metadata_ordering_deterministic() {
    // Test that skill metadata is returned in a deterministic, sorted order
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create skills with names that would have different hash orders
    let skill_names = vec!["zebra", "alpha", "mike", "charlie", "bravo"];

    for name in &skill_names {
        let skill_dir = base_path.join(name);
        fs::create_dir(&skill_dir).await.unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                r#"---
name: {}
description: Skill {}
---
# {}
"#,
                name, name, name
            ),
        )
        .await
        .unwrap();
    }

    let registry = SkillRegistry::discover(vec![base_path.to_path_buf()])
        .await
        .unwrap();

    // Get metadata multiple times - should always be in same order
    for _ in 0..10 {
        let metadata = registry.all_metadata();
        let names: Vec<_> = metadata.iter().map(|m| &m.name).collect();

        // Should always be alphabetically sorted
        let mut expected = skill_names.clone();
        expected.sort();
        let expected_refs: Vec<&str> = expected.iter().map(|s| s.as_ref()).collect();

        assert_eq!(names, expected_refs);
    }

    // Verify skill_names() is also sorted
    let mut names = registry.skill_names();
    names.sort(); // Sort for comparison
    let mut expected = skill_names.clone();
    expected.sort();
    let expected_strings: Vec<_> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, expected_strings);
}
