use sombrax_agentic_core::skill::Skill;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_parse_valid_skill() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("test-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: test-skill
description: A test skill for validation
---

# Test Skill

This is a test skill.
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let skill = Skill::from_path(skill_dir).await.unwrap();

    assert_eq!(skill.name(), "test-skill");
    assert_eq!(skill.description(), "A test skill for validation");
}

#[tokio::test]
async fn test_parse_skill_with_resources() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("resource-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: resource-skill
description: A skill with resources
---

# Resource Skill

Check out data.txt for more info.
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    fs::write(skill_dir.join("data.txt"), "Some data")
        .await
        .unwrap();

    let skill = Skill::from_path(skill_dir).await.unwrap();
    let resources = skill.list_resources().await.unwrap();

    assert!(resources.contains(&"data.txt".to_string()));
}

#[tokio::test]
async fn test_invalid_skill_name() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("invalid-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    // Invalid name: contains uppercase
    let skill_content = r#"---
name: Invalid-Name
description: This should fail
---

# Invalid Skill
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let result = Skill::from_path(skill_dir).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_missing_frontmatter() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("no-frontmatter");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"# No Frontmatter Skill

This skill is missing frontmatter.
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let result = Skill::from_path(skill_dir).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_reserved_word_in_name() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("reserved-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: claude-helper
description: Uses reserved word
---

# Reserved Skill
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let result = Skill::from_path(skill_dir).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_xml_tags_in_description() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("xml-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: xml-skill
description: Contains <script>alert('xss')</script> tags
---

# XML Skill
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let result = Skill::from_path(skill_dir).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_load_instructions() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("instruction-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: instruction-skill
description: Test instructions loading
---

# Instruction Skill

## Step 1
Do this first.

## Step 2
Then do this.
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let skill = Skill::from_path(skill_dir).await.unwrap();
    let instructions = skill.load_instructions().await.unwrap();

    assert!(instructions.contains("Step 1"));
    assert!(instructions.contains("Step 2"));
    assert!(!instructions.contains("name: instruction-skill")); // Frontmatter removed
}

#[tokio::test]
async fn test_skill_matches_request() {
    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("greeting-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: greeting-skill
description: Greet the user with hello or welcome messages
---

# Greeting Skill
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let skill = Skill::from_path(skill_dir).await.unwrap();

    // Should match requests containing keywords
    assert!(skill.matches("Hello there!"));
    assert!(skill.matches("I need a welcome message"));
    assert!(skill.matches("Can you greet me?"));

    // Should not match unrelated requests
    assert!(!skill.matches("Calculate 2+2"));
}

#[tokio::test]
async fn test_concurrent_load_instructions() {
    use std::sync::Arc;

    let temp_dir = TempDir::new().unwrap();
    let skill_dir = temp_dir.path().join("concurrent-skill");
    fs::create_dir(&skill_dir).await.unwrap();

    let skill_content = r#"---
name: concurrent-skill
description: Test concurrent loading
---

# Concurrent Skill

This skill tests concurrent access to load_instructions().
Multiple concurrent calls should all succeed and return the same result.
"#;

    fs::write(skill_dir.join("SKILL.md"), skill_content)
        .await
        .unwrap();

    let skill = Arc::new(Skill::from_path(skill_dir).await.unwrap());

    // Spawn 10 concurrent tasks all trying to load instructions
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let skill_clone = Arc::clone(&skill);
            tokio::spawn(async move {
                // Convert to String immediately to avoid lifetime issues
                skill_clone.load_instructions().await.map(|s| s.to_string())
            })
        })
        .collect();

    // All should succeed with same result
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        results.push(result);
    }

    // All results should be identical
    assert_eq!(results.len(), 10);
    for i in 1..results.len() {
        assert_eq!(results[0], results[i]);
    }

    // Verify content is correct
    assert!(results[0].contains("Concurrent Skill"));
    assert!(results[0].contains("concurrent access"));
}
