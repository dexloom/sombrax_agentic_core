use crate::skill::{SkillError, SkillRegistry};
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Tool for loading skill instructions
#[derive(Clone)]
pub struct SkillTool {
    registry: Arc<SkillRegistry>,
}

impl SkillTool {
    /// Create a new SkillTool with the given registry
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

/// Arguments for loading a skill
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SkillArgs {
    /// Name of the skill to load
    pub skill_name: String,
}

/// Output from loading a skill
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SkillOutput {
    /// Loaded skill instructions (Level 2)
    pub instructions: String,

    /// Available resource files (Level 3+)
    pub resources: Vec<String>,

    /// Skill name
    pub skill_name: String,
}

impl Tool for SkillTool {
    const NAME: &'static str = "load_skill";

    type Args = SkillArgs;
    type Output = SkillOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Load a skill's instructions and discover available resources. \
                         Use this when a user's request matches a skill description from the system prompt."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load"
                    }
                },
                "required": ["skill_name"]
            }),
        }
    }

    async fn call(&self, args: SkillArgs) -> Result<SkillOutput, ToolError> {
        // Get the skill from registry
        let skill = self.registry.get(&args.skill_name).ok_or_else(|| {
            ToolError::Validation(format!("Skill not found: {}", args.skill_name))
        })?;

        // Load Level 2 instructions
        let instructions = skill
            .load_instructions()
            .await
            .map_err(|e| ToolError::Validation(format!("Failed to load instructions: {}", e)))?
            .to_string();

        // Discover resources (Level 3+)
        let resources = skill
            .list_resources()
            .await
            .map_err(|e| ToolError::Validation(format!("Failed to list resources: {}", e)))?;

        Ok(SkillOutput {
            instructions,
            resources,
            skill_name: args.skill_name,
        })
    }
}

impl From<SkillError> for ToolError {
    fn from(e: SkillError) -> Self {
        ToolError::Validation(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_tool_name() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = SkillTool::new(registry);
        assert_eq!(tool.name(), "load_skill");
    }
}
