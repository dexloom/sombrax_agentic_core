//! Todo tools for tracking task progress

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;
use crate::tools::registry::{Tool, ToolDefinition};

/// A todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// Sequential ID (TASK-01, TASK-02, etc.)
    pub id: String,
    /// Task content (imperative form)
    pub content: String,
    /// Current status (freeform string, e.g., "pending", "in_progress", "completed")
    pub status: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Completion timestamp (if completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

// ============================================================================
// TodoReadTool
// ============================================================================

/// Read current todos
#[derive(Clone)]
pub struct TodoReadTool {
    context: ToolContext,
}

impl TodoReadTool {
    /// Create a new todo read tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Arguments for todo read (none required)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TodoReadArgs {}

/// Output of todo read
#[derive(Debug, Serialize)]
pub struct TodoReadOutput {
    /// All todos
    pub todos: Vec<TodoItem>,
    /// Total count
    pub total_count: usize,
    /// Count of todos by status (HashMap<Status, Count>)
    pub status_counts: std::collections::HashMap<String, usize>,
}

impl Tool for TodoReadTool {
    const NAME: &'static str = "todo_read";
    type Args = TodoReadArgs;
    type Output = TodoReadOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(TodoReadArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Read the current todo list to see all tasks and their status.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. Do I need to check the current task list?
2. Do I need to know what tasks are pending or in progress?

## PARAMETERS

No parameters required - this tool reads the entire todo list.

## OUTPUT

Returns:
- todos: List of all todo items with sequential id (TASK-01, TASK-02, etc.), content, status, timestamps
- total_count: Total number of todos
- status_counts: HashMap<Status, Count> - aggregates all todos by their status value
  Example: {"pending": 2, "in_progress": 1, "blocked": 3, "completed": 5}

## WHEN TO USE THIS TOOL

- Before starting work to review what needs to be done
- To check progress on the current task list
- To verify which tasks are completed

## COMMON MISTAKES TO AVOID

1. Do NOT call this just to update - use todo_write for updates
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "todo_read"))]
    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let todos = self.context.get_todos();

        let total_count = todos.len();

        // Aggregate by status
        let mut status_counts = std::collections::HashMap::new();
        for todo in &todos {
            *status_counts.entry(todo.status.clone()).or_insert(0) += 1;
        }

        Ok(TodoReadOutput {
            todos,
            total_count,
            status_counts,
        })
    }
}

// ============================================================================
// TodoWriteTool
// ============================================================================

/// Write/update todos
#[derive(Clone)]
pub struct TodoWriteTool {
    context: ToolContext,
}

impl TodoWriteTool {
    /// Create a new todo write tool
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

/// Item to write
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TodoWriteItem {
    /// Task content (imperative form)
    pub content: String,
    /// Status: "pending", "in_progress", or "completed"
    pub status: String,
}

/// Arguments for todo write
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TodoWriteArgs {
    /// List of todos to create/update
    pub todos: Vec<TodoWriteItem>,
}

/// Output of todo write
#[derive(Debug, Serialize)]
pub struct TodoWriteOutput {
    /// Number of todos written
    pub count: usize,
    /// IDs of todos
    pub todo_ids: Vec<String>,
    /// Count of todos by status (HashMap<Status, Count>)
    pub status_counts: std::collections::HashMap<String, usize>,
}

impl Tool for TodoWriteTool {
    const NAME: &'static str = "todo_write";
    type Args = TodoWriteArgs;
    type Output = TodoWriteOutput;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let schema = schemars::schema_for!(TodoWriteArgs);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: r#"Write or update the todo list to track task progress.

## BEFORE CALLING THIS TOOL

Think step-by-step:
1. What tasks do I need to track?
2. What is the status of each task?
3. Am I replacing the entire list or updating specific items?

NOTE: This tool REPLACES the entire todo list. Include ALL todos you want to keep.

## PARAMETERS

- `todos` (REQUIRED, ARRAY): Array of todo items, each with:
  - `content` (REQUIRED, STRING): Task description in imperative form
    CORRECT: "Fix authentication bug"
    CORRECT: "Add unit tests for login"
    WRONG: {"content": "..."} <-- Each item must be a proper object in the array

  - `status` (REQUIRED, STRING): Any string describing the task status
    EXAMPLES: "pending", "in_progress", "completed", "blocked", "waiting", etc.
    NOTE: Status is freeform and not validated

## EXAMPLES

Create initial todo list:
  todos: [
    {"content": "Read existing code", "status": "completed"},
    {"content": "Implement feature", "status": "in_progress"},
    {"content": "Write tests", "status": "pending"}
  ]

Mark a task complete (include ALL todos):
  todos: [
    {"content": "Read existing code", "status": "completed"},
    {"content": "Implement feature", "status": "completed"},
    {"content": "Write tests", "status": "in_progress"}
  ]

## BEST PRACTICES

- Keep exactly ONE task as "in_progress" at a time
- Mark tasks "completed" immediately after finishing
- Use clear, actionable task descriptions

## COMMON MISTAKES TO AVOID

1. Do NOT forget to include all existing todos when updating
2. Do NOT have multiple tasks as "in_progress" simultaneously
3. Do NOT pass the todos array as a string - use proper JSON array
"#
            .to_string(),
            parameters: serde_json::to_value(schema).unwrap_or_default(),
        }
    }

    #[instrument(skip(self), fields(tool = "todo_write", count = args.todos.len()))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let now = Utc::now();
        let mut todos = Vec::new();
        let mut todo_ids = Vec::new();

        for (index, item) in args.todos.into_iter().enumerate() {
            // Generate sequential ID: TASK-01, TASK-02, etc.
            let id = format!("TASK-{:02}", index + 1);
            todo_ids.push(id.clone());

            // Set completed_at if status is "completed"
            let completed_at = if item.status == "completed" {
                Some(now)
            } else {
                None
            };

            todos.push(TodoItem {
                id,
                content: item.content.clone(),
                status: item.status.clone(),
                created_at: now,
                updated_at: now,
                completed_at,
            });
        }

        let count = todos.len();

        // Aggregate by status
        let mut status_counts = std::collections::HashMap::new();
        for todo in &todos {
            *status_counts.entry(todo.status.clone()).or_insert(0) += 1;
        }

        // Store in context
        self.context.set_todos(todos);

        Ok(TodoWriteOutput {
            count,
            todo_ids,
            status_counts,
        })
    }
}
