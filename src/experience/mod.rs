//! Experience-based learning system for agent self-improvement.
//!
//! This module provides a generic framework for agents to learn from past
//! task executions without model fine-tuning. It implements the core concepts
//! from Experiential Reflective Learning (ERL):
//!
//! 1. **Store**: Persist lessons (heuristics) extracted from task outcomes
//! 2. **Retrieve**: Find relevant lessons for a new task based on context
//! 3. **Reflect**: Analyze execution traces to extract new lessons
//!
//! The module provides traits and a filesystem-backed implementation.
//! Domain-specific reflection logic is implemented by consumers (e.g.,
//! security audit agents implement their own `Reflector`).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
//! │  Reflector   │────>│ KnowledgeBase│<────│  Retriever   │
//! │ (post-task)  │     │  (storage)   │     │ (pre-task)   │
//! └─────────────┘     └──────────────┘     └─────────────┘
//! ```
//!
//! ## Filesystem Layout
//!
//! ```text
//! {base_path}/knowledge/
//! ├── {domain}/              # e.g., "fork_test", "audit", "onchain"
//! │   ├── patterns/          # General reusable lessons
//! │   │   ├── lesson_001.md
//! │   │   └── lesson_002.md
//! │   ├── contracts/         # Per-contract specific lessons
//! │   │   └── 0x1234...md
//! │   └── successes/         # What worked (positive examples)
//! │       └── confirmed_reentrancy.md
//! ```

mod store;
mod types;

pub use store::FileKnowledgeBase;
pub use types::{Lesson, LessonCategory, LessonQuery, LessonTriggers, Reflector};
