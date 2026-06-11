//! Unit tests for Message serialization/deserialization (T018)

use sombrax_agentic_core::message::{
    AssistantContent, Message, ToolCall, ToolCallFunction, UserContent,
};

#[test]
fn test_user_message_creation() {
    let msg = Message::user("Hello, world!");

    if let Message::User { content, id } = msg {
        assert_eq!(content.len(), 1);
        assert!(id.is_none());

        if let UserContent::Text { text } = &content[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected Text content");
        }
    } else {
        panic!("Expected User message");
    }
}

#[test]
fn test_assistant_message_creation() {
    let msg = Message::assistant("I'm here to help!");

    if let Message::Assistant { content, id, .. } = msg {
        assert_eq!(content.len(), 1);
        assert!(id.is_none());

        if let AssistantContent::Text { text } = &content[0] {
            assert_eq!(text, "I'm here to help!");
        } else {
            panic!("Expected Text content");
        }
    } else {
        panic!("Expected Assistant message");
    }
}

#[test]
fn test_message_serialization_user() {
    let msg = Message::user("Test message");

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();

    if let Message::User { content, .. } = parsed {
        if let UserContent::Text { text } = &content[0] {
            assert_eq!(text, "Test message");
        }
    }
}

#[test]
fn test_message_serialization_assistant() {
    let msg = Message::assistant("Response");

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();

    if let Message::Assistant { content, .. } = parsed {
        if let AssistantContent::Text { text } = &content[0] {
            assert_eq!(text, "Response");
        }
    }
}

#[test]
fn test_user_content_text() {
    let content = UserContent::Text {
        text: "Hello".to_string(),
    };

    let json = serde_json::to_string(&content).unwrap();
    assert!(json.contains("\"text\":"));

    let parsed: UserContent = serde_json::from_str(&json).unwrap();
    if let UserContent::Text { text } = parsed {
        assert_eq!(text, "Hello");
    }
}

#[test]
fn test_user_content_tool_result() {
    let content = UserContent::ToolResult {
        id: "call_123".to_string(),
        content: "42".to_string(),
    };

    let json = serde_json::to_string(&content).unwrap();
    let parsed: UserContent = serde_json::from_str(&json).unwrap();

    if let UserContent::ToolResult { id, content } = parsed {
        assert_eq!(id, "call_123");
        assert_eq!(content, "42");
    }
}

#[test]
fn test_user_content_image() {
    let content = UserContent::Image {
        data: "base64encodeddata".to_string(),
        media_type: "image/png".to_string(),
    };

    let json = serde_json::to_string(&content).unwrap();
    let parsed: UserContent = serde_json::from_str(&json).unwrap();

    if let UserContent::Image { data, media_type } = parsed {
        assert_eq!(data, "base64encodeddata");
        assert_eq!(media_type, "image/png");
    }
}

#[test]
fn test_assistant_content_text() {
    let content = AssistantContent::Text {
        text: "Response".to_string(),
    };

    let json = serde_json::to_string(&content).unwrap();
    let parsed: AssistantContent = serde_json::from_str(&json).unwrap();

    if let AssistantContent::Text { text } = parsed {
        assert_eq!(text, "Response");
    }
}

#[test]
fn test_assistant_content_tool_call() {
    let content = AssistantContent::ToolCall(ToolCall {
        id: "call_456".to_string(),
        function: ToolCallFunction {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"London"}"#.to_string(),
        },
    });

    let json = serde_json::to_string(&content).unwrap();
    let parsed: AssistantContent = serde_json::from_str(&json).unwrap();

    if let AssistantContent::ToolCall(tc) = parsed {
        assert_eq!(tc.id, "call_456");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, r#"{"city":"London"}"#);
    }
}

#[test]
fn test_assistant_content_reasoning() {
    let content = AssistantContent::Reasoning {
        reasoning: vec!["Step 1".to_string(), "Step 2".to_string()],
    };

    let json = serde_json::to_string(&content).unwrap();
    let parsed: AssistantContent = serde_json::from_str(&json).unwrap();

    if let AssistantContent::Reasoning { reasoning } = parsed {
        assert_eq!(reasoning.len(), 2);
        assert_eq!(reasoning[0], "Step 1");
        assert_eq!(reasoning[1], "Step 2");
    }
}

#[test]
fn test_message_with_id() {
    let mut msg = Message::user("Test");
    if let Message::User { ref mut id, .. } = msg {
        *id = Some("msg-001".to_string());
    }

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();

    if let Message::User { id, .. } = parsed {
        assert_eq!(id, Some("msg-001".to_string()));
    }
}

#[test]
fn test_complex_message_roundtrip() {
    let msg = Message::User {
        content: vec![
            UserContent::Text {
                text: "Hello".to_string(),
            },
            UserContent::ToolResult {
                id: "call_1".to_string(),
                content: "Result".to_string(),
            },
        ],
        id: Some("complex-msg".to_string()),
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();

    if let Message::User { content, id } = parsed {
        assert_eq!(content.len(), 2);
        assert_eq!(id, Some("complex-msg".to_string()));
    }
}

#[test]
fn test_message_role() {
    let user_msg = Message::user("test");
    assert_eq!(user_msg.role(), "user");

    let assistant_msg = Message::assistant("test");
    assert_eq!(assistant_msg.role(), "assistant");
}
