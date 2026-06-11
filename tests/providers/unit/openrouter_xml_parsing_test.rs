//! Unit tests for OpenRouter XML tool call parsing

use sombrax_agentic_core::providers::openrouter::{
    extract_first_json_object, parse_minimax_xml_tool_calls,
};

#[test]
fn test_parse_minimax_xml_single_tool_call() {
    // Note: The regex doesn't match across newlines, so we use single-line format
    let content = r#"<minimax:tool_call><invoke name="get_weather"><parameter name="location">New York</parameter><parameter name="units">celsius</parameter></invoke></minimax:tool_call>"#;

    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
    assert_eq!(args["location"], "New York");
    assert_eq!(args["units"], "celsius");
}

#[test]
fn test_parse_minimax_xml_multiple_tool_calls() {
    let content = r#"<minimax:tool_call><invoke name="get_weather"><parameter name="location">Paris</parameter></invoke></minimax:tool_call> <minimax:tool_call><invoke name="get_time"><parameter name="timezone">CET</parameter></invoke></minimax:tool_call>"#;

    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[1].name, "get_time");
}

#[test]
fn test_parse_minimax_xml_no_tool_calls() {
    let content = "This is just regular text without any tool calls.";
    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert!(tool_calls.is_empty());
}

#[test]
fn test_parse_minimax_xml_empty_content() {
    let content = "";
    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert!(tool_calls.is_empty());
}

#[test]
fn test_parse_minimax_xml_no_parameters() {
    let content =
        r#"<minimax:tool_call><invoke name="get_current_time"></invoke></minimax:tool_call>"#;

    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_current_time");

    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
    assert!(args.as_object().unwrap().is_empty());
}

#[test]
fn test_parse_minimax_xml_with_surrounding_text() {
    let content = r#"Let me check the weather for you. <minimax:tool_call><invoke name="get_weather"><parameter name="location">London</parameter></invoke></minimax:tool_call> I'll get that information right away."#;

    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
}

#[test]
fn test_extract_first_json_simple() {
    let input = r#"{"name": "test", "value": 42}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(r#"{"name": "test", "value": 42}"#.to_string()));
}

#[test]
fn test_extract_first_json_with_duplicates() {
    let input = r#"{"name": "first"}{"name": "second"}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(r#"{"name": "first"}"#.to_string()));
}

#[test]
fn test_extract_first_json_nested() {
    let input = r#"{"outer": {"inner": "value"}, "key": "data"}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(input.to_string()));
}

#[test]
fn test_extract_first_json_with_prefix() {
    let input = r#"Some text before {"name": "test"} and after"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(r#"{"name": "test"}"#.to_string()));
}

#[test]
fn test_extract_first_json_no_object() {
    let input = "This is just plain text without JSON";
    let result = extract_first_json_object(input);
    assert!(result.is_none());
}

#[test]
fn test_extract_first_json_empty() {
    let input = "";
    let result = extract_first_json_object(input);
    assert!(result.is_none());
}

#[test]
fn test_extract_first_json_incomplete() {
    let input = r#"{"name": "test"#; // Missing closing brace
    let result = extract_first_json_object(input);
    assert!(result.is_none());
}

#[test]
fn test_extract_first_json_deeply_nested() {
    let input = r#"{"a": {"b": {"c": {"d": "deep"}}}}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(input.to_string()));
}

#[test]
fn test_extract_first_json_braces_in_string_values() {
    // Solidity code with braces inside old_string — the exact bug that caused
    // "EOF while parsing a string" when the old implementation counted braces
    // inside JSON string values
    let input = r#"{"file_path": "test.sol", "old_string": "function test() public {\n    assert(true);\n}\n}", "new_string": "function test() public {\n    assert(false);\n}\n}"}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(input.to_string()));

    // Verify the result is valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["file_path"], "test.sol");
}

#[test]
fn test_extract_first_json_escaped_quotes_in_strings() {
    // Escaped quotes followed by braces inside strings
    let input = r#"{"old_string": "assertTrue(x != address(0), \"not deployed\");\n    }\n}", "new_string": ""}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(input.to_string()));

    let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert!(parsed["old_string"].is_string());
}

#[test]
fn test_extract_first_json_unbalanced_braces_in_strings() {
    // String value containing more closing braces than opening — would break
    // naive brace counting by going to depth < 0
    let input = r#"{"code": "if (x) { } } }"}"#;
    let result = extract_first_json_object(input);
    assert_eq!(result, Some(input.to_string()));
}

#[test]
fn test_tool_call_id_generation() {
    let content = r#"<minimax:tool_call><invoke name="test_tool"><parameter name="arg">value</parameter></invoke></minimax:tool_call>"#;

    let tool_calls = parse_minimax_xml_tool_calls(content);
    assert_eq!(tool_calls.len(), 1);
    // ID should use incremental format: call_minimax_N
    assert!(
        tool_calls[0].id.starts_with("call_minimax_"),
        "Expected ID to start with 'call_minimax_', got: {}",
        tool_calls[0].id
    );
}
