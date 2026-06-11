//! Flexible numeric deserialization helpers for tool arguments.
//!
//! LLMs sometimes send numeric arguments as strings (e.g., `"110"` instead of `110`).
//! These helpers accept both JSON numbers and JSON string representations of numbers.

use serde::de::{self, Deserializer};
use serde::Deserialize;
use serde_json::Value;

/// Deserializes an `Option<usize>` that accepts JSON numbers, strings of numbers, or null.
pub fn deserialize_flexible_optional_usize<'de, D>(
    deserializer: D,
) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            if let Some(v) = n.as_u64() {
                usize::try_from(v)
                    .map(Some)
                    .map_err(|_| de::Error::custom(format!("number {} out of range for usize", n)))
            } else {
                Err(de::Error::custom(format!(
                    "expected non-negative integer, got {}",
                    n
                )))
            }
        }
        Some(Value::String(s)) => {
            let s = s.trim();
            if s.is_empty() {
                return Ok(None);
            }
            s.parse::<usize>().map(Some).map_err(|_| {
                de::Error::custom(format!(
                    "expected a non-negative integer, got string {:?}",
                    s
                ))
            })
        }
        Some(other) => Err(de::Error::custom(format!(
            "expected number, string, or null, got {}",
            other
        ))),
    }
}

/// Deserializes an `Option<u64>` that accepts JSON numbers, strings of numbers, or null.
pub fn deserialize_flexible_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            if let Some(v) = n.as_u64() {
                Ok(Some(v))
            } else {
                Err(de::Error::custom(format!(
                    "expected non-negative integer, got {}",
                    n
                )))
            }
        }
        Some(Value::String(s)) => {
            let s = s.trim();
            if s.is_empty() {
                return Ok(None);
            }
            s.parse::<u64>().map(Some).map_err(|_| {
                de::Error::custom(format!(
                    "expected a non-negative integer, got string {:?}",
                    s
                ))
            })
        }
        Some(other) => Err(de::Error::custom(format!(
            "expected number, string, or null, got {}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestUsize {
        #[serde(default, deserialize_with = "deserialize_flexible_optional_usize")]
        value: Option<usize>,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestU64 {
        #[serde(default, deserialize_with = "deserialize_flexible_optional_u64")]
        value: Option<u64>,
    }

    // --- Option<usize> tests ---

    #[test]
    fn usize_from_number() {
        let r: TestUsize = serde_json::from_str(r#"{"value": 42}"#).unwrap();
        assert_eq!(r.value, Some(42));
    }

    #[test]
    fn usize_from_string() {
        let r: TestUsize = serde_json::from_str(r#"{"value": "110"}"#).unwrap();
        assert_eq!(r.value, Some(110));
    }

    #[test]
    fn usize_from_null() {
        let r: TestUsize = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert_eq!(r.value, None);
    }

    #[test]
    fn usize_from_missing() {
        let r: TestUsize = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(r.value, None);
    }

    #[test]
    fn usize_from_zero() {
        let r: TestUsize = serde_json::from_str(r#"{"value": 0}"#).unwrap();
        assert_eq!(r.value, Some(0));
    }

    #[test]
    fn usize_from_string_zero() {
        let r: TestUsize = serde_json::from_str(r#"{"value": "0"}"#).unwrap();
        assert_eq!(r.value, Some(0));
    }

    #[test]
    fn usize_rejects_non_numeric_string() {
        let r = serde_json::from_str::<TestUsize>(r#"{"value": "abc"}"#);
        assert!(r.is_err());
        let err = r.unwrap_err().to_string();
        assert!(err.contains("non-negative integer"), "error was: {}", err);
    }

    #[test]
    fn usize_rejects_negative_number() {
        let r = serde_json::from_str::<TestUsize>(r#"{"value": -5}"#);
        assert!(r.is_err());
    }

    #[test]
    fn usize_rejects_float_string() {
        let r = serde_json::from_str::<TestUsize>(r#"{"value": "3.14"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn usize_rejects_float_number() {
        let r = serde_json::from_str::<TestUsize>(r#"{"value": 3.14}"#);
        assert!(r.is_err());
    }

    #[test]
    fn usize_empty_string_is_none() {
        let r: TestUsize = serde_json::from_str(r#"{"value": ""}"#).unwrap();
        assert_eq!(r.value, None);
    }

    #[test]
    fn usize_whitespace_trimmed() {
        let r: TestUsize = serde_json::from_str(r#"{"value": " 99 "}"#).unwrap();
        assert_eq!(r.value, Some(99));
    }

    // --- Option<u64> tests ---

    #[test]
    fn u64_from_number() {
        let r: TestU64 = serde_json::from_str(r#"{"value": 120000}"#).unwrap();
        assert_eq!(r.value, Some(120000));
    }

    #[test]
    fn u64_from_string() {
        let r: TestU64 = serde_json::from_str(r#"{"value": "300000"}"#).unwrap();
        assert_eq!(r.value, Some(300000));
    }

    #[test]
    fn u64_from_null() {
        let r: TestU64 = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert_eq!(r.value, None);
    }

    #[test]
    fn u64_from_missing() {
        let r: TestU64 = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(r.value, None);
    }

    #[test]
    fn u64_rejects_negative_string() {
        let r = serde_json::from_str::<TestU64>(r#"{"value": "-1"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn u64_rejects_boolean() {
        let r = serde_json::from_str::<TestU64>(r#"{"value": true}"#);
        assert!(r.is_err());
    }
}
