//! Bash command utilities: splitting, normalization, and human-readable summarization.
//!
//! Provides three key utilities for bash command processing:
//! - [`split_command`]: Splits compound bash commands on shell operators (`&&`, `||`, `;`, `|`)
//!   while respecting quoted strings. Used for safety validation to ensure every segment
//!   of a chained command is checked independently.
//! - [`normalize_command`]: Normalizes whitespace in a command string — trims leading/trailing
//!   whitespace and collapses runs of internal whitespace to a single space.
//! - [`summarize_command`]: Produces short human-readable descriptions of bash commands
//!   for debug logging (e.g., `forge test VLN-01.t.sol`, `write file.sol`, `cast call 0x1234...`).

/// Split a bash command string on shell operators (`&&`, `||`, `;`, `|`).
///
/// Returns individual command segments with operators removed and whitespace trimmed.
/// Handles quoted strings (single and double) to avoid splitting inside them.
///
/// # Examples
///
/// ```
/// use sombrax_agentic_core::tools::shell::split_command;
///
/// let segments = split_command("forge build && forge test");
/// assert_eq!(segments, vec!["forge build", "forge test"]);
///
/// let segments = split_command("echo hello; rm -rf /");
/// assert_eq!(segments, vec!["echo hello", "rm -rf /"]);
///
/// // Quoted strings are preserved
/// let segments = split_command(r#"grep "a && b" file.txt"#);
/// assert_eq!(segments, vec![r#"grep "a && b" file.txt"#]);
/// ```
pub fn split_command(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                current.push(c);
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                current.push(c);
            }
            '&' if !in_single_quote && !in_double_quote => {
                if chars.peek() == Some(&'&') {
                    chars.next(); // consume second '&'
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        segments.push(trimmed);
                    }
                    current.clear();
                } else {
                    current.push(c); // single '&' (background), keep as-is
                }
            }
            '|' if !in_single_quote && !in_double_quote => {
                if chars.peek() == Some(&'|') {
                    chars.next(); // consume second '|'
                }
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    segments.push(trimmed);
                }
                current.clear();
            }
            ';' if !in_single_quote && !in_double_quote => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    segments.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        segments.push(trimmed);
    }

    if segments.is_empty() {
        segments.push(command.trim().to_string());
    }

    segments
}

/// Normalize whitespace in a command string.
///
/// Trims leading/trailing whitespace and collapses runs of internal whitespace
/// to a single space. This ensures consistent matching regardless of how the
/// LLM formats the command.
///
/// # Examples
/// ```
/// use sombrax_agentic_core::tools::shell::normalize_command;
/// assert_eq!(normalize_command("  forge   build   --via-ir  "), "forge build --via-ir");
/// assert_eq!(normalize_command("forge build"), "forge build");
/// ```
pub fn normalize_command(cmd: &str) -> String {
    cmd.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Summarize a bash command into a short human-readable description for debug logging.
///
/// Produces concise descriptions that are easy to scan in log output:
/// - `cat > file.sol << 'EOF' ...` → `"write file.sol"`
/// - `forge build --via-ir ./test/VLN-01.t.sol` → `"forge build VLN-01.t.sol"`
/// - `forge test --via-ir --match-path ./test/VLN-01.t.sol ...` → `"forge test VLN-01.t.sol"`
/// - `cast call 0x1234... "balanceOf(address)" ...` → `"cast call 0x1234..."`
/// - `cd /some/path && forge build` → `"cd /some/path && forge build"`
/// - `ls -la src/` → `"ls src/"`
/// - `grep -r "pattern" .` → `"grep pattern"`
///
/// Chained commands (`&&`, `|`) are recursively summarized.
/// Unknown commands are truncated to 40 characters.
pub fn summarize_command(cmd: &str) -> String {
    let cmd = cmd.trim();

    // cd command (check early — before forge checks since "cd x && forge build" contains "forge build")
    if let Some(rest) = cmd.strip_prefix("cd ") {
        let rest = rest.trim();
        if let Some(pos) = rest.find("&&") {
            let dir = rest[..pos].trim();
            let next = rest[pos + 2..].trim();
            let next_summary = summarize_command(next);
            return format!("cd {} && {}", dir, next_summary);
        }
        return format!("cd {}", rest);
    }

    // Chained commands with && (check early — before forge checks for same reason)
    if cmd.contains(" && ") {
        let parts: Vec<&str> = cmd.splitn(3, "&&").collect();
        if parts.len() >= 2 {
            let first = summarize_command(parts[0].trim());
            let second = summarize_command(parts[1].trim());
            if parts.len() > 2 {
                return format!("{} && {} && ...", first, second);
            }
            return format!("{} && {}", first, second);
        }
    }

    // Piped commands (check early — before forge checks for same reason)
    if cmd.contains(" | ") {
        let first = cmd.split('|').next().unwrap_or(cmd).trim();
        let summary = summarize_command(first);
        return format!("{} | ...", summary);
    }

    // cat/heredoc write patterns: cat > file << 'EOF', cat > file << "EOF"
    if cmd.starts_with("cat ") && (cmd.contains("<<") || cmd.contains("> ")) {
        if let Some(file) = extract_redirect_target(cmd) {
            let filename = extract_filename(file);
            return format!("write {}", filename);
        }
    }

    // echo/printf redirect: echo "..." > file, printf "..." > file
    if (cmd.starts_with("echo ") || cmd.starts_with("printf ")) && cmd.contains("> ") {
        if let Some(file) = extract_redirect_target(cmd) {
            let filename = extract_filename(file);
            return format!("write {}", filename);
        }
    }

    // forge build: extract test file path
    if cmd.contains("forge build") {
        if let Some(file) = extract_sol_file(cmd) {
            return format!("forge build {}", file);
        }
        return "forge build".to_string();
    }

    // forge test: extract test file path
    if cmd.contains("forge test") {
        if let Some(file) = extract_sol_file(cmd) {
            return format!("forge test {}", file);
        }
        return "forge test".to_string();
    }

    // forge clean / forge install
    if cmd.contains("forge clean") {
        return "forge clean".to_string();
    }
    if cmd.contains("forge install") {
        return "forge install".to_string();
    }

    // cast commands: cast <subcommand> <target>
    if cmd.starts_with("cast ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.len() >= 3 {
            let subcmd = parts[1];
            let target = parts[2];
            let short_target = if target.len() > 12 {
                format!("{}...", &target[..10])
            } else {
                target.to_string()
            };
            return format!("cast {} {}", subcmd, short_target);
        } else if parts.len() == 2 {
            return format!("cast {}", parts[1]);
        }
    }

    // mkdir
    if cmd.starts_with("mkdir ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if let Some(dir) = parts.last() {
            return format!("mkdir {}", extract_filename(dir));
        }
    }

    // rm
    if cmd.starts_with("rm ") {
        let parts: Vec<&str> = cmd
            .split_whitespace()
            .filter(|p| !p.starts_with('-'))
            .collect();
        if parts.len() >= 2 {
            return format!("rm {}", extract_filename(parts[1]));
        }
    }

    // sed
    if cmd.starts_with("sed ") {
        if let Some(file) = cmd.split_whitespace().last() {
            return format!("sed {}", extract_filename(file));
        }
    }

    // grep/rg/ag
    if cmd.starts_with("grep ") || cmd.starts_with("rg ") || cmd.starts_with("ag ") {
        let tool = cmd.split_whitespace().next().unwrap_or("grep");
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        for part in parts.iter().skip(1) {
            if !part.starts_with('-') {
                let pattern = if part.len() > 20 {
                    format!("{}...", &part[..17])
                } else {
                    part.to_string()
                };
                return format!("{} {}", tool, pattern.trim_matches('"').trim_matches('\''));
            }
        }
        return tool.to_string();
    }

    // ls/find/tree
    if cmd.starts_with("ls ") || cmd.starts_with("find ") || cmd.starts_with("tree ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let tool = parts[0];
        for part in parts.iter().skip(1) {
            if !part.starts_with('-') {
                return format!("{} {}", tool, part);
            }
        }
        return tool.to_string();
    }

    // cargo commands
    if cmd.starts_with("cargo ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.len() >= 2 {
            return format!("cargo {}", parts[1]);
        }
    }

    // git commands
    if cmd.starts_with("git ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.len() >= 2 {
            return format!("git {}", parts[1]);
        }
    }

    // npm/yarn/pnpm commands
    if cmd.starts_with("npm ") || cmd.starts_with("yarn ") || cmd.starts_with("pnpm ") {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.len() >= 2 {
            return format!("{} {}", parts[0], parts[1]);
        }
    }

    // Fallback: first 40 chars of the command
    if cmd.len() > 40 {
        format!("{}...", &cmd[..37])
    } else {
        cmd.to_string()
    }
}

/// Extract the redirect target file from a bash command (e.g., `cat > file.sol << ...`).
fn extract_redirect_target(cmd: &str) -> Option<&str> {
    let redirect_pos = cmd.find("> ")?;
    let after_redirect = &cmd[redirect_pos + 2..];
    let file = after_redirect.split_whitespace().next()?;
    if file.starts_with('>') || file.starts_with('<') {
        return None;
    }
    Some(file)
}

/// Extract a .sol file reference from a forge command string.
fn extract_sol_file(cmd: &str) -> Option<String> {
    // Look for --match-path argument first
    if let Some(pos) = cmd.find("--match-path") {
        let after = cmd[pos + 12..].trim_start();
        if let Some(path) = after.split_whitespace().next() {
            let path = path.trim_matches('"').trim_matches('\'');
            return Some(extract_filename(path));
        }
    }
    // Look for .t.sol or .sol file as a positional argument
    for part in cmd.split_whitespace() {
        let clean = part.trim_matches('"').trim_matches('\'');
        if clean.ends_with(".sol") {
            return Some(extract_filename(clean));
        }
    }
    None
}

/// Extract filename from a path string.
fn extract_filename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // split_command tests
    // ================================================================

    #[test]
    fn test_split_simple() {
        assert_eq!(split_command("forge build"), vec!["forge build"]);
    }

    #[test]
    fn test_split_and() {
        assert_eq!(
            split_command("forge build && forge test"),
            vec!["forge build", "forge test"]
        );
    }

    #[test]
    fn test_split_semicolon() {
        assert_eq!(
            split_command("echo hello; rm -rf /"),
            vec!["echo hello", "rm -rf /"]
        );
    }

    #[test]
    fn test_split_pipe() {
        assert_eq!(
            split_command("forge test | grep error"),
            vec!["forge test", "grep error"]
        );
    }

    #[test]
    fn test_split_or() {
        assert_eq!(
            split_command("forge test || echo failed"),
            vec!["forge test", "echo failed"]
        );
    }

    #[test]
    fn test_split_respects_double_quotes() {
        assert_eq!(
            split_command(r#"forge test --match-test "a && b""#),
            vec![r#"forge test --match-test "a && b""#]
        );
    }

    #[test]
    fn test_split_respects_single_quotes() {
        assert_eq!(
            split_command("grep 'a && b' file.txt"),
            vec!["grep 'a && b' file.txt"]
        );
    }

    #[test]
    fn test_split_triple_chain() {
        assert_eq!(
            split_command("echo a && echo b && echo c"),
            vec!["echo a", "echo b", "echo c"]
        );
    }

    #[test]
    fn test_split_mixed_operators() {
        assert_eq!(
            split_command("echo a && echo b; echo c | echo d"),
            vec!["echo a", "echo b", "echo c", "echo d"]
        );
    }

    #[test]
    fn test_split_background_ampersand() {
        // Single & should NOT split (it's background, not chaining)
        let segments = split_command("sleep 10 &");
        assert_eq!(segments, vec!["sleep 10 &"]);
    }

    // ================================================================
    // ================================================================
    // normalize_command tests
    // ================================================================

    #[test]
    fn test_normalize_simple() {
        assert_eq!(normalize_command("forge build"), "forge build");
    }

    #[test]
    fn test_normalize_extra_spaces() {
        assert_eq!(
            normalize_command("forge   build   --via-ir"),
            "forge build --via-ir"
        );
    }

    #[test]
    fn test_normalize_leading_trailing() {
        assert_eq!(normalize_command("  forge build  "), "forge build");
    }

    #[test]
    fn test_normalize_tabs_and_spaces() {
        assert_eq!(
            normalize_command("forge\t  build\t--via-ir"),
            "forge build --via-ir"
        );
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_command(""), "");
        assert_eq!(normalize_command("   "), "");
    }

    // ================================================================
    // summarize_command tests
    // ================================================================

    #[test]
    fn test_summarize_cat_heredoc() {
        let cmd = "cat > test/VLN-01.t.sol << 'EOF'\n// content\nEOF";
        assert_eq!(summarize_command(cmd), "write VLN-01.t.sol");
    }

    #[test]
    fn test_summarize_forge_build() {
        assert_eq!(
            summarize_command("forge build --via-ir ./test/VLN-01_20260202.t.sol"),
            "forge build VLN-01_20260202.t.sol"
        );
    }

    #[test]
    fn test_summarize_forge_build_no_file() {
        assert_eq!(summarize_command("forge build --via-ir"), "forge build");
    }

    #[test]
    fn test_summarize_forge_test_match_path() {
        assert_eq!(
            summarize_command("forge test --via-ir --match-path ./test/VLN-01.t.sol --match-test testExploit -vvvv"),
            "forge test VLN-01.t.sol"
        );
    }

    #[test]
    fn test_summarize_forge_clean() {
        assert_eq!(summarize_command("forge clean"), "forge clean");
    }

    #[test]
    fn test_summarize_forge_install() {
        assert_eq!(
            summarize_command("forge install openzeppelin/contracts"),
            "forge install"
        );
    }

    #[test]
    fn test_summarize_cast_call() {
        assert_eq!(
            summarize_command(
                "cast call 0x1234567890abcdef1234 \"balanceOf(address)\" 0xABCD --rpc-url $RPC"
            ),
            "cast call 0x12345678..."
        );
    }

    #[test]
    fn test_summarize_cast_short_target() {
        assert_eq!(summarize_command("cast block latest"), "cast block latest");
    }

    #[test]
    fn test_summarize_cd_chained() {
        assert_eq!(
            summarize_command("cd /some/path && forge build --via-ir"),
            "cd /some/path && forge build"
        );
    }

    #[test]
    fn test_summarize_grep() {
        assert_eq!(summarize_command("grep -r import src/"), "grep import");
    }

    #[test]
    fn test_summarize_ls() {
        assert_eq!(summarize_command("ls -la src/"), "ls src/");
    }

    #[test]
    fn test_summarize_echo_redirect() {
        assert_eq!(
            summarize_command("echo \"content\" > output.txt"),
            "write output.txt"
        );
    }

    #[test]
    fn test_summarize_sed() {
        assert_eq!(
            summarize_command("sed -i 's/old/new/g' test/VLN-01.t.sol"),
            "sed VLN-01.t.sol"
        );
    }

    #[test]
    fn test_summarize_rm() {
        assert_eq!(
            summarize_command("rm -rf test/old_test.t.sol"),
            "rm old_test.t.sol"
        );
    }

    #[test]
    fn test_summarize_mkdir() {
        assert_eq!(
            summarize_command("mkdir -p reconnaissance"),
            "mkdir reconnaissance"
        );
    }

    #[test]
    fn test_summarize_piped() {
        assert_eq!(
            summarize_command("forge test --via-ir -vvvv | grep -i error"),
            "forge test | ..."
        );
    }

    #[test]
    fn test_summarize_cargo() {
        assert_eq!(summarize_command("cargo build --release"), "cargo build");
    }

    #[test]
    fn test_summarize_git() {
        assert_eq!(summarize_command("git status"), "git status");
    }

    #[test]
    fn test_summarize_npm() {
        assert_eq!(summarize_command("npm install"), "npm install");
    }

    #[test]
    fn test_summarize_long_fallback() {
        let cmd =
            "some-unknown-command --with-many-args --flag1 --flag2 --flag3 value1 value2 value3";
        let result = summarize_command(cmd);
        assert!(result.len() <= 40, "Fallback should truncate: {}", result);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_summarize_short_fallback() {
        assert_eq!(summarize_command("whoami"), "whoami");
    }
}
