//! API tokens: from the environment, or a `NAME=...` line in `~/.tokens`. Only the one line asked for is read;
//! the file is never sourced (it holds other secrets).

use std::path::Path;

pub fn token(name: &str) -> Option<String> {
    if let Ok(t) = std::env::var(name)
        && !t.is_empty()
    {
        return Some(t);
    }
    let home = std::env::var_os("HOME")?;
    let text = std::fs::read_to_string(Path::new(&home).join(".tokens")).ok()?;
    text.lines().find_map(|l| parse_token_line(l, name))
}

/// `NAME=abc`, `export NAME="abc"` -> abc
fn parse_token_line(line: &str, name: &str) -> Option<String> {
    let line = line.trim_start();
    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let v = line.strip_prefix(name)?.strip_prefix('=')?;
    let v = v.trim().trim_matches(|c| c == '"' || c == '\'');
    (!v.is_empty()).then(|| v.to_string())
}

#[cfg(test)]
mod tests {
    fn p(l: &str) -> Option<String> {
        super::parse_token_line(l, "HA_API_KEY")
    }

    #[test]
    fn token_lines() {
        assert_eq!(p("HA_API_KEY=abc").as_deref(), Some("abc"));
        assert_eq!(p("export HA_API_KEY=\"abc\"").as_deref(), Some("abc"));
        assert_eq!(p("  export HA_API_KEY='abc'  ").as_deref(), Some("abc"));
        assert_eq!(p("OTHER_KEY=abc"), None);
        assert_eq!(p("HA_API_KEY_2=abc"), None);
        assert_eq!(p("HA_API_KEY="), None);
    }
}
