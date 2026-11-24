//! Where boardvu keeps its files: config in `$XDG_CONFIG_HOME/boardvu` (default `~/.config/boardvu`), models in
//! `$XDG_DATA_HOME/boardvu/models` (default `~/.local/share/boardvu/models`).

use std::path::PathBuf;

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var).filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(fallback),
    }
    .join("boardvu")
}

/// A file in the config directory.
pub fn config(name: &str) -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join(name)
}

/// A model file (see scripts/fetch-model.sh and scripts/fetch-pose-model.sh).
pub fn model(name: &str) -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("models").join(name)
}
