//! Where boardcast keeps its files: config in `$XDG_CONFIG_HOME/boardcast` (default `~/.config/boardcast`), models in
//! `$XDG_DATA_HOME/boardcast/models` (default `~/.local/share/boardcast/models`).

use std::path::PathBuf;

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var).filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(fallback),
    }
    .join("boardcast")
}

/// A file in the config directory.
pub fn config(name: &str) -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join(name)
}

/// A model file (see scripts/fetch-model.sh and scripts/fetch-pose-model.sh).
pub fn model(name: &str) -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share")
        .join("models")
        .join(name)
}

/// The CUDA + cuDNN libraries for a `--features cuda` build (see scripts/fetch-cuda.sh).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn cuda() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("cuda")
}
