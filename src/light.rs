//! The board light, through Home Assistant: on while boardvu runs, off when it stops.
//!
//! Settings live in `light.json` next to `board.json` (gitignored; see `light.example.json`). The token comes from
//! the `HA_API_KEY` environment variable, or a `HA_API_KEY=...` line in `~/.tokens` (only that line is read).

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct LightConfig {
    /// Home Assistant base URL, e.g. https://homeassistant.local:8123
    url: String,
    /// Light entity id, e.g. light.blackboard
    entity: String,
    /// Extra `light.turn_on` service data while boardvu runs (brightness, color_temp_kelvin, ...).
    #[serde(default)]
    on: serde_json::Map<String, serde_json::Value>,
}

pub struct Light {
    cfg: LightConfig,
    token: String,
    agent: ureq::Agent,
}

impl Light {
    /// Ok(None) if there's no light.json (light control is optional).
    pub fn from_config(path: &Path) -> Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let cfg: LightConfig =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let token = crate::secrets::token("HA_API_KEY")
            .ok_or_else(|| anyhow!("no HA_API_KEY in the environment or ~/.tokens"))?;
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(8)))
            .build()
            .into();
        Ok(Some(Light { cfg, token, agent }))
    }

    fn call(
        &self,
        service: &str,
        mut data: serde_json::Map<String, serde_json::Value>,
    ) -> Result<()> {
        data.insert("entity_id".into(), self.cfg.entity.clone().into());
        let url = format!(
            "{}/api/services/light/{service}",
            self.cfg.url.trim_end_matches('/')
        );
        self.agent
            .post(&url)
            .header("Authorization", &format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .send(serde_json::Value::Object(data).to_string())
            .map_err(|e| anyhow!("Home Assistant light.{service}: {e}"))?;
        Ok(())
    }

    pub fn on(&self) -> Result<()> {
        self.call("turn_on", self.cfg.on.clone())
    }

    pub fn off(&self) -> Result<()> {
        self.call("turn_off", Default::default())
    }
}

/// Turns the light on when created and off when dropped (normal exit, Ctrl-C, or an error).
pub struct LightGuard(Light);

impl LightGuard {
    pub fn new(light: Light) -> Self {
        match light.on() {
            Ok(()) => crate::say!("board light on"),
            Err(e) => crate::say!("warning: {e:#}"),
        }
        LightGuard(light)
    }
}

impl Drop for LightGuard {
    fn drop(&mut self) {
        match self.0.off() {
            Ok(()) => crate::say!("board light off"),
            Err(e) => crate::say!("warning: {e:#}"),
        }
    }
}
