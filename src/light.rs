//! The board light, through Home Assistant: on while boardcast runs, off when it stops, and adjustable from the keys
//! (on/off, brightness, colour temperature).
//!
//! Settings live in `light.json` next to `board.json` (gitignored; see `light.example.json`). The token comes from
//! the `HA_API_KEY` environment variable, or a `HA_API_KEY=...` line in `~/.tokens` (only that line is read).
//!
//! Calls go out on a background thread so a slow Home Assistant never stalls the video; if several changes queue up
//! while one is in flight, only the newest is sent.

use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
pub struct LightConfig {
    /// Home Assistant base URL, e.g. https://homeassistant.local:8123
    url: String,
    /// Light entity id, e.g. light.blackboard
    entity: String,
    /// Extra `light.turn_on` service data while boardcast runs (brightness, color_temp_kelvin, ...).
    #[serde(default)]
    on: Map<String, Value>,
}

/// What the light should be doing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub on: bool,
    /// 1..=255 (Home Assistant's scale).
    pub brightness: u8,
    /// Colour temperature, if the config sets one (lights without colour temperature reject it).
    pub kelvin: Option<u32>,
}

/// Colour temperatures `k` steps through.
const KELVINS: [u32; 5] = [2700, 3500, 4500, 5500, 6500];
/// Brightness step for `-` / `=`: a tenth of full.
const STEP: i32 = 26;

impl State {
    pub fn brighter(self, steps: i32) -> State {
        let b = (self.brightness as i32 + steps * STEP).clamp(3, 255) as u8;
        State {
            on: true,
            brightness: b,
            ..self
        }
    }

    /// The next colour temperature up, wrapping from the warmest back round; starts at 6500 K if none was set.
    pub fn next_kelvin(self) -> State {
        let k = match self.kelvin {
            None => 6500,
            Some(k) => KELVINS
                .iter()
                .copied()
                .find(|&x| x > k)
                .unwrap_or(KELVINS[0]),
        };
        State {
            on: true,
            kelvin: Some(k),
            ..self
        }
    }

    pub fn describe(&self) -> String {
        if !self.on {
            return "off".into();
        }
        let pct = (self.brightness as f32 / 255.0 * 100.0).round();
        match self.kelvin {
            Some(k) => format!("on, {pct:.0}%, {k} K"),
            None => format!("on, {pct:.0}%"),
        }
    }
}

struct Client {
    cfg: LightConfig,
    token: String,
    agent: ureq::Agent,
}

impl Client {
    fn call(&self, service: &str, mut data: Map<String, Value>) -> Result<()> {
        data.insert("entity_id".into(), self.cfg.entity.clone().into());
        let url = format!(
            "{}/api/services/light/{service}",
            self.cfg.url.trim_end_matches('/')
        );
        self.agent
            .post(&url)
            .header("Authorization", &format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .send(Value::Object(data).to_string())
            .map_err(|e| anyhow!("Home Assistant light.{service}: {e}"))?;
        Ok(())
    }

    fn apply(&self, s: &State) -> Result<()> {
        if !s.on {
            return self.call("turn_off", Map::new());
        }
        // The config's extras, with our brightness and colour temperature (dropping its other spellings of them,
        // which Home Assistant won't take together).
        let mut data = self.cfg.on.clone();
        for k in [
            "brightness",
            "brightness_pct",
            "brightness_step",
            "brightness_step_pct",
            "color_temp",
            "kelvin",
            "color_temp_kelvin",
        ] {
            data.remove(k);
        }
        data.insert("brightness".into(), s.brightness.into());
        if let Some(k) = s.kelvin {
            data.insert("color_temp_kelvin".into(), k.into());
        }
        self.call("turn_on", data)
    }
}

struct Shared {
    /// The newest state not yet sent, and whether to stop once it's sent.
    want: Mutex<(Option<State>, bool)>,
    wake: Condvar,
    /// The last error, cleared by a successful call.
    error: Mutex<Option<String>>,
}

/// The board light: on when created, off when dropped (normal exit, Ctrl-C, or an error).
pub struct Light {
    state: State,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Light {
    /// Ok(None) if there's no light.json (light control is optional). Turns the light on before returning, so the
    /// camera's auto-exposure settles on the lit board.
    pub fn start(path: &Path) -> Result<Option<Self>> {
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
        let state = initial_state(&cfg.on);
        let client = Client { cfg, token, agent };
        let shared = Arc::new(Shared {
            want: Mutex::new((None, false)),
            wake: Condvar::new(),
            error: Mutex::new(None),
        });
        match client.apply(&state) {
            Ok(()) => crate::say!("board light {}", state.describe()),
            Err(e) => {
                crate::say!("warning: {e:#}");
                *shared.error.lock().unwrap() = Some(format!("{e:#}"));
            }
        }
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("light".into())
            .spawn(move || {
                loop {
                    let (next, stop) = {
                        let mut w = sh.want.lock().unwrap();
                        while w.0.is_none() && !w.1 {
                            w = sh.wake.wait(w).unwrap();
                        }
                        (w.0.take(), w.1)
                    };
                    if let Some(s) = next {
                        let r = client.apply(&s);
                        match &r {
                            Ok(()) if stop => crate::say!("board light off"),
                            Ok(()) => {}
                            Err(e) => crate::say!("warning: {e:#}"),
                        }
                        *sh.error.lock().unwrap() = r.err().map(|e| format!("{e:#}"));
                    }
                    if stop && sh.want.lock().unwrap().0.is_none() {
                        return;
                    }
                }
            })?;
        Ok(Some(Light {
            state,
            shared,
            thread: Some(thread),
        }))
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// Change the light (sent in the background).
    pub fn set(&mut self, s: State) {
        if s == self.state {
            return;
        }
        self.state = s;
        self.shared.want.lock().unwrap().0 = Some(s);
        self.shared.wake.notify_one();
    }

    /// For the status display.
    pub fn describe(&self) -> String {
        match &*self.shared.error.lock().unwrap() {
            Some(_) => format!("{} (Home Assistant failed: see log)", self.state.describe()),
            None => self.state.describe(),
        }
    }
}

impl Drop for Light {
    fn drop(&mut self) {
        {
            let mut w = self.shared.want.lock().unwrap();
            *w = (
                Some(State {
                    on: false,
                    ..self.state
                }),
                true,
            );
        }
        self.shared.wake.notify_one();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Brightness and colour temperature from the config's `on` extras (defaults: full, no colour temperature).
fn initial_state(on: &Map<String, Value>) -> State {
    let brightness = on
        .get("brightness")
        .and_then(Value::as_f64)
        .or_else(|| {
            on.get("brightness_pct")
                .and_then(Value::as_f64)
                .map(|p| p * 2.55)
        })
        .map_or(255, |b| b.round().clamp(1.0, 255.0) as u8);
    let kelvin = on
        .get("color_temp_kelvin")
        .or_else(|| on.get("kelvin"))
        .and_then(Value::as_u64)
        .map(|k| k as u32);
    State {
        on: true,
        brightness,
        kelvin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_brightness_and_colour_from_the_config() {
        let on: Map<String, Value> =
            serde_json::from_str(r#"{"brightness": 200, "color_temp_kelvin": 6500}"#).unwrap();
        assert_eq!(
            initial_state(&on),
            State {
                on: true,
                brightness: 200,
                kelvin: Some(6500)
            }
        );
        let pct: Map<String, Value> = serde_json::from_str(r#"{"brightness_pct": 50}"#).unwrap();
        assert_eq!(
            initial_state(&pct),
            State {
                on: true,
                brightness: 127,
                kelvin: None
            }
        );
    }

    #[test]
    fn brightness_steps_stay_in_range_and_kelvin_wraps() {
        let s = State {
            on: false,
            brightness: 250,
            kelvin: Some(6500),
        };
        assert_eq!(s.brighter(1).brightness, 255);
        assert!(s.brighter(1).on, "brightening turns it on");
        assert_eq!(s.brighter(-20).brightness, 3);
        assert_eq!(s.next_kelvin().kelvin, Some(2700));
        assert_eq!(
            State {
                kelvin: Some(3000),
                ..s
            }
            .next_kelvin()
            .kelvin,
            Some(3500)
        );
    }
}
