//! Messages while streaming: kept for the TUI's log panel while it's up, printed to stderr otherwise.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static LINES: Mutex<VecDeque<(String, String)>> = Mutex::new(VecDeque::new());
static TUI: AtomicBool = AtomicBool::new(false);
const KEEP: usize = 200;

/// `say!("...", args)`: like eprintln!, but into the TUI's log when it's showing.
#[macro_export]
macro_rules! say {
    ($($t:tt)*) => { $crate::log::say(format!($($t)*)) };
}

/// Always kept (so the TUI shows what happened before it started); printed too while there's no TUI.
pub fn say(msg: String) {
    if !TUI.load(Ordering::Relaxed) {
        eprintln!("{msg}");
    }
    let mut lines = LINES.lock().unwrap();
    for part in msg.lines() {
        lines.push_back((clock(), part.to_string()));
    }
    while lines.len() > KEEP {
        lines.pop_front();
    }
}

pub fn set_tui(on: bool) {
    TUI.store(on, Ordering::Relaxed);
}

/// The newest `n` lines, oldest first.
pub fn recent(n: usize) -> Vec<(String, String)> {
    let lines = LINES.lock().unwrap();
    lines
        .iter()
        .skip(lines.len().saturating_sub(n))
        .cloned()
        .collect()
}

/// Local wall-clock time, HH:MM:SS.
fn clock() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}
