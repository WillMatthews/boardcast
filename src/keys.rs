//! Single-key commands from the terminal while streaming (no Enter needed).

use std::io::Read;
use std::sync::mpsc::{Receiver, channel};

/// Puts the terminal in no-echo, unbuffered mode and restores it when dropped.
pub struct Keys {
    pub rx: Receiver<u8>,
    saved: Option<libc::termios>,
}

impl Keys {
    /// Starts reading keys from stdin. If stdin isn't a terminal, no keys ever arrive.
    pub fn start() -> Self {
        let (tx, rx) = channel();
        let saved = unsafe {
            if libc::isatty(0) != 1 {
                return Keys { rx, saved: None };
            }
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut t) != 0 {
                return Keys { rx, saved: None };
            }
            let saved = t;
            // Keep ISIG so Ctrl-C still stops boardcast.
            t.c_lflag &= !(libc::ICANON | libc::ECHO);
            t.c_cc[libc::VMIN] = 1;
            t.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(0, libc::TCSANOW, &t);
            saved
        };
        std::thread::Builder::new()
            .name("keys".into())
            .spawn(move || {
                let mut b = [0u8; 1];
                while std::io::stdin().read_exact(&mut b).is_ok() {
                    if tx.send(b[0]).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning key reader");
        Keys {
            rx,
            saved: Some(saved),
        }
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        if let Some(t) = self.saved {
            unsafe { libc::tcsetattr(0, libc::TCSANOW, &t) };
        }
    }
}
