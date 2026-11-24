//! Full-screen status while streaming: feeds, view, board tracking, performance, log, keys.

use std::io::IsTerminal;
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

use crate::calib::Tracker;
use crate::enhance::Look;

pub struct Feed<'a> {
    pub path: &'a str,
    pub size: (usize, usize),
    /// Something has the device open (a meeting app, ffplay, OBS).
    pub watched: bool,
}

#[derive(Default, Clone, Copy)]
pub struct Perf {
    pub fps: f64,
    pub wait_ms: f64,
    pub decode_ms: f64,
    pub work_ms: f64,
    /// CPU used by boardvu, in cores.
    pub cpu: f64,
}

pub struct Status<'a> {
    pub camera: (&'a str, usize, usize),
    pub board: Feed<'a>,
    pub raw: Option<Feed<'a>>,
    pub look: Look,
    /// Ghost opacity; None when ghost mode is off.
    pub ghost: Option<f32>,
    /// Picture-in-picture state; None without ghost mode.
    pub pip: Option<String>,
    /// Board light state; None without light control.
    pub light: Option<String>,
    pub focus: String,
    /// Left-hand pointer state; None without ghost mode.
    pub pointer: Option<String>,
    pub tracker: Option<&'a Tracker>,
    pub perf: Perf,
    pub started: Instant,
}

/// Restores the terminal when dropped (normal exit, error, or panic via ratatui's hook).
pub struct Tui {
    term: DefaultTerminal,
    last_draw: Instant,
}

impl Tui {
    /// None when stdin or stdout isn't a terminal (then messages go to stderr as plain lines).
    pub fn start() -> Option<Self> {
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            return None;
        }
        let term = ratatui::init();
        crate::log::set_tui(true);
        Some(Tui {
            term,
            last_draw: Instant::now() - Duration::from_secs(1),
        })
    }

    /// Keys pressed since the last call; Ctrl-C and Esc come back as `q`.
    pub fn keys(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while event::poll(Duration::ZERO).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read()
                && k.kind == KeyEventKind::Press
            {
                match k.code {
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        out.push(b'q')
                    }
                    KeyCode::Esc => out.push(b'q'),
                    KeyCode::Char(c) if c.is_ascii() => out.push(c.to_ascii_lowercase() as u8),
                    _ => {}
                }
            }
        }
        out
    }

    /// Redraw, at most 5 times a second.
    pub fn draw(&mut self, s: &Status) {
        if self.last_draw.elapsed() < Duration::from_millis(200) {
            return;
        }
        self.last_draw = Instant::now();
        let _ = self.term.draw(|f| render(f, s));
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        ratatui::restore();
        crate::log::set_tui(false);
    }
}

const LABEL: Style = Style::new().fg(Color::DarkGray);

fn kv<'a>(k: &'a str, v: impl Into<Line<'a>>) -> Line<'a> {
    let mut spans = vec![Span::styled(format!("{k:<9}"), LABEL)];
    spans.extend(v.into().spans);
    Line::from(spans)
}

fn dot(on: bool) -> Span<'static> {
    if on {
        "● ".green()
    } else {
        "○ ".dark_gray()
    }
}

fn block(title: &str) -> Block<'_> {
    Block::bordered().title(Span::styled(
        format!(" {title} "),
        Style::new().add_modifier(Modifier::BOLD),
    ))
}

fn ago(t: Instant) -> String {
    let s = t.elapsed().as_secs();
    if s < 60 {
        format!("{s} s ago")
    } else {
        format!("{} min ago", s / 60)
    }
}

fn render(f: &mut ratatui::Frame, s: &Status) {
    let [title, top, mid, log, keys] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(8),
        Constraint::Length(6),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(f.area());

    let up = s.started.elapsed().as_secs();
    f.render_widget(
        Line::from(vec![
            " boardvu ".bold().black().on_green(),
            format!("  up {}:{:02}:{:02}", up / 3600, up / 60 % 60, up % 60).dark_gray(),
        ]),
        title,
    );

    let [feeds, view] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(top);
    let feed = |name: &'static str, fd: &Feed| {
        kv(
            name,
            vec![
                dot(fd.watched),
                format!("{} {}x{}", fd.path, fd.size.0, fd.size.1).into(),
                if fd.watched {
                    "  in use".green()
                } else {
                    "  idle".dark_gray()
                },
            ],
        )
    };
    let mut lines = vec![kv(
        "camera",
        format!("{} {}x{}", s.camera.0, s.camera.1, s.camera.2),
    )];
    lines.push(feed("board", &s.board));
    match &s.raw {
        Some(r) => lines.push(feed("raw", r)),
        None => lines.push(kv("raw", "off".dark_gray())),
    }
    lines.push(match s.tracker {
        None => kv("tracking", "off".dark_gray()),
        Some(t) if t.failures >= 3 => kv(
            "tracking",
            format!("can't see the edges ({} checks)", t.failures).yellow(),
        ),
        Some(t) => match (t.checked, t.shift) {
            (Some(at), Some(px)) => {
                kv("tracking", format!("aligned, {px:.1} px off ({})", ago(at)))
            }
            _ => kv("tracking", "waiting for the first check".dark_gray()),
        },
    });
    f.render_widget(Paragraph::new(lines).block(block("feeds")), feeds);

    let look = |l: Look| {
        let name = format!("{l:?}").to_lowercase();
        if l == s.look {
            Span::styled(format!("[{name}]"), Style::new().bold().fg(Color::Cyan))
        } else {
            Span::styled(format!(" {name} "), LABEL)
        }
    };
    let mut lines = vec![kv(
        "look",
        vec![look(Look::Plain), look(Look::Natural), look(Look::Paper)],
    )];
    lines.push(match s.ghost {
        None => kv("ghost", "off (--no-ghost)".dark_gray()),
        Some(o) => {
            let step = |v: f32, name: &'static str| {
                if (o - v).abs() < 1e-3 {
                    Span::styled(format!("[{name}]"), Style::new().bold().fg(Color::Cyan))
                } else {
                    Span::styled(format!(" {name} "), LABEL)
                }
            };
            let mut v = vec![
                step(0.0, "hidden"),
                step(0.1, "0.1"),
                step(0.25, "0.25"),
                step(0.5, "0.5"),
                step(1.0, "opaque"),
            ];
            if !crate::ghost::OPACITY_STEPS
                .iter()
                .any(|s| (o - s).abs() < 1e-3)
            {
                v.push(Span::styled(
                    format!("[{o:.2}]"),
                    Style::new().bold().fg(Color::Cyan),
                ));
            }
            kv("ghost", v)
        }
    });
    lines.push(match &s.pip {
        None => kv("pip", "needs ghost mode".dark_gray()),
        Some(d) => kv("pip", vec![dot(d != "off"), d.clone().into()]),
    });
    lines.push(match &s.light {
        None => kv("light", "not controlled".dark_gray()),
        Some(d) => kv("light", vec![dot(!d.starts_with("off")), d.clone().into()]),
    });
    lines.push(kv("focus", s.focus.clone()));
    lines.push(match &s.pointer {
        None => kv("pointer", "needs ghost mode and the hand model".dark_gray()),
        Some(d) => kv("pointer", vec![dot(d != "off"), d.clone().into()]),
    });
    f.render_widget(Paragraph::new(lines).block(block("view")), view);

    let p = s.perf;
    let lines = vec![
        kv("fps", format!("{:.1}", p.fps).bold()),
        kv("cpu", format!("{:.0}% ({:.1} cores)", p.cpu * 100.0, p.cpu)),
        kv(
            "camera",
            format!(
                "{:.1} ms waiting, {:.1} ms decoding",
                p.wait_ms, p.decode_ms
            ),
        ),
        kv(
            "work",
            format!("{:.1} ms per frame (warp, ghost, look, write)", p.work_ms),
        ),
    ];
    f.render_widget(Paragraph::new(lines).block(block("performance")), mid);

    let n = log.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = crate::log::recent(n)
        .into_iter()
        .map(|(t, m)| {
            let style = if m.starts_with("warning") || m.contains("failed") {
                Style::new().yellow()
            } else {
                Style::new()
            };
            Line::from(vec![
                Span::styled(format!("{t} "), LABEL),
                Span::styled(m, style),
            ])
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block("log")),
        log,
    );

    let key = |k: &'static str, what: &'static str| {
        vec![
            Span::styled(format!(" {k} "), Style::new().black().on_gray()),
            Span::raw(format!(" {what}  ")),
        ]
    };
    let mut spans = Vec::new();
    for (k, w) in [
        ("l", "look"),
        ("g", "ghost"),
        ("p", "picture-in-picture"),
        ("o", "move it"),
        ("b", "light"),
        ("-=", "brightness"),
        ("k", "colour"),
        ("f", "focus"),
        (",.", "adjust"),
        ("w", "pointer"),
        ("q", "quit"),
    ] {
        spans.extend(key(k, w));
    }
    f.render_widget(Line::from(spans), keys);
}
