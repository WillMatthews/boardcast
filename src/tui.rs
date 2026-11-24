//! Full-screen status while streaming: feeds, view, board tracking, LaTeX costs, performance, log, keys.

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
use crate::latex::{State, Transcriber};

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
    pub latex: Option<&'a Transcriber>,
    pub latex_view: bool,
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

fn dollars(x: f64) -> String {
    if x < 0.01 {
        format!("${x:.4}")
    } else {
        format!("${x:.2}")
    }
}

fn render(f: &mut ratatui::Frame, s: &Status) {
    let [title, top, mid, log, keys] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(6),
        Constraint::Length(7),
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
        if l == s.look && !(s.latex_view && s.latex.is_some_and(|t| t.page.is_some())) {
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
    lines.push(match s.latex {
        None => kv("LaTeX", "unavailable (see log)".dark_gray()),
        Some(t) => {
            let showing = s.latex_view && t.page.is_some();
            let mut v = vec![
                dot(showing),
                if showing {
                    "showing".cyan().bold()
                } else if s.latex_view {
                    "on".into()
                } else {
                    "off".into()
                },
            ];
            if let State::Working(since) = t.state {
                v.push(format!("  transcribing… {} s", since.elapsed().as_secs()).yellow());
            }
            kv("LaTeX", v)
        }
    });
    f.render_widget(Paragraph::new(lines).block(block("view")), view);

    let [cost, perf] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(mid);
    let lines = match s.latex {
        None => vec![Line::from("LaTeX view off".dark_gray())],
        Some(t) => {
            let model = if t.p.effort.is_empty() {
                t.p.model.clone()
            } else {
                format!("{} ({} effort)", t.p.model, t.p.effort)
            };
            let mut l = vec![kv("model", model)];
            if t.calls == 0 {
                l.push(kv("last", "no transcriptions yet".dark_gray()));
            } else {
                l.push(kv(
                    "last",
                    format!(
                        "{} in {:.0} s · {}k in / {}k out",
                        dollars(t.last_cost),
                        t.last_secs,
                        (t.last_usage.input as f64 / 1e3 * 10.0).round() / 10.0,
                        (t.last_usage.output as f64 / 1e3 * 10.0).round() / 10.0
                    ),
                ));
            }
            l.push(kv(
                "session",
                vec![
                    dollars(t.total_cost()).bold(),
                    format!("  {} call{}", t.calls, if t.calls == 1 { "" } else { "s" }).into(),
                    if t.calls > 0 {
                        format!(", {} each", dollars(t.total_cost() / t.calls as f64)).dark_gray()
                    } else {
                        "".into()
                    },
                ],
            ));
            let per = if t.calls > 0 {
                t.total_cost() / t.calls as f64
            } else {
                1.3e3 * t.p.price_in / 1e6 + 2e3 * t.p.price_out / 1e6
            };
            l.push(kv(
                "estimate",
                format!(
                    "{} per transcription · 100 for {}",
                    dollars(per),
                    dollars(per * 100.0)
                )
                .dark_gray(),
            ));
            l.push(kv(
                "prices",
                format!(
                    "${:.2} / ${:.2} per M tokens in / out",
                    t.p.price_in, t.p.price_out
                )
                .dark_gray(),
            ));
            l
        }
    };
    f.render_widget(Paragraph::new(lines).block(block("LaTeX cost")), cost);

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
    f.render_widget(Paragraph::new(lines).block(block("performance")), perf);

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
        ("t", "LaTeX"),
        ("r", "re-transcribe"),
        ("q", "quit"),
    ] {
        spans.extend(key(k, w));
    }
    f.render_widget(Line::from(spans), keys);
}
