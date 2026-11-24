//! The LaTeX view: the board transcribed by a vision model (OpenAI Responses API) into LaTeX + TikZ, typeset with
//! pdflatex, and shown in place of the chalk.
//!
//! Work happens on a background thread; the video keeps running. Every transcription is kept under
//! `out/latex/<time>/` (input.png, board.tex, board.pdf). pdflatex runs without shell escape and can't read files
//! outside its work directory, since the .tex comes from a model.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};

use crate::image::{ResampleMap, Rgb};

#[derive(Clone)]
pub struct Params {
    pub model: String,
    pub effort: String,
    /// US dollars per million tokens.
    pub price_in: f64,
    pub price_out: f64,
    pub out_dir: PathBuf,
    /// Real board width / height.
    pub aspect: f64,
}

#[derive(Clone, Copy, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
}

impl Usage {
    pub fn cost(&self, p: &Params) -> f64 {
        (self.input as f64 * p.price_in + self.output as f64 * p.price_out) / 1e6
    }
}

struct Job {
    image: Rgb,
    canvas: (usize, usize),
    board: (usize, usize),
    bg: [u8; 3],
}

struct Done {
    result: Result<(Rgb, PathBuf)>,
    usage: Usage,
}

pub enum State {
    Idle,
    Working(Instant),
    Ready,
    Failed(String),
}

/// What the view loop sees: the latest typeset page, the work in progress, and what it has cost.
pub struct Transcriber {
    pub p: Params,
    tx: Sender<Job>,
    rx: Receiver<Done>,
    pub state: State,
    pub page: Option<Rgb>,
    /// Fingerprint of the board the current page (or the job in progress) was made from.
    made_from: Option<Vec<u8>>,
    pub calls: u32,
    pub usage: Usage,
    pub last_cost: f64,
    pub last_secs: f64,
    pub last_dir: Option<PathBuf>,
}

impl Transcriber {
    pub fn start(p: Params) -> Result<Self> {
        let key = crate::secrets::token("OPENAI_API_KEY")
            .ok_or_else(|| anyhow!("no OPENAI_API_KEY in the environment or ~/.tokens"))?;
        for tool in ["pdflatex", "pdftoppm"] {
            Command::new(tool)
                .arg("-v")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .with_context(|| {
                    format!("the LaTeX view needs {tool} (texlive / poppler-utils)")
                })?;
        }
        let (tx, jobs) = channel::<Job>();
        let (done, rx) = channel();
        let wp = p.clone();
        std::thread::Builder::new()
            .name("latex".into())
            .spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .timeout_global(Some(Duration::from_secs(180)))
                    .http_status_as_error(false)
                    .build()
                    .into();
                for job in jobs {
                    let mut usage = Usage::default();
                    let result = transcribe(&agent, &key, &wp, &job, &mut usage);
                    if done.send(Done { result, usage }).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Transcriber {
            p,
            tx,
            rx,
            state: State::Idle,
            page: None,
            made_from: None,
            calls: 0,
            usage: Usage::default(),
            last_cost: 0.0,
            last_secs: 0.0,
            last_dir: None,
        })
    }

    pub fn busy(&self) -> bool {
        matches!(self.state, State::Working(_))
    }

    pub fn total_cost(&self) -> f64 {
        self.usage.cost(&self.p)
    }

    /// Start a transcription of `paper` (the board in the paper look, me removed) unless one is running, or
    /// (without `force`) the page already shows this board.
    pub fn request(
        &mut self,
        paper: &Rgb,
        canvas: (usize, usize),
        board: (usize, usize),
        bg: [u8; 3],
        force: bool,
    ) -> bool {
        if self.busy() {
            return false;
        }
        let fp = fingerprint(paper);
        if !force
            && self.page.is_some()
            && self
                .made_from
                .as_ref()
                .is_some_and(|old| same_board(old, &fp))
        {
            return false;
        }
        self.made_from = Some(fp);
        self.state = State::Working(Instant::now());
        let _ = self.tx.send(Job {
            image: paper.clone(),
            canvas,
            board,
            bg,
        });
        true
    }

    /// Collect a finished transcription, if any. Returns true when there's a new page.
    pub fn poll(&mut self) -> bool {
        let Ok(d) = self.rx.try_recv() else {
            return false;
        };
        let since = match self.state {
            State::Working(t) => t,
            _ => Instant::now(),
        };
        self.calls += 1;
        self.usage.input += d.usage.input;
        self.usage.output += d.usage.output;
        self.last_cost = d.usage.cost(&self.p);
        self.last_secs = since.elapsed().as_secs_f64();
        match d.result {
            Ok((page, dir)) => {
                self.page = Some(page);
                self.last_dir = Some(dir);
                self.state = State::Ready;
                true
            }
            Err(e) => {
                self.made_from = None;
                self.state = State::Failed(format!("{e:#}"));
                false
            }
        }
    }
}

/// Coarse grey thumbnail, to tell whether the board changed since the last transcription.
fn fingerprint(img: &Rgb) -> Vec<u8> {
    const W: usize = 48;
    const H: usize = 64;
    let mut out = Vec::with_capacity(W * H);
    for ty in 0..H {
        for tx in 0..W {
            let (x0, x1) = (
                tx * img.w / W,
                ((tx + 1) * img.w / W).max(tx * img.w / W + 1),
            );
            let (y0, y1) = (
                ty * img.h / H,
                ((ty + 1) * img.h / H).max(ty * img.h / H + 1),
            );
            let mut sum = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * img.w + x) * 3;
                    sum +=
                        (img.data[i] as u64 + img.data[i + 1] as u64 + img.data[i + 2] as u64) / 3;
                }
            }
            out.push((sum / ((x1 - x0) * (y1 - y0)) as u64) as u8);
        }
    }
    out
}

/// A new stroke darkens a few thumbnail cells clearly; noise moves them all a little.
fn same_board(a: &[u8], b: &[u8]) -> bool {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.abs_diff(**y) > 12)
        .count()
        < 3
}

fn prompt(aspect: f64) -> String {
    let w = format!("{:.1}", 20.0 * aspect);
    r"This is a photo of a chalkboard (portrait, width:height = @A@), shown as dark ink on white.
Transcribe it into a LaTeX document that recreates the board as a typeset page.

- Keep the layout: put each item roughly where it is on the board.
- Mathematics in proper LaTeX math. Text as text.
- Redraw drawings (axes, arrows, lines, plots, sketches) in TikZ, faithfully in shape and position.
- Match the size of things: text and maths should be about as large relative to the page as on the board (use
  \Large or \huge where the chalk is big; \displaystyle for fractions).
- Draw every stroke that's there: count repeated lines (e.g. two dashed lines are two \draw commands). Small doodles
  (a smiley, a tick) are drawn too, in TikZ.
- Ignore sticky notes, smudges and eraser marks.
- Don't add anything that isn't on the board; don't solve or correct anything. If a word is illegible, write your
  best reading followed by a superscript ? rather than inventing a plausible word.
- Use only the packages in the frame below.

Output a complete document and nothing else (no Markdown fences), using exactly this frame:
\documentclass{article}
\usepackage[paperwidth=@W@cm,paperheight=20cm,margin=0cm]{geometry}
\usepackage{amsmath,amssymb,tikz}
\usetikzlibrary{arrows.meta,calc}
\pagestyle{empty}
\begin{document}
\noindent\begin{tikzpicture}[x=1cm,y=1cm]
\useasboundingbox (0,0) rectangle (@W@,20);
% (0,0) is the board's bottom-left, (@W@,20) its top-right; place everything with \node and \draw.
...
\end{tikzpicture}
\end{document}"
        .replace("@A@", &format!("{aspect:.2}"))
        .replace("@W@", &w)
}

fn transcribe(
    agent: &ureq::Agent,
    key: &str,
    p: &Params,
    job: &Job,
    usage: &mut Usage,
) -> Result<(Rgb, PathBuf)> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let dir = p.out_dir.join(stamp.to_string());
    std::fs::create_dir_all(&dir)?;
    let png = encode_png(&job.image)?;
    std::fs::write(dir.join("input.png"), &png)?;
    let image_url = format!("data:image/png;base64,{}", base64(&png));
    let mut input = vec![serde_json::json!({"role": "user", "content": [
        {"type": "input_text", "text": prompt(p.aspect)},
        {"type": "input_image", "image_url": image_url},
    ]})];
    let mut tex = ask(agent, key, p, &input, usage)?;
    let mut compiled = compile(&dir, &tex);
    if let Err(e) = &compiled {
        // One repair round: show the model its document and pdflatex's complaint.
        input.push(serde_json::json!({"role": "assistant", "content": tex}));
        input.push(serde_json::json!({"role": "user", "content": format!(
            "pdflatex failed with:\n{e:#}\nReply with the corrected complete document only."
        )}));
        std::fs::rename(dir.join("board.tex"), dir.join("board.failed.tex")).ok();
        tex = ask(agent, key, p, &input, usage)?;
        compiled = compile(&dir, &tex);
    }
    compiled?;
    let page = rasterise(&dir, job)?;
    Ok((page, dir))
}

/// One Responses API call; returns the output text and adds to `usage`.
fn ask(
    agent: &ureq::Agent,
    key: &str,
    p: &Params,
    input: &[serde_json::Value],
    usage: &mut Usage,
) -> Result<String> {
    let mut body = serde_json::json!({"model": p.model, "input": input});
    if !p.effort.is_empty() {
        body["reasoning"] = serde_json::json!({"effort": p.effort});
    }
    let mut resp = agent
        .post("https://api.openai.com/v1/responses")
        .header("Authorization", &format!("Bearer {key}"))
        .header("Content-Type", "application/json")
        .send(body.to_string())
        .map_err(|e| anyhow!("OpenAI request: {e}"))?;
    let status = resp.status();
    let text = resp
        .body_mut()
        .with_config()
        .limit(16 << 20)
        .read_to_string()?;
    let v: serde_json::Value = serde_json::from_str(&text).context("OpenAI reply isn't JSON")?;
    if !status.is_success() {
        bail!(
            "OpenAI {}: {}",
            status.as_u16(),
            v["error"]["message"].as_str().unwrap_or(&text)
        );
    }
    usage.input += v["usage"]["input_tokens"].as_u64().unwrap_or(0);
    usage.output += v["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let out: String = v["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|o| o["type"] == "message")
        .flat_map(|o| o["content"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["text"].as_str().map(str::to_string))
        .collect();
    let out = out.trim();
    let out = out
        .strip_prefix("```latex")
        .or_else(|| out.strip_prefix("```tex"))
        .or_else(|| out.strip_prefix("```"))
        .unwrap_or(out);
    let out = out.strip_suffix("```").unwrap_or(out).trim();
    if !out.contains("\\begin{document}") {
        bail!("the model didn't return a LaTeX document");
    }
    Ok(out.to_string())
}

/// pdflatex in `dir`; on failure, the error lines from the log.
fn compile(dir: &Path, tex: &str) -> Result<()> {
    std::fs::write(dir.join("board.tex"), tex)?;
    let mut child = Command::new("pdflatex")
        .args([
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-no-shell-escape",
            "board.tex",
        ])
        .current_dir(dir)
        // kpathsea "paranoid": no reading or writing dotfiles or files outside the work directory.
        .env("openin_any", "p")
        .env("openout_any", "p")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("running pdflatex")?;
    let started = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if started.elapsed() > Duration::from_secs(30) {
            child.kill().ok();
            child.wait().ok();
            bail!("pdflatex took over 30 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    if status.success() && dir.join("board.pdf").exists() {
        return Ok(());
    }
    let log = std::fs::read_to_string(dir.join("board.log")).unwrap_or_default();
    let errs: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with('!') || l.starts_with("l."))
        .take(6)
        .collect();
    bail!(
        "{}",
        if errs.is_empty() {
            "pdflatex failed".to_string()
        } else {
            errs.join("\n")
        }
    )
}

/// board.pdf -> the board canvas (page scaled to the board area, padded like the video).
fn rasterise(dir: &Path, job: &Job) -> Result<Rgb> {
    let (bw, bh) = job.board;
    // The page is 20 cm tall; render at the board's height in pixels.
    let dpi = ((bh as f64 / (20.0 / 2.54)).ceil() as u32).max(30);
    let ok = Command::new("pdftoppm")
        .args(["-r", &dpi.to_string(), "-singlefile", "board.pdf", "page"])
        .current_dir(dir)
        .status()?
        .success();
    if !ok {
        bail!("pdftoppm failed");
    }
    let page = read_ppm(&dir.join("page.ppm"))?;
    let (pw, ph) = (page.w as f64, page.h as f64);
    let quad = [[0.0, 0.0], [pw, 0.0], [pw, ph], [0.0, ph]];
    let map = ResampleMap::board(
        &quad,
        page.w,
        page.h,
        job.canvas.0,
        job.canvas.1,
        bw,
        bh,
        job.bg,
    )?;
    let mut out = Rgb::new(job.canvas.0, job.canvas.1);
    map.apply(&page, &mut out);
    std::fs::remove_file(dir.join("page.ppm")).ok();
    Ok(out)
}

fn read_ppm(path: &Path) -> Result<Rgb> {
    let data = std::fs::read(path)?;
    // Header: "P6", width, height, maxval, each separated by whitespace, then one whitespace byte and the pixels.
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while i < data.len() && data[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < data.len() && !data[i].is_ascii_whitespace() {
            i += 1;
        }
        if start == i {
            bail!("truncated PPM");
        }
        fields.push(std::str::from_utf8(&data[start..i])?.to_string());
    }
    if fields[0] != "P6" || fields[3] != "255" {
        bail!("unexpected PPM format {} / {}", fields[0], fields[3]);
    }
    let (w, h): (usize, usize) = (fields[1].parse()?, fields[2].parse()?);
    let px = data
        .get(i + 1..i + 1 + w * h * 3)
        .ok_or_else(|| anyhow!("truncated PPM"))?;
    Ok(Rgb {
        w,
        h,
        data: px.to_vec(),
    })
}

fn encode_png(img: &Rgb) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, img.w as u32, img.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header()?;
        w.write_image_data(&img.data)?;
    }
    Ok(buf)
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63]);
        out.push(T[(n >> 12) as usize & 63]);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63]
        } else {
            b'='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63]
        } else {
            b'='
        });
    }
    String::from_utf8(out).expect("base64 is ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn a_new_stroke_is_a_new_board_but_noise_is_not() {
        let mut a = Rgb::new(480, 640);
        a.data.fill(250);
        let mut noisy = a.clone();
        for (i, v) in noisy.data.iter_mut().enumerate() {
            *v -= (i % 7) as u8;
        }
        assert!(same_board(&fingerprint(&a), &fingerprint(&noisy)));
        let mut stroke = a.clone();
        for y in 100..400 {
            for x in 200..216 {
                let i = (y * 480 + x) * 3;
                stroke.data[i..i + 3].fill(20);
            }
        }
        assert!(!same_board(&fingerprint(&a), &fingerprint(&stroke)));
    }

    #[test]
    fn reads_ppm() {
        let dir = std::env::temp_dir().join(format!("boardvu-ppm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut ppm = b"P6\n4 2\n255\n".to_vec();
        ppm.extend(std::iter::repeat_n(200u8, 4 * 2 * 3));
        std::fs::write(dir.join("t.ppm"), &ppm).unwrap();
        let img = read_ppm(&dir.join("t.ppm")).unwrap();
        assert_eq!((img.w, img.h, img.data[0]), (4, 2, 200));
        std::fs::remove_dir_all(&dir).ok();
    }
}
