//! boardvu: a webcam pointed at my blackboard -> two v4l2loopback cameras (raw passthrough + deskewed board).

mod calib;
mod camera;
mod enhance;
mod geom;
mod ghost;
mod image;
mod keys;
mod latex;
mod light;
mod log;
mod secrets;
mod segment;
mod tui;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Deserialize;

use camera::{Camera, Focus, Sink, SinkIo};
use image::{ResampleMap, Rgb, rgb_to_yuyv};

#[derive(Parser)]
#[command(about = "Blackboard virtual camera", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Cmd {
    /// Stream to the virtual cameras (the default).
    Run,
    /// Grab one frame and write raw.png (full frame, board quad outlined) and board.png to a directory.
    Snap {
        #[arg(default_value = ".")]
        dir: PathBuf,
    },
    /// Find the board's edges and update the corners in the config (writes calib.png to check).
    Calibrate {
        /// Where calib.png goes.
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        /// How far (pixels) the board may have moved from the starting corners.
        #[arg(long, default_value_t = 120)]
        search: i32,
        /// Start from these corners instead of the config's: "x,y x,y x,y x,y" (TL TR BR BL).
        #[arg(long, value_parser = parse_corners)]
        from: Option<calib::Corners>,
        /// Manual fallback: write exactly these corners, no detection.
        #[arg(long, value_parser = parse_corners, conflicts_with = "from")]
        set: Option<calib::Corners>,
        /// Report only; don't write the config.
        #[arg(long)]
        dry_run: bool,
    },
    /// Debug: time person segmentation on live frames and save the last matte.
    #[command(hide = true)]
    SegBench {
        #[arg(default_value = ".")]
        dir: PathBuf,
        #[arg(long, default_value = "512x288", value_parser = parse_size)]
        res: (usize, usize),
        #[arg(long, default_value_t = 0.75)]
        ratio: f32,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Use a still JPEG instead of the camera.
        #[arg(long)]
        image: Option<PathBuf>,
    },
}

#[derive(Args)]
struct Common {
    /// Real webcam.
    #[arg(long, default_value = "/dev/video0")]
    cam: String,
    /// Calibration file (board corners, focus, aspect).
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/board.json"))]
    config: PathBuf,
    /// Focus override: "auto" or a focus_absolute value (default: the config's).
    #[arg(long)]
    focus: Option<String>,
    /// Pad the board into a canvas of this size (e.g. 1920x1080 for a 16:9 frame). Default: no padding, the
    /// board at the resolution the camera actually resolves it.
    #[arg(long, value_parser = parse_size)]
    size: Option<(usize, usize)>,
}

#[derive(Args)]
struct RunArgs {
    /// v4l2loopback device for the raw passthrough.
    #[arg(long, default_value = "/dev/video10")]
    raw: String,
    /// v4l2loopback device for the board.
    #[arg(long, default_value = "/dev/video11")]
    board: String,
    /// Raw passthrough size.
    #[arg(long, default_value = "1920x1080", value_parser = parse_size)]
    raw_size: (usize, usize),
    /// Don't write the raw passthrough.
    #[arg(long)]
    no_raw: bool,
    /// Debug: replace the board with frames alternating black/white, to check readers never see torn frames.
    #[arg(long, hide = true)]
    test_flash: bool,
    /// How frames are written to the virtual cameras.
    #[arg(long, value_enum, default_value_t = SinkIo::Mmap)]
    io: SinkIo,
    /// How the board looks (switch while running with the `l` key).
    #[arg(long, value_enum, default_value_t = enhance::Look::Natural)]
    look: enhance::Look,
    /// Don't watch for the board moving in the frame (a bumped camera).
    #[arg(long)]
    no_track: bool,
    /// Plain deskewed board: no segmentation, no board memory, no ghost.
    #[arg(long)]
    no_ghost: bool,
    /// How visible I am over the board in ghost mode (0 = invisible, 1 = fully live).
    #[arg(long, default_value_t = 0.25)]
    ghost_opacity: f32,
    /// Segmentation model (Robust Video Matting, MobileNetV3 ONNX).
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/models/rvm_mobilenetv3_fp32.onnx"))]
    model: PathBuf,
    /// Resolution the camera frame is downscaled to for segmentation.
    #[arg(long, default_value = "512x288", value_parser = parse_size)]
    seg_res: (usize, usize),
    /// RVM's internal downsample ratio (aim for ratio * width of roughly 256-512).
    #[arg(long, default_value_t = 0.75)]
    seg_ratio: f32,
    /// CPU threads for segmentation.
    #[arg(long, default_value_t = 2)]
    seg_threads: usize,
    /// Maximum segmentation rate. Higher follows me more closely and costs more CPU.
    #[arg(long, default_value_t = 10.0)]
    seg_fps: f32,
    /// Threads for the per-frame image work (warp, ghost, colour conversion).
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Light control config (Home Assistant). Missing file = no light control.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/light.json"))]
    light: PathBuf,
    /// Leave the board light alone.
    #[arg(long)]
    no_light: bool,
    /// Plain log lines instead of the full-screen status display.
    #[arg(long)]
    no_tui: bool,
    /// Turn off the LaTeX view (`t`), which sends the board to OpenAI.
    #[arg(long)]
    no_latex: bool,
    /// OpenAI model for the LaTeX view.
    #[arg(long, default_value = "gpt-6-luna")]
    latex_model: String,
    /// Reasoning effort for the LaTeX view (none, low, medium, high; empty = the model's default). Higher reads
    /// messy writing better but takes longer: low ~15 s, medium ~40 s.
    #[arg(long, default_value = "low")]
    latex_effort: String,
    /// Model price, US dollars per million input tokens (for the cost shown).
    #[arg(long, default_value_t = 0.10)]
    latex_price_in: f64,
    /// Model price, US dollars per million output tokens (reasoning included).
    #[arg(long, default_value_t = 0.50)]
    latex_price_out: f64,
}

#[derive(Deserialize)]
struct Config {
    /// TL, TR, BR, BL in full-resolution camera pixels, on the edges of the writing surface.
    corners: [[f64; 2]; 4],
    /// Camera frame size the corners refer to.
    frame: [usize; 2],
    focus: Option<i64>,
    /// Real board width / height. Estimated from the corners if absent (perspective makes that a guess).
    aspect: Option<f64>,
    /// Degrees to turn the camera picture clockwise to make it upright (camera mounted on its side): 0, 90, 180,
    /// 270. Corners are always listed in the board's own order (its top-left first), whatever the rotation.
    #[serde(default)]
    rotate: u32,
}

const BG: [u8; 3] = [24, 24, 24];

fn parse_size(s: &str) -> Result<(usize, usize), String> {
    let (w, h) = s.split_once('x').ok_or("expected WxH")?;
    let (w, h): (usize, usize) = (
        w.parse().map_err(|_| "bad width")?,
        h.parse().map_err(|_| "bad height")?,
    );
    if w < 2 || h < 2 || w % 2 != 0 || h % 2 != 0 {
        return Err("width and height must be even (4:2:x chroma pairs pixels and rows)".into());
    }
    Ok((w, h))
}

fn parse_corners(s: &str) -> Result<calib::Corners, String> {
    let pts: Vec<[f64; 2]> = s
        .split_whitespace()
        .map(|p| {
            let (x, y) = p.split_once(',').ok_or("expected x,y")?;
            Ok([
                x.parse().map_err(|_| "bad x")?,
                y.parse().map_err(|_| "bad y")?,
            ])
        })
        .collect::<Result<_, &str>>()?;
    pts.try_into()
        .map_err(|_| "expected 4 corners: \"x,y x,y x,y x,y\" (TL TR BR BL)".into())
}

/// Replace the corners in the config file, keeping everything else.
fn save_corners(path: &Path, corners: &calib::Corners) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let mut v: serde_json::Value = serde_json::from_str(&text)?;
    let round = |x: f64| (x * 10.0).round() / 10.0;
    v["corners"] = serde_json::json!(corners.map(|[x, y]| [round(x), round(y)]));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&v)? + "\n")?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn load_config(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: Config =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        [0, 90, 180, 270].contains(&cfg.rotate),
        "rotate must be 0, 90, 180 or 270 (got {})",
        cfg.rotate
    );
    Ok(cfg)
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn aspect(cfg: &Config) -> f64 {
    let c = &cfg.corners;
    cfg.aspect.unwrap_or_else(|| {
        (dist(c[0], c[1]) + dist(c[3], c[2])) / (dist(c[0], c[3]) + dist(c[1], c[2]))
    })
}

/// The board at the resolution the camera resolves it: as tall as its longer (nearer) vertical edge
/// in camera pixels, so nothing is downsampled and nothing is upscaled beyond that. Both sides even: YUYV pairs pixels, and
/// readers converting to 4:2:0 pair rows (an odd last row shows up as a green line).
fn native_dims(cfg: &Config) -> (usize, usize) {
    let c = &cfg.corners;
    let h = (dist(c[0], c[3]).max(dist(c[1], c[2])).round() as usize) & !1;
    (((h as f64 * aspect(cfg)).round() as usize) & !1, h)
}

/// Board size inside a padded canvas: as large as fits, keeping the aspect, both sides even.
fn fit_dims(cfg: &Config, (cw, ch): (usize, usize)) -> (usize, usize) {
    let aspect = aspect(cfg);
    let (mut w, mut h) = ((ch as f64 * aspect).round() as usize, ch);
    if w > cw {
        (w, h) = (cw, (cw as f64 / aspect).round() as usize);
    }
    (w & !1, h & !1)
}

fn focus_of(common: &Common, cfg: &Config) -> Result<Focus> {
    Ok(match common.focus.as_deref() {
        Some("auto") => Focus::Auto,
        Some(v) => Focus::Fixed(v.parse().context("--focus must be 'auto' or an integer")?),
        None => cfg.focus.map_or(Focus::Auto, Focus::Fixed),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.run.threads)
        .thread_name(|i| format!("img{i}"))
        .build_global()?;
    let cfg = load_config(&cli.common.config)?;
    let [fw, fh] = cfg.frame;
    let ((cw, ch), (bw, bh)) = match cli.common.size {
        Some(canvas) => (canvas, fit_dims(&cfg, canvas)),
        None => (native_dims(&cfg), native_dims(&cfg)),
    };
    let board_map = ResampleMap::board(&cfg.corners, fw, fh, cw, ch, bw, bh, BG)?;
    if let Some(Cmd::SegBench {
        dir,
        res,
        ratio,
        threads,
        image,
    }) = &cli.cmd
    {
        let mut source: Box<dyn FnMut(&mut Rgb) -> Result<()>> = match image {
            Some(path) => {
                let img = camera::decode_jpeg(&std::fs::read(path)?)?;
                Box::new(move |out: &mut Rgb| {
                    *out = Rgb {
                        w: img.w,
                        h: img.h,
                        data: img.data.clone(),
                    };
                    Ok(())
                })
            }
            None => {
                let mut cam = Camera::open(&cli.common.cam, fw, fh, focus_of(&cli.common, &cfg)?)?;
                Box::new(move |out: &mut Rgb| cam.next_into(out))
            }
        };
        return seg_bench(&mut *source, (fw, fh), dir, *res, *ratio, *threads);
    }
    if let Some(Cmd::Calibrate {
        dir,
        search,
        from,
        set,
        dry_run,
    }) = &cli.cmd
    {
        return calibrate_cmd(&cli.common, &cfg, dir, *search, *from, *set, *dry_run);
    }
    let mut cam = Camera::open(&cli.common.cam, fw, fh, focus_of(&cli.common, &cfg)?)?;
    eprintln!(
        "camera {} {fw}x{fh}; board {bw}x{bh} in a {cw}x{ch} canvas",
        cli.common.cam
    );

    match cli.cmd.unwrap_or(Cmd::Run) {
        Cmd::Snap { dir } => snap(&mut cam, &board_map, &cfg, &dir),
        Cmd::SegBench { .. } | Cmd::Calibrate { .. } => unreachable!("handled above"),
        Cmd::Run => run(
            &mut cam,
            board_map,
            &cfg,
            &cli.common.config,
            ((cw, ch), (bw, bh)),
            &cli.run,
        ),
    }
}

fn snap(cam: &mut Camera, board_map: &ResampleMap, cfg: &Config, dir: &Path) -> Result<()> {
    let mut frame = Rgb::new(cam.w, cam.h);
    // Let auto-exposure and the locked focus settle.
    for _ in 0..30 {
        cam.next_into(&mut frame)?;
    }
    // Average 16 frames: a still is then about as clean as ghost mode's denoised board memory.
    let mut board = Rgb::new(board_map.out_w, board_map.out_h);
    let mut sum = vec![0u32; board.data.len()];
    const N: u32 = 16;
    for _ in 0..N {
        cam.next_into(&mut frame)?;
        board_map.apply(&frame, &mut board);
        for (s, &v) in sum.iter_mut().zip(&board.data) {
            *s += v as u32;
        }
    }
    for (b, s) in board.data.iter_mut().zip(&sum) {
        *b = ((s + N / 2) / N) as u8;
    }
    std::fs::create_dir_all(dir)?;
    write_png(&dir.join("board.png"), &board)?;
    let mut enhanced = Rgb::new(board.w, board.h);
    for (look, name) in [
        (enhance::Look::Natural, "natural"),
        (enhance::Look::Paper, "paper"),
    ] {
        enhance::Enhancer::new(look, board.w, board.h).apply(&board, &mut enhanced);
        write_png(&dir.join(format!("board_{name}.png")), &enhanced)?;
    }
    draw_quad(&mut frame, &cfg.corners, [255, 0, 0]);
    write_png(&dir.join("raw.png"), &frame)?;
    eprintln!(
        "wrote raw.png, board.png, board_natural.png, board_paper.png to {}",
        dir.display()
    );
    Ok(())
}

fn calibrate_cmd(
    common: &Common,
    cfg: &Config,
    dir: &Path,
    search: i32,
    from: Option<calib::Corners>,
    set: Option<calib::Corners>,
    dry_run: bool,
) -> Result<()> {
    let [fw, fh] = cfg.frame;
    let mut cam = Camera::open(&common.cam, fw, fh, focus_of(common, cfg)?)?;
    let mut frame = Rgb::new(fw, fh);
    for _ in 0..30 {
        cam.next_into(&mut frame)?; // let exposure settle
    }
    // Average 8 frames to cut noise.
    let mut sum = vec![0u32; frame.data.len()];
    for _ in 0..8 {
        cam.next_into(&mut frame)?;
        for (s, &v) in sum.iter_mut().zip(&frame.data) {
            *s += v as u32;
        }
    }
    for (b, s) in frame.data.iter_mut().zip(&sum) {
        *b = ((s + 4) / 8) as u8;
    }
    let corners = match set {
        Some(c) => {
            eprintln!("using the given corners as-is");
            c
        }
        None => {
            let start = from.unwrap_or(cfg.corners);
            let fit = calib::calibrate(&frame, &start, search)
                .context("couldn't find all four board edges near the starting corners (try --search or --from)")?;
            eprintln!(
                "found the board {:.1} px from the starting corners; edge support (top, right, bottom, left) {:?}",
                calib::max_shift(&fit.corners, &start),
                fit.support.map(|s| (s * 100.0).round() / 100.0)
            );
            let problem = calib::check_plausible(&fit.corners, (fw, fh), &start)
                .err()
                .or_else(|| {
                    (fit.min_support() < 0.35).then(|| "an edge is poorly supported".to_string())
                });
            if let Some(why) = problem {
                std::fs::create_dir_all(dir)?;
                draw_quad(&mut frame, &start, [255, 160, 0]);
                draw_quad(&mut frame, &fit.corners, [255, 0, 0]);
                write_png(&dir.join("calib.png"), &frame)?;
                anyhow::bail!(
                    "not saving: {why}. See {}/calib.png (red: rejected fit, orange: start). Make sure the whole \
                     board is in view, or start nearer with --from, or set corners by hand with --set.",
                    dir.display()
                );
            }
            fit.corners
        }
    };
    let fmt: Vec<String> = corners
        .iter()
        .map(|[x, y]| format!("{x:.1},{y:.1}"))
        .collect();
    eprintln!("corners (TL TR BR BL): {}", fmt.join(" "));
    std::fs::create_dir_all(dir)?;
    draw_quad(&mut frame, &cfg.corners, [255, 160, 0]);
    draw_quad(&mut frame, &corners, [0, 255, 0]);
    write_png(&dir.join("calib.png"), &frame)?;
    eprintln!(
        "wrote {}/calib.png (green: new, orange: previous)",
        dir.display()
    );
    if dry_run {
        eprintln!("dry run: {} not changed", common.config.display());
    } else {
        save_corners(&common.config, &corners)?;
        eprintln!("updated {}", common.config.display());
    }
    Ok(())
}

fn seg_bench(
    next: &mut dyn FnMut(&mut Rgb) -> Result<()>,
    (fw, fh): (usize, usize),
    dir: &Path,
    (w, h): (usize, usize),
    ratio: f32,
    threads: usize,
) -> Result<()> {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/rvm_mobilenetv3_fp32.onnx");
    let mut m = segment::Matting::new(&model, w, h, ratio, threads)?;
    let down = ResampleMap::scale(fw, fh, w, h);
    let (mut frame, mut small) = (Rgb::new(fw, fh), Rgb::new(w, h));
    let mut alpha = vec![0f32; w * h];
    let mut times = vec![];
    for i in 0..60 {
        next(&mut frame)?;
        let t = Instant::now();
        down.apply(&frame, &mut small);
        m.run(&small, &mut alpha)?;
        if i >= 10 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "{w}x{h} ratio {ratio} threads {threads}: median {:.1} ms, p90 {:.1} ms",
        times[times.len() / 2],
        times[times.len() * 9 / 10]
    );
    std::fs::create_dir_all(dir)?;
    let mut vis = Rgb::new(w, h);
    for (i, a) in alpha.iter().enumerate() {
        let px = &small.data[i * 3..i * 3 + 3];
        // Person tinted red, background dimmed.
        vis.data[i * 3] = (px[0] as f32 * (0.4 + 0.6 * a) + 120.0 * a).min(255.0) as u8;
        vis.data[i * 3 + 1] = (px[1] as f32 * (0.4 + 0.2 * a)) as u8;
        vis.data[i * 3 + 2] = (px[2] as f32 * (0.4 + 0.2 * a)) as u8;
    }
    write_png(&dir.join("matte.png"), &vis)?;
    eprintln!("wrote {}/matte.png", dir.display());
    Ok(())
}

/// Start a LaTeX transcription of `board` (the board memory, or the live board without ghost mode).
fn request_latex(
    t: &mut latex::Transcriber,
    board: &Rgb,
    canvas: (usize, usize),
    dims: (usize, usize),
    force: bool,
) {
    let mut paper = Rgb::new(board.w, board.h);
    // A fresh enhancer measures the lighting on its first frame.
    enhance::Enhancer::new(enhance::Look::Paper, board.w, board.h).apply(board, &mut paper);
    if t.request(&paper, canvas, dims, BG, force) {
        say!("LaTeX: transcribing with {}...", t.p.model);
    } else if t.busy() {
        say!("LaTeX: already transcribing");
    } else {
        say!("LaTeX: board unchanged, showing the last page (r to redo)");
    }
}

/// Segmentation-frame -> board-canvas warp for the ghost's matte, and segmentation pixels per board pixel.
/// Segmentation runs on the upright picture, so the corners are turned too.
fn mask_map_for(
    corners: &calib::Corners,
    cam: (usize, usize),
    rot: u32,
    (sw, sh): (usize, usize),
    ((cw, ch), (bw, bh)): ((usize, usize), (usize, usize)),
) -> Result<(ResampleMap, f32)> {
    let (uw, uh) = image::upright_size(cam, rot);
    let (kx, ky) = (sw as f64 / uw as f64, sh as f64 / uh as f64);
    let c = corners.map(|p| {
        let [x, y] = image::to_upright(p, cam, rot);
        [x * kx, y * ky]
    });
    let map = ResampleMap::board(&c, sw, sh, cw, ch, bw, bh, [0; 3])?;
    // Along the board's longer (nearer) vertical edge.
    let scale = (dist(c[0], c[3]).max(dist(c[1], c[2])) / bh as f64) as f32;
    Ok((map, scale))
}

fn run(
    cam: &mut Camera,
    mut board_map: ResampleMap,
    cfg: &Config,
    config_path: &Path,
    dims: ((usize, usize), (usize, usize)),
    args: &RunArgs,
) -> Result<()> {
    let ((cw, ch), (bw, bh)) = dims;
    // Light first, so the camera's auto-exposure settles on the lit board. Dropped (light off) on any exit.
    let _light = if args.no_light {
        None
    } else {
        light::Light::from_config(&args.light)?.map(light::LightGuard::new)
    };
    // Sizes are given landscape; a camera on its side gives portrait upright pictures.
    let seg_res = image::upright_size(args.seg_res, cfg.rotate);
    let mut ghost = if args.no_ghost {
        None
    } else {
        let (sw, sh) = seg_res;
        let seg = ghost::Segmenter::start(
            &args.model,
            (cam.w, cam.h),
            cfg.rotate,
            (sw, sh),
            args.seg_ratio,
            args.seg_threads,
            args.seg_fps,
        )?;
        let (mask_map, seg_scale) =
            mask_map_for(&cfg.corners, (cam.w, cam.h), cfg.rotate, (sw, sh), dims)?;
        let params = ghost::GhostParams {
            opacity: args.ghost_opacity,
            ..Default::default()
        };
        say!(
            "ghost mode: segmenting at {sw}x{sh}, opacity {}",
            args.ghost_opacity
        );
        Some((
            seg,
            ghost::Ghost::new(params, cw, ch, mask_map, sw, sh, seg_scale),
            Rgb::new(cw, ch),
        ))
    };
    let mut board_sink = Sink::open(&args.board, board_map.out_w, board_map.out_h, args.io)?;
    let mut raw_out = if args.no_raw {
        None
    } else {
        let (w, h) = image::upright_size(args.raw_size, cfg.rotate);
        Some((
            Sink::open(&args.raw, w, h, args.io)?,
            ResampleMap::upright(cam.w, cam.h, w, h, cfg.rotate),
            Rgb::new(w, h),
            camera::watch_openers(&args.raw),
        ))
    };
    if let Some((s, ..)) = &raw_out {
        say!("raw   -> {} ({}x{})", s.path, s.w, s.h);
    }
    say!(
        "board -> {} ({}x{})",
        board_sink.path,
        board_sink.w,
        board_sink.h
    );

    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    let mut tracker = (!args.no_track).then(|| calib::Tracker::new(cfg.corners));
    let mut frame = Rgb::new(cam.w, cam.h);
    let mut board = Rgb::new(board_map.out_w, board_map.out_h);
    let mut enhancer = enhance::Enhancer::new(args.look, board.w, board.h);
    let mut enhanced = Rgb::new(board.w, board.h);
    let mut last_enhance = Instant::now() - Duration::from_secs(1);
    let mut transcriber = if args.no_latex {
        None
    } else {
        let p = latex::Params {
            model: args.latex_model.clone(),
            effort: args.latex_effort.clone(),
            price_in: args.latex_price_in,
            price_out: args.latex_price_out,
            out_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out/latex"),
            aspect: bw as f64 / bh as f64,
        };
        match latex::Transcriber::start(p) {
            Ok(t) => Some(t),
            Err(e) => {
                say!("LaTeX view off: {e:#}");
                None
            }
        }
    };
    let mut latex_view = false;
    // Set by `t`/`r`: transcribe the board once this frame's board memory is ready (true = even if unchanged).
    let mut want_latex: Option<bool> = None;
    let board_watched = camera::watch_openers(&args.board);
    let started = Instant::now();
    // Full-screen status in a terminal; plain log lines otherwise (then keys come through `Keys`).
    let mut tui = (!args.no_tui).then(tui::Tui::start).flatten();
    let keys = tui.is_none().then(keys::Keys::start);
    if tui.is_none() {
        say!(
            "look: {:?} | keys: l = cycle look (plain/natural/paper), g = toggle ghost, t = LaTeX view, \
             r = re-transcribe",
            args.look
        );
    }
    let mut perf = tui::Perf::default();
    let mut cpu_last = (cpu_seconds(), Instant::now());
    let mut last_print = Instant::now();
    let mut board_yuyv = vec![0u8; board.w * board.h * 2];
    let mut raw_yuyv = raw_out
        .as_ref()
        .map(|(_, _, r, _)| vec![0u8; r.w * r.h * 2])
        .unwrap_or_default();

    let (mut n, mut t_wait, mut t_dec, mut t_work) =
        (0u32, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut last_report = Instant::now();
    let mut errors = 0;
    while !stop.load(Ordering::SeqCst) {
        let t0 = Instant::now();
        match cam.next_into(&mut frame) {
            Ok(()) => errors = 0,
            Err(_) if stop.load(Ordering::SeqCst) => break, // Ctrl-C interrupted the read
            Err(e) => {
                // A corrupt MJPEG frame now and then is normal; a run of them means the camera has gone.
                errors += 1;
                if errors > 30 {
                    return Err(e.context("camera keeps failing"));
                }
                say!("skipping frame: {e:#}");
                continue;
            }
        }
        let t1 = Instant::now();
        if let Some(c) = tracker.as_mut().and_then(|t| t.check(&frame)) {
            say!(
                "board moved in the frame: re-aligned (corners {})",
                c.iter()
                    .map(|[x, y]| format!("{x:.1},{y:.1}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            board_map = ResampleMap::board(&c, cam.w, cam.h, cw, ch, bw, bh, BG)?;
            if let Some((_, g, _)) = &mut ghost {
                let (map, scale) = mask_map_for(&c, (cam.w, cam.h), cfg.rotate, seg_res, dims)?;
                g.recalibrate(map, scale);
            }
            if let Err(e) = save_corners(config_path, &c) {
                say!("warning: couldn't save the new corners: {e:#}");
            }
        }
        board_map.apply(&frame, &mut board);
        if args.test_flash {
            board.data.fill(if n % 2 == 0 { 0 } else { 255 });
        }
        let pressed: Vec<u8> = match (&mut tui, &keys) {
            (Some(t), _) => t.keys(),
            (None, Some(k)) => k.rx.try_iter().collect(),
            (None, None) => Vec::new(),
        };
        for k in pressed {
            match k {
                b'q' => stop.store(true, Ordering::SeqCst),
                b'l' => {
                    enhancer.look = enhancer.look.next();
                    last_enhance = Instant::now() - Duration::from_secs(1);
                    say!("look: {:?}", enhancer.look);
                }
                b't' if transcriber.is_some() => {
                    latex_view = !latex_view;
                    if latex_view {
                        want_latex = Some(false);
                    }
                    say!("LaTeX view {}", if latex_view { "on" } else { "off" });
                }
                b'r' if transcriber.is_some() => {
                    latex_view = true;
                    want_latex = Some(true);
                }
                b'g' => {
                    if let Some((_, g, _)) = &mut ghost {
                        let o = if g.opacity() > 0.0 {
                            0.0
                        } else {
                            args.ghost_opacity.max(0.05)
                        };
                        g.set_opacity(o);
                        say!("ghost opacity: {o}");
                    }
                }
                _ => {}
            }
        }
        if let Some(t) = &mut transcriber {
            if t.poll() {
                say!(
                    "LaTeX: typeset in {:.0} s, ${:.4} (session: {} calls, ${:.4}) -> {}",
                    t.last_secs,
                    t.last_cost,
                    t.calls,
                    t.total_cost(),
                    t.last_dir
                        .as_ref()
                        .map(|d| d.display().to_string())
                        .unwrap_or_default()
                );
            } else if let latex::State::Failed(e) = &t.state {
                say!("LaTeX failed after {:.0} s: {e}", t.last_secs);
                t.state = latex::State::Idle;
                latex_view = t.page.is_some() && latex_view;
            }
        }
        // Board memory (or the live board) -> chalk enhancement -> me drawn over it.
        let shown = match &mut ghost {
            Some((seg, g, out)) => {
                seg.submit(&frame);
                g.step(&board, seg, out);
                if let (Some(force), Some(t)) = (want_latex.take(), &mut transcriber) {
                    request_latex(t, out, (cw, ch), (bw, bh), force);
                }
                let page = transcriber
                    .as_ref()
                    .filter(|_| latex_view)
                    .and_then(|t| t.page.as_ref());
                // The memory changes slowly (half-second fades), so ~10 Hz enhancement looks the same and
                // costs a third as much at 30 fps; I'm drawn over it every frame.
                // (Not needed at all while the LaTeX page is shown.)
                if page.is_none()
                    && (enhancer.look == enhance::Look::Plain
                        || last_enhance.elapsed() >= Duration::from_millis(90))
                {
                    enhancer.apply(out, &mut enhanced);
                    last_enhance = Instant::now();
                }
                g.overlay(page.unwrap_or(&enhanced), &board, out);
                &*out
            }
            None => {
                if let (Some(force), Some(t)) = (want_latex.take(), &mut transcriber) {
                    request_latex(t, &board, (cw, ch), (bw, bh), force);
                }
                let page = transcriber
                    .as_ref()
                    .filter(|_| latex_view)
                    .and_then(|t| t.page.as_ref());
                match page {
                    Some(p) => p,
                    None => {
                        enhancer.apply(&board, &mut enhanced);
                        &enhanced
                    }
                }
            }
        };
        rgb_to_yuyv(shown, &mut board_yuyv);
        board_sink.write(&board_yuyv)?;
        // The raw feed is ~0.7 of a core at 1080p: only produce it while something is reading it.
        if let Some((sink, map, img, watched)) = &mut raw_out
            && watched.load(Ordering::Relaxed)
        {
            map.apply(&frame, img);
            rgb_to_yuyv(img, &mut raw_yuyv);
            sink.write(&raw_yuyv)?;
        }
        if let Some(t) = &mut tui {
            fn feed<'a>(sink: &'a Sink, watched: &AtomicBool) -> tui::Feed<'a> {
                tui::Feed {
                    path: &sink.path,
                    size: (sink.w, sink.h),
                    watched: watched.load(Ordering::Relaxed),
                }
            }
            t.draw(&tui::Status {
                camera: (&cam.path, cam.w, cam.h),
                board: feed(&board_sink, &board_watched),
                raw: raw_out.as_ref().map(|(s, _, _, w)| feed(s, w)),
                look: enhancer.look,
                ghost: ghost.as_ref().map(|(_, g, _)| g.opacity()),
                latex: transcriber.as_ref(),
                latex_view,
                tracker: tracker.as_ref(),
                perf,
                started,
            });
        }
        t_dec += cam.last_decode;
        t_wait += t1 - t0 - cam.last_decode;
        t_work += t1.elapsed();
        n += 1;
        if last_report.elapsed() >= Duration::from_secs(2) {
            let secs = last_report.elapsed().as_secs_f64();
            let cpu_now = (cpu_seconds(), Instant::now());
            perf = tui::Perf {
                fps: n as f64 / secs,
                wait_ms: t_wait.as_secs_f64() * 1e3 / n as f64,
                decode_ms: t_dec.as_secs_f64() * 1e3 / n as f64,
                work_ms: t_work.as_secs_f64() * 1e3 / n as f64,
                cpu: (cpu_now.0 - cpu_last.0) / (cpu_now.1 - cpu_last.1).as_secs_f64(),
            };
            cpu_last = cpu_now;
            if tui.is_none() && last_print.elapsed() >= Duration::from_secs(10) {
                last_print = Instant::now();
                say!(
                    "{:.1} fps, {:.1} cores | per frame: waiting for camera {:.1} ms, decode {:.1} ms, \
                     warp+ghost+write {:.1} ms",
                    perf.fps,
                    perf.cpu,
                    perf.wait_ms,
                    perf.decode_ms,
                    perf.work_ms
                );
            }
            (n, t_wait, t_dec, t_work, last_report) = (
                0,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Instant::now(),
            );
        }
    }
    drop(tui);
    say!("stopped");
    Ok(())
}

/// CPU time this process has used (user + system), in seconds.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the ")" that ends the command name; utime and stime are fields 14 and 15 overall.
    let rest = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    (ticks(11) + ticks(12)) / hz.max(1.0)
}

fn draw_quad(img: &mut Rgb, c: &[[f64; 2]; 4], col: [u8; 3]) {
    for i in 0..4 {
        let (a, b) = (c[i], c[(i + 1) % 4]);
        let steps = dist(a, b).ceil() as usize * 2;
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (x as usize + dx, y as usize + dy);
                if x < img.w && y < img.h {
                    let i = (y * img.w + x) * 3;
                    img.data[i..i + 3].copy_from_slice(&col);
                }
            }
        }
    }
}

fn write_png(path: &Path, img: &Rgb) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header()?;
    w.write_image_data(&img.data)?;
    Ok(())
}
