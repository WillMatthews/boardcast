//! boardvu: a webcam pointed at my blackboard -> two v4l2loopback cameras (raw passthrough + deskewed board).

mod camera;
mod geom;
mod ghost;
mod image;
mod light;
mod segment;

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

fn load_config(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
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
    let mut cam = Camera::open(&cli.common.cam, fw, fh, focus_of(&cli.common, &cfg)?)?;
    eprintln!(
        "camera {} {fw}x{fh}; board {bw}x{bh} in a {cw}x{ch} canvas",
        cli.common.cam
    );

    match cli.cmd.unwrap_or(Cmd::Run) {
        Cmd::Snap { dir } => snap(&mut cam, &board_map, &cfg, &dir),
        Cmd::SegBench { .. } => unreachable!("handled above"),
        Cmd::Run => run(&mut cam, &board_map, &cfg, ((cw, ch), (bw, bh)), &cli.run),
    }
}

fn snap(cam: &mut Camera, board_map: &ResampleMap, cfg: &Config, dir: &Path) -> Result<()> {
    let mut frame = Rgb::new(cam.w, cam.h);
    // Let auto-exposure and the locked focus settle.
    for _ in 0..30 {
        cam.next_into(&mut frame)?;
    }
    let mut board = Rgb::new(board_map.out_w, board_map.out_h);
    board_map.apply(&frame, &mut board);
    std::fs::create_dir_all(dir)?;
    write_png(&dir.join("board.png"), &board)?;
    draw_quad(&mut frame, &cfg.corners, [255, 0, 0]);
    write_png(&dir.join("raw.png"), &frame)?;
    eprintln!("wrote {0}/raw.png and {0}/board.png", dir.display());
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

fn run(
    cam: &mut Camera,
    board_map: &ResampleMap,
    cfg: &Config,
    ((cw, ch), (bw, bh)): ((usize, usize), (usize, usize)),
    args: &RunArgs,
) -> Result<()> {
    // Light first, so the camera's auto-exposure settles on the lit board. Dropped (light off) on any exit.
    let _light = if args.no_light {
        None
    } else {
        light::Light::from_config(&args.light)?.map(light::LightGuard::new)
    };
    let mut ghost = if args.no_ghost {
        None
    } else {
        let (sw, sh) = args.seg_res;
        let seg = ghost::Segmenter::start(
            &args.model,
            (cam.w, cam.h),
            (sw, sh),
            args.seg_ratio,
            args.seg_threads,
            args.seg_fps,
        )?;
        let (kx, ky) = (sw as f64 / cam.w as f64, sh as f64 / cam.h as f64);
        let corners = cfg.corners.map(|[x, y]| [x * kx, y * ky]);
        let mask_map = ResampleMap::board(&corners, sw, sh, cw, ch, bw, bh, [0; 3])?;
        // Segmentation pixels per board pixel, along the board's longer (nearer) vertical edge.
        let seg_scale =
            (dist(corners[0], corners[3]).max(dist(corners[1], corners[2])) / bh as f64) as f32;
        let params = ghost::GhostParams {
            opacity: args.ghost_opacity,
            ..Default::default()
        };
        eprintln!(
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
        let (w, h) = args.raw_size;
        Some((
            Sink::open(&args.raw, w, h, args.io)?,
            ResampleMap::scale(cam.w, cam.h, w, h),
            Rgb::new(w, h),
            camera::watch_openers(&args.raw),
        ))
    };
    if let Some((s, ..)) = &raw_out {
        eprintln!("raw   -> {} ({}x{})", s.path, s.w, s.h);
    }
    eprintln!(
        "board -> {} ({}x{})",
        board_sink.path, board_sink.w, board_sink.h
    );

    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    let mut frame = Rgb::new(cam.w, cam.h);
    let mut board = Rgb::new(board_map.out_w, board_map.out_h);
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
            Err(e) => {
                // A corrupt MJPEG frame now and then is normal; a run of them means the camera has gone.
                errors += 1;
                if errors > 30 {
                    return Err(e.context("camera keeps failing"));
                }
                eprintln!("skipping frame: {e:#}");
                continue;
            }
        }
        let t1 = Instant::now();
        board_map.apply(&frame, &mut board);
        if args.test_flash {
            board.data.fill(if n % 2 == 0 { 0 } else { 255 });
        }
        let shown = match &mut ghost {
            Some((seg, g, out)) => {
                seg.submit(&frame);
                g.step(&board, seg, out);
                &*out
            }
            None => &board,
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
        t_dec += cam.last_decode;
        t_wait += t1 - t0 - cam.last_decode;
        t_work += t1.elapsed();
        n += 1;
        if last_report.elapsed() >= Duration::from_secs(10) {
            let secs = last_report.elapsed().as_secs_f64();
            eprintln!(
                "{:.1} fps | per frame: waiting for camera {:.1} ms, decode {:.1} ms, warp+ghost+write {:.1} ms",
                n as f64 / secs,
                t_wait.as_secs_f64() * 1e3 / n as f64,
                t_dec.as_secs_f64() * 1e3 / n as f64,
                t_work.as_secs_f64() * 1e3 / n as f64
            );
            (n, t_wait, t_dec, t_work, last_report) = (
                0,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Instant::now(),
            );
        }
    }
    eprintln!("stopped");
    Ok(())
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
