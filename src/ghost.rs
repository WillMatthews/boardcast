//! Ghost mode: remember the board where I'm not standing, and draw me faintly over it.
//!
//! - A worker thread runs person segmentation on a downscaled copy of each camera frame and publishes the
//!   newest matte; the video loop never waits for it.
//! - The matte is warped into board space, grown (dilated) to cover mask error and lag, and held for a few
//!   frames after it moves on, so the memory never learns me.
//! - Board memory is a per-pixel running average, updated only where I'm not. Small differences update slowly
//!   once a pixel is still (temporal denoise of faint chalk); large ones (new chalk, erasing) only after the
//!   pixel has been still for ~1/3 s, then quickly, so they arrive in about half a second. The stillness
//!   requirement is what keeps a fast-moving hand that the laggy matte missed out of the memory.
//! - Cast shadows: while someone is in the matte, a pixel that has gone clearly darker than the bare board around
//!   it (`shadow_dark` of a coarse board-brightness estimate) is held, not learned. My shadow falls outside the
//!   matte and keeps as still as I do, so the stillness test alone lets it in. Erasing still works: wiped chalk
//!   goes back to about the board's own brightness, not well below it. Shadowed chalk can stay brighter than
//!   bare board, so any darkening within `shadow_reach` of such a shadow pixel is held too: chalk strokes are thin,
//!   so a stroke in a shadow always has shadowed board beside it. The threshold follows the room light: it is
//!   scaled by how much the whole board has brightened or dimmed since the memory learned it (a median over the
//!   board, which a shadow covering part of it doesn't move), so dimming the lights isn't taken for a shadow.
//! - Output: the memory (optionally chalk-enhanced by the caller), then `overlay` blends the live frame in at
//!   `opacity` where the matte says person.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::Result;
use rayon::prelude::*;

use crate::image::{ROWS_PER_TASK, ResampleMap, Rgb};
use crate::segment::Matting;

/// What `g` steps through: hidden, faint, the default ghost, half, and opaque (me fully drawn, no ghosting).
pub const OPACITY_STEPS: [f32; 5] = [0.0, 0.1, 0.25, 0.5, 1.0];

/// The next step after `current` (wrapping to hidden after opaque; a custom --ghost-opacity joins at the next step up).
pub fn next_opacity(current: f32) -> f32 {
    OPACITY_STEPS
        .iter()
        .copied()
        .find(|&o| o > current + 1e-3)
        .unwrap_or(0.0)
}

pub fn describe(opacity: f32) -> String {
    if opacity <= 0.0 {
        "hidden".into()
    } else if opacity >= 1.0 {
        "opaque (no ghosting)".into()
    } else {
        format!("ghost {opacity:.2}")
    }
}

/// What the board memory did at a pixel on the last frame, for the debug view (`d`).
pub mod debug {
    /// Matches the memory (within noise); denoised slowly.
    pub const BOARD: u8 = 0;
    /// A small change (below `ink_diff`, above `CREEP_DIFF`): faint chalk or a soft shadow edge, fading in at
    /// the slow rate.
    pub const CREEPING: u8 = 1;
    /// A big change waiting for the pixel to keep still for `settle` frames.
    pub const WAITING: u8 = 2;
    /// A big change going into memory at the fast rate.
    pub const ACCEPTING: u8 = 3;
    /// Frozen: inside the grown matte, or its hold after I've moved on.
    pub const HELD: u8 = 4;
    /// Frozen: me (the undilated matte).
    pub const PERSON: u8 = 5;
    /// Frozen: clearly darker than the bare board while someone is in view, i.e. a cast shadow.
    pub const SHADOW: u8 = 6;
    /// Colour difference from memory above which a small change counts as creeping rather than noise.
    pub const CREEP_DIFF: f32 = 8.0;
    pub const NAMES: [&str; 7] = [
        "board",
        "creeping",
        "waiting",
        "accepting",
        "held",
        "person",
        "shadow",
    ];
}

pub struct GhostParams {
    /// How visible I am over the board (0 = invisible, 1 = fully live).
    pub opacity: f32,
    /// Mask dilation radius, as a fraction of board height.
    pub grow: f32,
    /// Per-frame decay of the "recently a person here" hold (lower = memory resumes sooner).
    pub hold_decay: f32,
    /// Memory update rate for small differences (denoise) and large stable ones (new chalk).
    pub rate_slow: f32,
    pub rate_fast: f32,
    /// Colour difference (largest channel, 0-255) from memory above which a change counts as real ink, not noise.
    pub ink_diff: f32,
    /// Colour change since the previous frame above which a pixel counts as moving.
    pub motion: f32,
    /// Frames a pixel must stay still before a big change (new chalk) is accepted into memory. Keeps a moving
    /// hand that the (laggy) matte missed out of the memory: no pixel sees a moving hand for this long.
    pub settle: u8,
    /// Frames of stillness needed for the slow (denoising) update.
    pub settle_slow: u8,
    /// While someone is in view, a pixel whose luma is below this fraction of the local bare-board luma is taken
    /// to be in a cast shadow and held. A hard shadow under a board light measured about 0.42; erased chalk
    /// returns to about 1.
    pub shadow_dark: f32,
    /// Darkening this close to a shadow pixel (fraction of board height) is held as shadow too: shadowed chalk,
    /// which can stay brighter than bare board, and soft shadow edges. Wider than a chalk stroke's half-width.
    pub shadow_reach: f32,
}

impl Default for GhostParams {
    fn default() -> Self {
        GhostParams {
            opacity: 0.25,
            grow: 0.025,
            hold_decay: 0.7,
            rate_slow: 0.08,
            rate_fast: 0.45,
            ink_diff: 24.0,
            motion: 14.0,
            settle: 10,
            settle_slow: 3,
            shadow_dark: 0.8,
            shadow_reach: 0.006,
        }
    }
}

// ------------------------------------------------------------------------------------------- worker

struct Shared {
    /// Newest frame waiting to be segmented, with its number (older ones are overwritten: we only care about the
    /// latest).
    frame: Mutex<Option<(u64, Rgb)>>,
    ready: Condvar,
    /// Newest matte, a counter bumped on every publish, and the number of the frame it was made from.
    matte: Mutex<(u64, Vec<f32>, u64)>,
    matte_ready: Condvar,
    /// Whether to run pose (only while the picture-in-picture needs it), and its newest result + counter.
    pose_on: AtomicBool,
    person: Mutex<(u64, crate::pose::Person)>,
    stop: AtomicBool,
}

pub struct Segmenter {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    down: ResampleMap,
    spare: Option<Rgb>,
    pub w: usize,
    pub h: usize,
    seen: u64,
    /// Frames submitted so far (the last one's number).
    submitted: u64,
    has_pose: bool,
    /// Segmentation runs on the GPU.
    pub gpu: bool,
}

impl Segmenter {
    pub fn start(
        model: &Path,
        (cam_w, cam_h): (usize, usize),
        rot: u32,
        (w, h): (usize, usize),
        ratio: f32,
        threads: usize,
        gpu: bool,
        max_fps: Option<f32>,
        pose_model: Option<&Path>,
    ) -> Result<Self> {
        let mut matting = Matting::new(model, w, h, ratio, threads, gpu)?;
        let gpu = matting.gpu;
        let mut pose = match pose_model {
            Some(m) => match crate::pose::Pose::new(m, w, h, threads, gpu) {
                Ok(p) => Some(p),
                Err(e) => {
                    crate::say!("no head tracking for the picture-in-picture: {e:#}");
                    None
                }
            },
            None => None,
        };
        let has_pose = pose.is_some();
        // On the GPU a run is a few ms, so by default keep up with the camera; on the CPU cap at 10 Hz.
        let max_fps = max_fps.unwrap_or(if gpu { 1000.0 } else { 10.0 });
        let min_period = std::time::Duration::from_secs_f32(1.0 / max_fps.max(0.1));
        let shared = Arc::new(Shared {
            frame: Mutex::new(None),
            ready: Condvar::new(),
            matte: Mutex::new((0, vec![0.0; w * h], 0)),
            matte_ready: Condvar::new(),
            pose_on: AtomicBool::new(false),
            person: Mutex::new((0, Default::default())),
            stop: AtomicBool::new(false),
        });
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("segment".into())
            .spawn(move || {
                let mut alpha = vec![0.0; w * h];
                loop {
                    let started = std::time::Instant::now();
                    let frame = {
                        let mut slot = sh.frame.lock().unwrap();
                        while slot.is_none() && !sh.stop.load(Ordering::Relaxed) {
                            slot = sh.ready.wait(slot).unwrap();
                        }
                        if sh.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        slot.take().unwrap()
                    };
                    let (number, frame) = frame;
                    match matting.run(&frame, &mut alpha) {
                        Ok(()) => {
                            let mut m = sh.matte.lock().unwrap();
                            m.0 += 1;
                            m.1.copy_from_slice(&alpha);
                            m.2 = number;
                            sh.matte_ready.notify_all();
                        }
                        Err(e) => crate::say!("segmentation failed: {e:#}"),
                    }
                    if let Some(p) = &mut pose
                        && sh.pose_on.load(Ordering::Relaxed)
                    {
                        match p.run(&frame) {
                            Ok(person) => {
                                let mut h = sh.person.lock().unwrap();
                                h.0 += 1;
                                h.1 = person;
                            }
                            Err(e) => crate::say!("pose failed: {e:#}"),
                        }
                    }
                    // Cap the rate; the compositor's stillness gate tolerates matte lag, so ~10 Hz is plenty on the CPU.
                    // Sleeping here (not before the run) means the next run takes the freshest frame.
                    if let Some(rest) = min_period.checked_sub(started.elapsed()) {
                        std::thread::sleep(rest);
                    }
                }
            })?;
        Ok(Segmenter {
            shared,
            thread: Some(thread),
            down: ResampleMap::upright(cam_w, cam_h, w, h, rot),
            spare: Some(Rgb::new(w, h)),
            w,
            h,
            seen: 0,
            submitted: 0,
            has_pose,
            gpu,
        })
    }

    /// Run pose on each segmented frame (costs ~25 ms a run, so only while something uses it).
    pub fn set_pose(&self, on: bool) {
        self.shared.pose_on.store(on, Ordering::Relaxed);
    }

    /// The newest pose result (in the upright segmentation frame) and a counter that changes with each run;
    /// None without a pose model.
    pub fn person(&self) -> Option<(u64, crate::pose::Person)> {
        self.has_pose.then(|| *self.shared.person.lock().unwrap())
    }

    /// Hand the worker the latest camera frame (downscaled here; cheap).
    pub fn submit(&mut self, frame: &Rgb) {
        let mut small = self
            .spare
            .take()
            .unwrap_or_else(|| Rgb::new(self.w, self.h));
        self.down.apply(frame, &mut small);
        self.submitted += 1;
        let mut slot = self.shared.frame.lock().unwrap();
        // If the worker hasn't taken the previous frame yet, recycle it.
        self.spare = slot.replace((self.submitted, small)).map(|(_, f)| f);
        self.shared.ready.notify_one();
    }

    /// Wait (at most `limit`) for the matte of the last frame submitted, so I'm cut out with my own outline
    /// rather than the previous frame's. Worth it only when segmentation is fast (the GPU: ~8 ms).
    pub fn wait_current(&self, limit: std::time::Duration) {
        let m = self.shared.matte.lock().unwrap();
        let _ = self
            .shared
            .matte_ready
            .wait_timeout_while(m, limit, |m| m.2 < self.submitted)
            .unwrap();
    }

    /// Copy out the newest matte if there's one we haven't seen.
    pub fn latest(&mut self, out: &mut [f32]) -> bool {
        let m = self.shared.matte.lock().unwrap();
        if m.0 == self.seen {
            return false;
        }
        self.seen = m.0;
        out.copy_from_slice(&m.1);
        true
    }
}

impl Drop for Segmenter {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.ready.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ------------------------------------------------------------------------------------------- compositor

/// Longest the ghost waits for the current frame's matte on the GPU (it normally takes ~8 ms; a frame at 15 fps
/// is 67 ms). Past this it uses the newest matte there is, as on the CPU.
const MATTE_WAIT: std::time::Duration = std::time::Duration::from_millis(30);

/// Board-brightness cells, in board pixels.
const LEVEL_CELL: usize = 16;
/// Frames between refreshes of the board-brightness estimate (the memory changes slowly).
const LEVEL_REFRESH: u32 = 15;
/// Share of a cell's pixels that must be bare board for its estimate to be bare board: the estimate is this
/// percentile of the cell's luma, so chalk can cover up to three quarters of a cell.
const LEVEL_PERCENTILE: f32 = 0.25;
/// Share of segmentation pixels that must be person (alpha > 0.5) for someone to count as in view.
const PRESENT_SHARE: f32 = 0.002;
/// Frames someone still counts as in view after the matte last showed them (bridges matte flicker and lag).
const PRESENT_HOLD: u32 = 15;
/// Sampling stride (board pixels, both ways) for the board-wide live / memory brightness ratio.
const LIGHT_STRIDE: usize = 8;

fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

pub struct Ghost {
    p: GhostParams,
    w: usize,
    h: usize,
    /// Segmentation-space matte -> board canvas.
    mask_map: ResampleMap,
    seg_alpha: Vec<f32>,
    /// Dilated matte in segmentation space (dilating there is ~30x cheaper than at board resolution).
    seg_grown: Vec<f32>,
    seg_scratch: Vec<f32>,
    seg_w: usize,
    seg_h: usize,
    /// Dilation radius in segmentation pixels.
    seg_r: usize,
    /// Person alpha in board space (soft, undilated): shapes the ghost.
    alpha: Vec<f32>,
    /// Dilated alpha, and the decaying hold built from it: gates memory updates.
    grown: Vec<f32>,
    hold: Vec<f32>,
    memory: Vec<f32>,
    prev: Vec<u8>,
    /// Consecutive frames each pixel has been still.
    still: Vec<u8>,
    /// What the memory did at each pixel on the last frame (see `debug`), for the debug view.
    state: Vec<u8>,
    /// Bare-board luma per `LEVEL_CELL` cell of the memory (a low percentile, then a 3x3-cell median), and the
    /// grid's width; refreshed every `LEVEL_REFRESH` frames.
    level: Vec<f32>,
    level_w: usize,
    frames: u32,
    /// Frames since the matte last showed someone (saturating).
    away: u32,
    /// Scratch for the brightness-ratio samples.
    ratios: Vec<f32>,
    /// Pixels clearly darker than bare board on the last frame (1 = shadow), and that grown by `reach` pixels:
    /// darkening inside the grown map is held as shadow on the next frame.
    dark: Vec<u8>,
    dark_tmp: Vec<u8>,
    near: Vec<u8>,
    reach: usize,
    started: bool,
}

impl Ghost {
    /// `mask_map` maps the segmentation frame onto the same board canvas as the video's board map;
    /// `seg_scale` is segmentation pixels per board pixel (for sizing the dilation).
    pub fn new(
        p: GhostParams,
        w: usize,
        h: usize,
        mask_map: ResampleMap,
        seg_w: usize,
        seg_h: usize,
        seg_scale: f32,
    ) -> Self {
        let n = w * h;
        let seg_r = ((p.grow * h as f32 * seg_scale).round() as usize).max(1);
        let reach = ((p.shadow_reach * h as f32).round() as usize).max(2);
        Ghost {
            seg_grown: vec![0.0; seg_w * seg_h],
            seg_scratch: vec![0.0; seg_w * seg_h],
            seg_w,
            seg_h,
            seg_r,
            p,
            w,
            h,
            mask_map,
            seg_alpha: vec![0.0; seg_w * seg_h],
            alpha: vec![0.0; n],
            grown: vec![0.0; n],
            hold: vec![0.0; n],
            memory: vec![0.0; n * 3],
            prev: vec![0; n * 3],
            still: vec![0; n],
            state: vec![debug::BOARD; n],
            level: vec![0.0; w.div_ceil(LEVEL_CELL) * h.div_ceil(LEVEL_CELL)],
            level_w: w.div_ceil(LEVEL_CELL),
            frames: 0,
            away: u32::MAX,
            ratios: Vec::new(),
            dark: vec![0; n],
            dark_tmp: vec![0; n],
            near: vec![0; n],
            reach,
            started: false,
        }
    }

    /// Update the board memory with this frame (taking the newest matte from the segmentation worker if there
    /// is one) and write the memory, without me, to `board`. Draw me over it afterwards with `overlay`.
    pub fn step(&mut self, live: &Rgb, seg: &mut Segmenter, board: &mut Rgb) {
        if seg.gpu {
            seg.wait_current(MATTE_WAIT);
        }
        let fresh = seg.latest(&mut self.seg_alpha);
        self.composite(live, fresh, board);
    }

    /// The board moved in the camera frame: use a new mask warp and start the memory afresh (the old memory
    /// is of the board at the old position).
    pub fn recalibrate(&mut self, mask_map: ResampleMap, seg_scale: f32) {
        self.mask_map = mask_map;
        self.seg_r = ((self.p.grow * self.h as f32 * seg_scale).round() as usize).max(1);
        self.started = false;
        self.hold.fill(0.0);
        self.still.fill(0);
        self.near.fill(0);
    }

    pub fn opacity(&self) -> f32 {
        self.p.opacity
    }

    pub fn set_opacity(&mut self, opacity: f32) {
        self.p.opacity = opacity;
    }

    /// Draw me (the live frame where the matte says person) over `board` at the ghost opacity.
    pub fn overlay(&self, board: &Rgb, live: &Rgb, out: &mut Rgb) {
        let (w, opacity) = (self.w, self.p.opacity);
        out.data
            .par_chunks_mut(w * 3)
            .zip(board.data.par_chunks(w * 3))
            .zip(live.data.par_chunks(w * 3))
            .zip(self.alpha.par_chunks(w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|(((out, board), live), alpha)| {
                for x in 0..w {
                    let a = opacity * alpha[x];
                    for c in x * 3..x * 3 + 3 {
                        let (b, l) = (board[c] as f32, live[c] as f32);
                        out[c] = (b + a * (l - b)).round() as u8;
                    }
                }
            });
    }

    /// Debug view: paint over `out` (which holds the memory, as `step` wrote it) what the memory did at each
    /// pixel on the last frame. The memory shows through as a dim grey so the writing stays readable.
    pub fn draw_debug(&self, out: &mut Rgb) {
        assert_eq!((out.w, out.h), (self.w, self.h));
        let (w, settle) = (self.w, self.p.settle.max(1) as f32);
        out.data
            .par_chunks_mut(w * 3)
            .zip(self.state.par_chunks(w))
            .zip(self.still.par_chunks(w))
            .zip(self.hold.par_chunks(w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|(((out, state), still), hold)| {
                for x in 0..w {
                    let i = x * 3;
                    let grey = 0.6
                        * (0.299 * out[i] as f32
                            + 0.587 * out[i + 1] as f32
                            + 0.114 * out[i + 2] as f32);
                    let (tint, mix) = match state[x] {
                        debug::PERSON => ([60.0, 110.0, 255.0], 0.55),
                        // Stronger where the hold is fresh, fading as it decays.
                        debug::HELD => ([230.0, 60.0, 230.0], 0.2 + 0.4 * hold[x].min(1.0)),
                        // Red when the pixel has just moved, through orange to yellow as it nears `settle`.
                        debug::WAITING => {
                            let p = (still[x] as f32 / settle).min(1.0);
                            ([255.0, 40.0 + 200.0 * p, 0.0], 0.7)
                        }
                        debug::ACCEPTING => ([40.0, 255.0, 80.0], 0.7),
                        debug::CREEPING => ([0.0, 200.0, 230.0], 0.5),
                        debug::SHADOW => ([150.0, 80.0, 220.0], 0.7),
                        _ => ([grey; 3], 0.0),
                    };
                    for c in 0..3 {
                        out[i + c] = (grey + mix * (tint[c] - grey)).round() as u8;
                    }
                }
            });
    }

    /// Share of board pixels in each debug state on the last frame, in `debug::NAMES` order.
    pub fn debug_shares(&self) -> [f32; debug::NAMES.len()] {
        let mut n = [0usize; debug::NAMES.len()];
        for &s in &self.state {
            n[s as usize] += 1;
        }
        n.map(|c| c as f32 / self.state.len().max(1) as f32)
    }

    /// Update the memory with one frame and write it to `out`. `fresh` says `seg_alpha` holds a new matte.
    fn composite(&mut self, live: &Rgb, fresh: bool, out: &mut Rgb) {
        assert_eq!((live.w, live.h), (self.w, self.h));
        if fresh {
            dilate(
                &self.seg_alpha,
                &mut self.seg_scratch,
                &mut self.seg_grown,
                self.seg_w,
                self.seg_h,
                self.seg_r,
            );
            self.mask_map.apply_gray(&self.seg_alpha, &mut self.alpha);
            self.mask_map.apply_gray(&self.seg_grown, &mut self.grown);
            let people = self.seg_alpha.iter().filter(|&&a| a > 0.5).count();
            if people as f32 > PRESENT_SHARE * self.seg_alpha.len() as f32 {
                self.away = 0;
            }
        }
        if !self.started {
            for (m, &v) in self.memory.iter_mut().zip(&live.data) {
                *m = v as f32;
            }
            self.prev.copy_from_slice(&live.data);
            self.started = true;
            self.frames = 0;
        }
        if self.frames.is_multiple_of(LEVEL_REFRESH) {
            self.refresh_level();
        }
        self.frames = self.frames.wrapping_add(1);
        let present = self.away < PRESENT_HOLD;
        self.away = self.away.saturating_add(1);
        let light = if present { self.light_change(live) } else { 1.0 };
        let p = &self.p;
        let w = self.w;
        let (level, level_w) = (&self.level, self.level_w);
        self.memory
            .par_chunks_mut(w * 3)
            .zip(out.data.par_chunks_mut(w * 3))
            .zip(live.data.par_chunks(w * 3))
            .zip(self.hold.par_chunks_mut(w))
            .zip(self.grown.par_chunks(w))
            .zip(self.prev.par_chunks_mut(w * 3))
            .zip(self.still.par_chunks_mut(w))
            .zip(self.state.par_chunks_mut(w))
            .zip(self.alpha.par_chunks(w))
            .zip(self.dark.par_chunks_mut(w))
            .zip(self.near.par_chunks(w))
            .enumerate()
            .with_min_len(ROWS_PER_TASK)
            .for_each(
                |(
                    y,
                    ((((((((((mem, out), live), hold), grown), prev), still), state), alpha), dark), near),
                )| {
                    let level = &level[(y / LEVEL_CELL) * level_w..][..level_w];
                    for x in 0..w {
                        let i = x * 3;
                        let (r, g, b) = (live[i] as f32, live[i + 1] as f32, live[i + 2] as f32);
                        let moved = (0..3)
                            .map(|c| (live[i + c] as i16 - prev[i + c] as i16).unsigned_abs())
                            .max()
                            .unwrap() as f32;
                        prev[i..i + 3].copy_from_slice(&live[i..i + 3]);
                        still[x] = if moved > p.motion {
                            0
                        } else {
                            still[x].saturating_add(1)
                        };
                        hold[x] = grown[x].max(hold[x] * p.hold_decay);
                        dark[x] = 0;
                        state[x] = if alpha[x] > 0.5 {
                            debug::PERSON
                        } else {
                            debug::HELD
                        };
                        if hold[x] < 0.05 {
                            let d = (r - mem[i])
                                .abs()
                                .max((g - mem[i + 1]).abs())
                                .max((b - mem[i + 2]).abs());
                            let y_live = luma(r, g, b);
                            let darker = present
                                && d > debug::CREEP_DIFF
                                && y_live < luma(mem[i], mem[i + 1], mem[i + 2]);
                            dark[x] =
                                (darker && y_live < p.shadow_dark * light * level[x / LEVEL_CELL]) as u8;
                            let shadow = darker && (dark[x] == 1 || near[x] == 1);
                            let k = if shadow {
                                state[x] = debug::SHADOW;
                                0.0
                            } else if d > p.ink_diff {
                                if still[x] >= p.settle {
                                    state[x] = debug::ACCEPTING;
                                    p.rate_fast
                                } else {
                                    state[x] = debug::WAITING;
                                    0.0
                                }
                            } else {
                                state[x] = if d > debug::CREEP_DIFF {
                                    debug::CREEPING
                                } else {
                                    debug::BOARD
                                };
                                if still[x] >= p.settle_slow {
                                    p.rate_slow
                                } else {
                                    0.0
                                }
                            };
                            mem[i] += k * (r - mem[i]);
                            mem[i + 1] += k * (g - mem[i + 1]);
                            mem[i + 2] += k * (b - mem[i + 2]);
                        }
                        out[i] = mem[i].round() as u8;
                        out[i + 1] = mem[i + 1].round() as u8;
                        out[i + 2] = mem[i + 2].round() as u8;
                    }
                },
            );
        // Grow this frame's shadow pixels for the next frame (a shadow keeps still, so a frame's lag is harmless).
        if present {
            spread(
                &self.dark,
                &mut self.dark_tmp,
                &mut self.near,
                w,
                self.h,
                self.reach,
            );
        } else {
            self.near.fill(0);
        }
    }
}

/// Binary dilation of a 0/1 map by a square of radius `r`: horizontal by a sliding count, vertical by OR.
fn spread(src: &[u8], tmp: &mut [u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    tmp.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(o, s)| {
            // Count of set pixels in [x - r, x + r], slid along the row.
            let mut n: u32 = s[..r.min(w)].iter().map(|&v| v as u32).sum();
            for x in 0..w {
                if x + r < w {
                    n += s[x + r] as u32;
                }
                if x > r {
                    n -= s[x - r - 1] as u32;
                }
                o[x] = (n > 0) as u8;
            }
        });
    dst.par_chunks_mut(w)
        .enumerate()
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(y, o)| {
            o.fill(0);
            for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                for (d, &v) in o.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                    *d |= v;
                }
            }
        });
}

impl Ghost {
    /// How much the room light has changed the board since the memory learned it: the median live / memory luma
    /// ratio over a sparse sample of pixels clear of me. A shadow covers a minority of the board, so it barely
    /// moves the median; dimming or brightening the lights moves all of it.
    fn light_change(&mut self, live: &Rgb) -> f32 {
        let (w, h) = (self.w, self.h);
        self.ratios.clear();
        for y in (LIGHT_STRIDE / 2..h).step_by(LIGHT_STRIDE) {
            for x in (LIGHT_STRIDE / 2..w).step_by(LIGHT_STRIDE) {
                let j = y * w + x;
                if self.hold[j] >= 0.05 {
                    continue;
                }
                let (m, l) = (&self.memory[j * 3..j * 3 + 3], &live.data[j * 3..j * 3 + 3]);
                let ym = luma(m[0], m[1], m[2]);
                if ym >= 8.0 {
                    self.ratios
                        .push(luma(l[0] as f32, l[1] as f32, l[2] as f32) / ym);
                }
            }
        }
        if self.ratios.len() < 50 {
            return 1.0;
        }
        let k = self.ratios.len() / 2;
        self.ratios
            .select_nth_unstable_by(k, f32::total_cmp)
            .1
            .clamp(0.2, 3.0)
    }

    /// Re-estimate the bare-board luma per cell from the memory: a low percentile per cell (ignores chalk covering
    /// up to three quarters of it), then a 3x3-cell median (ignores a cell that is all sticky note or writing).
    fn refresh_level(&mut self) {
        let (w, h, gw) = (self.w, self.h, self.level_w);
        let gh = h.div_ceil(LEVEL_CELL);
        let mem = &self.memory;
        let cells: Vec<f32> = (0..gw * gh)
            .into_par_iter()
            .map(|ci| {
                let (cx, cy) = (ci % gw, ci / gw);
                let mut v = Vec::with_capacity(LEVEL_CELL * LEVEL_CELL);
                for y in cy * LEVEL_CELL..((cy + 1) * LEVEL_CELL).min(h) {
                    for x in cx * LEVEL_CELL..((cx + 1) * LEVEL_CELL).min(w) {
                        let i = (y * w + x) * 3;
                        v.push(luma(mem[i], mem[i + 1], mem[i + 2]));
                    }
                }
                let k = ((v.len() - 1) as f32 * LEVEL_PERCENTILE) as usize;
                *v.select_nth_unstable_by(k, f32::total_cmp).1
            })
            .collect();
        self.level
            .par_iter_mut()
            .enumerate()
            .for_each(|(ci, out)| {
                let (cx, cy) = ((ci % gw) as isize, (ci / gw) as isize);
                let mut v = [0.0f32; 9];
                let mut n = 0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (x, y) = (cx + dx, cy + dy);
                        if (0..gw as isize).contains(&x) && (0..gh as isize).contains(&y) {
                            v[n] = cells[y as usize * gw + x as usize];
                            n += 1;
                        }
                    }
                }
                let v = &mut v[..n];
                *out = *v.select_nth_unstable_by(n / 2, f32::total_cmp).1;
            });
    }
}

/// Separable max filter of radius `r` (square neighbourhood).
fn dilate(src: &[f32], tmp: &mut [f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    tmp.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(o, s)| {
            for x in 0..w {
                let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
                o[x] = s[a..=b].iter().fold(0.0, |m, &v| m.max(v));
            }
        });
    dst.par_chunks_mut(w)
        .enumerate()
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(y, o)| {
            o.fill(0.0);
            for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                for (d, &v) in o.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                    *d = d.max(v);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOARD: [u8; 3] = [40, 90, 70];
    const SKIN: [u8; 3] = [220, 150, 120];
    const CHALK: [u8; 3] = [230, 235, 230];

    /// A ghost over a 120x80 board whose segmenter never sees anyone (the worst case of mask lag).
    fn blind_ghost() -> Ghost {
        let mask_map = ResampleMap::scale(12, 8, 120, 80);
        Ghost::new(GhostParams::default(), 120, 80, mask_map, 12, 8, 0.1)
    }

    fn frame(paint: impl Fn(usize, usize) -> Option<[u8; 3]>) -> Rgb {
        let mut f = Rgb::new(120, 80);
        for y in 0..80 {
            for x in 0..120 {
                let c = paint(x, y).unwrap_or(BOARD);
                f.data[(y * 120 + x) * 3..][..3].copy_from_slice(&c);
            }
        }
        f
    }

    fn mem_at(g: &Ghost, x: usize, y: usize) -> [f32; 3] {
        let i = (y * 120 + x) * 3;
        [g.memory[i], g.memory[i + 1], g.memory[i + 2]]
    }

    #[test]
    fn unmasked_moving_hand_leaves_no_trace() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        g.composite(&frame(|_, _| None), false, &mut out);
        // A 30x30 "hand" sweeping left to right at 4 px/frame, never masked.
        for t in 0..30 {
            let hx = t * 4;
            let f =
                frame(|x, y| ((hx..hx + 30).contains(&x) && (25..55).contains(&y)).then_some(SKIN));
            g.composite(&f, false, &mut out);
        }
        for x in 0..120 {
            let m = mem_at(&g, x, 40);
            let d = (0..3)
                .map(|c| (m[c] - BOARD[c] as f32).abs())
                .fold(0.0, f32::max);
            assert!(d < 4.0, "hand burned into memory at x={x}: {m:?}");
        }
    }

    #[test]
    fn new_chalk_arrives_within_half_a_second() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        g.composite(&frame(|_, _| None), false, &mut out);
        let chalked =
            frame(|x, y| ((50..70).contains(&x) && (38..42).contains(&y)).then_some(CHALK));
        let mut arrived = None;
        for t in 1..=30 {
            g.composite(&chalked, false, &mut out);
            if arrived.is_none() && out.data[(40 * 120 + 60) * 3] as f32 > CHALK[0] as f32 * 0.9 {
                arrived = Some(t);
            }
        }
        let t = arrived.expect("chalk never appeared");
        assert!(t <= 15, "chalk took {t} frames (> 0.5 s at 30 fps)");
    }

    #[test]
    fn debug_state_follows_new_chalk_from_waiting_to_accepted() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        g.composite(&frame(|_, _| None), false, &mut out);
        let chalked =
            frame(|x, y| ((50..70).contains(&x) && (38..42).contains(&y)).then_some(CHALK));
        let at = |g: &Ghost| g.state[40 * 120 + 60];
        g.composite(&chalked, false, &mut out);
        assert_eq!(at(&g), debug::WAITING);
        assert_eq!(g.state[10 * 120 + 10], debug::BOARD);
        let mut seen_accepting = false;
        for _ in 0..30 {
            g.composite(&chalked, false, &mut out);
            seen_accepting |= at(&g) == debug::ACCEPTING;
        }
        assert!(seen_accepting, "chalk never shown as going in");
        assert_eq!(at(&g), debug::BOARD, "settled chalk should read as board");
        let shares = g.debug_shares();
        assert!((shares.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        g.draw_debug(&mut out);
    }

    /// Composite `f` with someone standing at the left edge of the board (segmentation x 0..2, y 2..6, i.e.
    /// board x 0..20, y 20..60), as a fresh matte every frame.
    fn with_person(g: &mut Ghost, f: &Rgb, out: &mut Rgb) {
        for y in 0..8 {
            for x in 0..12 {
                g.seg_alpha[y * 12 + x] = if x < 2 && (2..6).contains(&y) { 1.0 } else { 0.0 };
            }
        }
        g.composite(f, true, out);
    }

    fn dist(m: [f32; 3], c: [u8; 3]) -> f32 {
        (0..3).map(|i| (m[i] - c[i] as f32).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn still_cast_shadow_stays_out_of_memory_while_someone_is_in_view() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        with_person(&mut g, &frame(|_, _| None), &mut out);
        // A hard shadow at 0.4x board brightness, well outside the grown matte, keeping perfectly still.
        let shadow = BOARD.map(|c| (c as f32 * 0.4) as u8);
        let shaded =
            frame(|x, y| ((60..95).contains(&x) && (15..65).contains(&y)).then_some(shadow));
        for _ in 0..60 {
            with_person(&mut g, &shaded, &mut out);
        }
        for (x, y) in [(60, 15), (77, 40), (94, 64), (61, 30)] {
            let m = mem_at(&g, x, y);
            assert!(dist(m, BOARD) < 4.0, "shadow learned at ({x},{y}): {m:?}");
            assert_eq!(g.state[y * 120 + x], debug::SHADOW);
        }
        // Once I'm gone (and my shadow with me) the memory is still the bare board.
        for _ in 0..30 {
            g.composite(&frame(|_, _| None), false, &mut out);
        }
        assert!(dist(mem_at(&g, 77, 40), BOARD) < 4.0);
    }

    #[test]
    fn chalk_under_a_shadow_stays_as_it_was() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        let stroke = |x: usize, y: usize| (50..90).contains(&x) && (38..41).contains(&y);
        with_person(&mut g, &frame(|x, y| stroke(x, y).then_some(CHALK)), &mut out);
        // The shadow covers the stroke. Shadowed chalk (0.4x) is still brighter than bare board, so on its own it
        // looks like chalk being rubbed out; only the shadowed board around it gives it away.
        let dim = |c: [u8; 3]| c.map(|v| (v as f32 * 0.4) as u8);
        let shaded = frame(|x, y| {
            ((60..95).contains(&x) && (15..65).contains(&y))
                .then(|| dim(if stroke(x, y) { CHALK } else { BOARD }))
                .or(stroke(x, y).then_some(CHALK))
        });
        for _ in 0..60 {
            with_person(&mut g, &shaded, &mut out);
        }
        for x in [61, 70, 80, 89] {
            let m = mem_at(&g, x, 39);
            assert!(dist(m, CHALK) < 4.0, "shadowed chalk learned at x={x}: {m:?}");
        }
        assert!(dist(mem_at(&g, 77, 25), BOARD) < 4.0);
    }

    #[test]
    fn dimming_the_room_light_is_not_a_shadow() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        let chalk = |x: usize, y: usize| (50..90).contains(&x) && (38..41).contains(&y);
        with_person(&mut g, &frame(|x, y| chalk(x, y).then_some(CHALK)), &mut out);
        // The whole room dims to 0.6x, with my (still) shadow on part of the board.
        let dim = |c: [u8; 3], k: f32| c.map(|v| (v as f32 * k) as u8);
        let dimmed = frame(|x, y| {
            let base = if chalk(x, y) { CHALK } else { BOARD };
            Some(if (100..115).contains(&x) && (15..65).contains(&y) {
                dim(base, 0.6 * 0.4)
            } else {
                dim(base, 0.6)
            })
        });
        for _ in 0..90 {
            with_person(&mut g, &dimmed, &mut out);
        }
        for (x, y) in [(40, 10), (77, 70), (60, 39)] {
            let want = dim(if chalk(x, y) { CHALK } else { BOARD }, 0.6);
            let m = mem_at(&g, x, y);
            assert!(dist(m, want) < 4.0, "room dimming not learned at ({x},{y}): {m:?}");
        }
        // The shadow is still a shadow against the dimmed board.
        assert!(dist(mem_at(&g, 107, 40), BOARD) < 4.0);
        assert_eq!(g.state[40 * 120 + 107], debug::SHADOW);
    }

    #[test]
    fn erasing_goes_in_while_someone_is_in_view() {
        let mut g = blind_ghost();
        let mut out = Rgb::new(120, 80);
        let chalked =
            frame(|x, y| ((50..70).contains(&x) && (38..42).contains(&y)).then_some(CHALK));
        with_person(&mut g, &chalked, &mut out);
        assert!(dist(mem_at(&g, 60, 40), CHALK) < 1.0);
        // Wiped back to the board (a little lighter, as wiped chalk is).
        let wiped = frame(|x, y| {
            ((50..70).contains(&x) && (38..42).contains(&y)).then_some(BOARD.map(|c| c + 6))
        });
        let mut gone = None;
        for t in 1..=30 {
            with_person(&mut g, &wiped, &mut out);
            if gone.is_none() && dist(mem_at(&g, 60, 40), BOARD) < 24.0 {
                gone = Some(t);
            }
        }
        let t = gone.expect("erasure never went in");
        assert!(t <= 15, "erasure took {t} frames");
    }

    #[test]
    fn g_cycles_hidden_faint_ghost_opaque() {
        let mut o = 0.25; // the default
        let mut seen = Vec::new();
        for _ in 0..6 {
            o = next_opacity(o);
            seen.push(o);
        }
        assert_eq!(seen, [0.5, 1.0, 0.0, 0.1, 0.25, 0.5]);
        assert_eq!(next_opacity(0.4), 0.5); // a custom --ghost-opacity
    }

    #[test]
    fn spread_grows_a_point_into_a_square() {
        let (w, h) = (9, 7);
        let mut src = vec![0u8; w * h];
        src[3 * w + 4] = 1;
        let (mut tmp, mut dst) = (vec![0; w * h], vec![0; w * h]);
        spread(&src, &mut tmp, &mut dst, w, h, 2);
        for y in 0..h {
            for x in 0..w {
                let inside = (1..=5).contains(&y) && (2..=6).contains(&x);
                assert_eq!(dst[y * w + x], inside as u8, "({x},{y})");
            }
        }
    }

    #[test]
    fn dilate_grows_a_point_into_a_square() {
        let (w, h) = (7, 5);
        let mut src = vec![0.0; w * h];
        src[2 * w + 3] = 1.0;
        let (mut tmp, mut dst) = (vec![0.0; w * h], vec![0.0; w * h]);
        dilate(&src, &mut tmp, &mut dst, w, h, 1);
        for y in 0..h {
            for x in 0..w {
                let inside = (1..=3).contains(&y) && (2..=4).contains(&x);
                assert_eq!(dst[y * w + x], if inside { 1.0 } else { 0.0 }, "({x},{y})");
            }
        }
    }
}
