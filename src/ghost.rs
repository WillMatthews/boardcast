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
        }
    }
}

// ------------------------------------------------------------------------------------------- worker

struct Shared {
    /// Newest frame waiting to be segmented (older ones are overwritten: we only care about the latest).
    frame: Mutex<Option<Rgb>>,
    ready: Condvar,
    /// Newest matte and a counter bumped on every publish.
    matte: Mutex<(u64, Vec<f32>)>,
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
    has_pose: bool,
}

impl Segmenter {
    pub fn start(
        model: &Path,
        (cam_w, cam_h): (usize, usize),
        rot: u32,
        (w, h): (usize, usize),
        ratio: f32,
        threads: usize,
        max_fps: f32,
        pose_model: Option<&Path>,
    ) -> Result<Self> {
        let mut matting = Matting::new(model, w, h, ratio, threads)?;
        let mut pose = match pose_model {
            Some(m) => match crate::pose::Pose::new(m, w, h, threads) {
                Ok(p) => Some(p),
                Err(e) => {
                    crate::say!("head tracking falls back to the matte: {e:#}");
                    None
                }
            },
            None => None,
        };
        let has_pose = pose.is_some();
        let min_period = std::time::Duration::from_secs_f32(1.0 / max_fps.max(0.1));
        let shared = Arc::new(Shared {
            frame: Mutex::new(None),
            ready: Condvar::new(),
            matte: Mutex::new((0, vec![0.0; w * h])),
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
                    match matting.run(&frame, &mut alpha) {
                        Ok(()) => {
                            let mut m = sh.matte.lock().unwrap();
                            m.0 += 1;
                            m.1.copy_from_slice(&alpha);
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
                    // Cap the rate; the compositor's stillness gate tolerates matte lag, so ~10 Hz is plenty.
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
            has_pose,
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
        let mut slot = self.shared.frame.lock().unwrap();
        // If the worker hasn't taken the previous frame yet, recycle it.
        self.spare = slot.replace(small);
        self.shared.ready.notify_one();
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

pub struct Ghost {
    p: GhostParams,
    w: usize,
    h: usize,
    /// Segmentation-space matte -> board canvas.
    mask_map: ResampleMap,
    seg_alpha: Vec<f32>,
    /// Counts new mattes (for things that follow me, like the picture-in-picture).
    matte_version: u64,
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
            matte_version: 0,
            alpha: vec![0.0; n],
            grown: vec![0.0; n],
            hold: vec![0.0; n],
            memory: vec![0.0; n * 3],
            prev: vec![0; n * 3],
            still: vec![0; n],
            started: false,
        }
    }

    /// Update the board memory with this frame (taking the newest matte from the segmentation worker if there
    /// is one) and write the memory, without me, to `board`. Draw me over it afterwards with `overlay`.
    pub fn step(&mut self, live: &Rgb, seg: &mut Segmenter, board: &mut Rgb) {
        let fresh = seg.latest(&mut self.seg_alpha);
        self.matte_version += fresh as u64;
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
    }

    /// The latest person matte, in the (upright) segmentation frame: (matte, width, height, version).
    pub fn matte(&self) -> (&[f32], usize, usize, u64) {
        (&self.seg_alpha, self.seg_w, self.seg_h, self.matte_version)
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
        }
        if !self.started {
            for (m, &v) in self.memory.iter_mut().zip(&live.data) {
                *m = v as f32;
            }
            self.prev.copy_from_slice(&live.data);
            self.started = true;
        }
        let p = &self.p;
        let w = self.w;
        self.memory
            .par_chunks_mut(w * 3)
            .zip(out.data.par_chunks_mut(w * 3))
            .zip(live.data.par_chunks(w * 3))
            .zip(self.hold.par_chunks_mut(w))
            .zip(self.grown.par_chunks(w))
            .zip(self.prev.par_chunks_mut(w * 3))
            .zip(self.still.par_chunks_mut(w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|((((((mem, out), live), hold), grown), prev), still)| {
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
                    if hold[x] < 0.05 {
                        let d = (r - mem[i])
                            .abs()
                            .max((g - mem[i + 1]).abs())
                            .max((b - mem[i + 2]).abs());
                        let k = if d > p.ink_diff {
                            if still[x] >= p.settle {
                                p.rate_fast
                            } else {
                                0.0
                            }
                        } else if still[x] >= p.settle_slow {
                            p.rate_slow
                        } else {
                            0.0
                        };
                        mem[i] += k * (r - mem[i]);
                        mem[i + 1] += k * (g - mem[i + 1]);
                        mem[i + 2] += k * (b - mem[i + 2]);
                    }
                    out[i] = mem[i].round() as u8;
                    out[i + 1] = mem[i + 1].round() as u8;
                    out[i + 2] = mem[i + 2].round() as u8;
                }
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
