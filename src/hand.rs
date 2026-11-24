//! My hand, finely: MediaPipe's hand landmark model (21 points; Apache-2.0, converted to ONNX by
//! scripts/fetch-hand-model.sh) on a crop of the full-resolution camera frame around the hand.
//!
//! Like MediaPipe's own tracker, the crop comes from the previous frame's landmarks while the hand is tracked,
//! and from pose (wrist and elbow) to find it in the first place. The crop is turned so the hand points up, which
//! is what the model expects.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;

use crate::image::{Rgb, from_upright};

const SIZE: usize = 224;
/// Hand presence needed to use the landmarks.
const PRESENT: f32 = 0.5;

/// Crop region, in upright camera pixels: centre, side, and the hand's "up" (unit vector, wrist towards fingers).
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub centre: [f32; 2],
    pub side: f32,
    pub up: [f32; 2],
}

impl Roi {
    /// From pose: the hand continues the forearm beyond the wrist.
    pub fn from_arm(wrist: [f32; 2], elbow: [f32; 2]) -> Option<Self> {
        let d = [wrist[0] - elbow[0], wrist[1] - elbow[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if !(len > 4.0) {
            return None;
        }
        let up = [d[0] / len, d[1] / len];
        Some(Roi {
            centre: [wrist[0] + 0.45 * len * up[0], wrist[1] + 0.45 * len * up[1]],
            side: 1.5 * len,
            up,
        })
    }

    /// From the last landmarks: aligned wrist -> middle knuckle, sized to the hand with a margin.
    pub fn from_hand(pts: &[[f32; 2]; 21]) -> Option<Self> {
        let d = [pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if !(len > 2.0) {
            return None;
        }
        let up = [d[0] / len, d[1] / len];
        let side_dir = [-up[1], up[0]];
        let proj = |p: &[f32; 2], a: [f32; 2]| p[0] * a[0] + p[1] * a[1];
        let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in pts {
            let (a, b) = (proj(p, side_dir), proj(p, up));
            (a0, a1, b0, b1) = (a0.min(a), a1.max(a), b0.min(b), b1.max(b));
        }
        let (ca, cb) = ((a0 + a1) / 2.0, (b0 + b1) / 2.0);
        let centre = [ca * side_dir[0] + cb * up[0], ca * side_dir[1] + cb * up[1]];
        let side = (a1 - a0).max(b1 - b0).max(len) * 2.0;
        Some(Roi { centre, side, up })
    }

    /// Crop pixel (x right, y down, 0..SIZE) -> upright camera pixel.
    fn to_upright(&self, x: f32, y: f32) -> [f32; 2] {
        let s = self.side / SIZE as f32;
        let (a, b) = ((x - SIZE as f32 / 2.0) * s, (y - SIZE as f32 / 2.0) * s);
        // Crop "down" is the hand's -up; crop "right" is perpendicular (a rotation, not a mirror).
        let ey = [-self.up[0], -self.up[1]];
        let ex = [ey[1], -ey[0]];
        [
            self.centre[0] + a * ex[0] + b * ey[0],
            self.centre[1] + a * ex[1] + b * ey[1],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hand {
    /// The 21 MediaPipe landmarks (0 wrist, 4 thumb tip, 8 index tip, ...), upright camera pixels.
    pub pts: [[f32; 2]; 21],
}

impl Hand {
    pub fn index_tip(&self) -> [f32; 2] {
        self.pts[8]
    }

    /// Index finger out, at least two of the other three fingers curled.
    pub fn pointing(&self) -> bool {
        let d = |i: usize| {
            let (p, w) = (self.pts[i], self.pts[0]);
            ((p[0] - w[0]).powi(2) + (p[1] - w[1]).powi(2)).sqrt()
        };
        let index_out = d(8) > 1.25 * d(6);
        let curled = [(12, 10), (16, 14), (20, 18)]
            .iter()
            .filter(|&&(tip, pip)| d(tip) < 1.05 * d(pip))
            .count();
        index_out && curled >= 2
    }
}

struct Shared {
    /// Crop, its region, and which hand (0 left, 1 right).
    job: Mutex<Option<(Vec<f32>, Roi, usize)>>,
    ready: Condvar,
    result: Mutex<[(u64, Option<Hand>); 2]>,
    stop: AtomicBool,
}

/// Runs the hand model on its own thread; `submit` crops (cheap) on the caller's thread.
pub struct HandTracker {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    min_period: Duration,
    last_submit: Instant,
    /// Which hand to look at next (they take turns).
    next: usize,
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("onnxruntime: {e}")
}

impl HandTracker {
    pub fn start(model: &Path, threads: usize, max_fps: f32) -> Result<Self> {
        let env = crate::segment::ort_env()?;
        let mut session = Session::builder(&env)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            .with_intra_op_spinning(false)
            .map_err(ort_err)?
            .with_inter_op_spinning(false)
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)
            .with_context(|| {
                format!(
                    "loading {} (run scripts/fetch-hand-model.sh)",
                    model.display()
                )
            })?;
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            ready: Condvar::new(),
            result: Mutex::new([(0, None); 2]),
            stop: AtomicBool::new(false),
        });
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("hand".into())
            .spawn(move || {
                let _env = env;
                loop {
                    let (crop, roi, side) = {
                        let mut slot = sh.job.lock().unwrap();
                        while slot.is_none() && !sh.stop.load(Ordering::Relaxed) {
                            slot = sh.ready.wait(slot).unwrap();
                        }
                        if sh.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        slot.take().unwrap()
                    };
                    let hand = match run(&mut session, crop, &roi) {
                        Ok(h) => h,
                        Err(e) => {
                            crate::say!("hand tracking failed: {e:#}");
                            None
                        }
                    };
                    let mut r = sh.result.lock().unwrap();
                    r[side].0 += 1;
                    r[side].1 = hand;
                }
            })?;
        Ok(HandTracker {
            shared,
            thread: Some(thread),
            min_period: Duration::from_secs_f32(1.0 / max_fps.max(0.1)),
            last_submit: Instant::now() - Duration::from_secs(1),
            next: 0,
        })
    }

    /// Look for my hands in `rois` ([left, right]; None where there's nothing to look at): crop the next one's
    /// region from the camera `frame` (turned `rot` to upright) and queue it. The hands take turns, and the total
    /// is at most `max_fps` a second.
    pub fn submit(&mut self, frame: &Rgb, rot: u32, rois: [Option<Roi>; 2]) {
        if self.last_submit.elapsed() < self.min_period {
            return;
        }
        let side = if rois[self.next].is_some() {
            self.next
        } else {
            1 - self.next
        };
        let Some(roi) = rois[side] else { return };
        self.next = 1 - side;
        self.last_submit = Instant::now();
        let crop = crop(frame, rot, &roi);
        *self.shared.job.lock().unwrap() = Some((crop, roi, side));
        self.shared.ready.notify_one();
    }

    /// The newest result for each hand ([left, right]) and counters that change with each run.
    pub fn latest(&self) -> [(u64, Option<Hand>); 2] {
        *self.shared.result.lock().unwrap()
    }
}

impl Drop for HandTracker {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.ready.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The model's input: SIZE x SIZE RGB in 0..1, NHWC, averaged over each pixel's footprint when shrinking.
fn crop(frame: &Rgb, rot: u32, roi: &Roi) -> Vec<f32> {
    let scale = roi.side / SIZE as f32;
    let n = (scale.ceil() as usize).clamp(1, 3);
    let mut out = vec![0.0f32; SIZE * SIZE * 3];
    let at = |x: usize, y: usize, c: usize| frame.data[(y * frame.w + x) * 3 + c] as f32;
    for j in 0..SIZE {
        for i in 0..SIZE {
            let mut acc = [0.0f32; 3];
            for b in 0..n {
                for a in 0..n {
                    let u = roi.to_upright(
                        i as f32 + (a as f32 + 0.5) / n as f32,
                        j as f32 + (b as f32 + 0.5) / n as f32,
                    );
                    let [x, y] = from_upright([u[0] as f64, u[1] as f64], (frame.w, frame.h), rot);
                    let (x, y) = (
                        (x as f32 - 0.5).clamp(0.0, (frame.w - 2) as f32),
                        (y as f32 - 0.5).clamp(0.0, (frame.h - 2) as f32),
                    );
                    let (x0, y0) = (x as usize, y as usize);
                    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                    for c in 0..3 {
                        let t = at(x0, y0, c) * (1.0 - fx) + at(x0 + 1, y0, c) * fx;
                        let bt = at(x0, y0 + 1, c) * (1.0 - fx) + at(x0 + 1, y0 + 1, c) * fx;
                        acc[c] += t * (1.0 - fy) + bt * fy;
                    }
                }
            }
            let o = (j * SIZE + i) * 3;
            for c in 0..3 {
                out[o + c] = acc[c] / (n * n) as f32 / 255.0;
            }
        }
    }
    out
}

fn run(session: &mut Session, crop: Vec<f32>, roi: &Roi) -> Result<Option<Hand>> {
    let t = Tensor::from_array((vec![1i64, SIZE as i64, SIZE as i64, 3], crop)).map_err(ort_err)?;
    let out = session.run(ort::inputs!["input_1" => t]).map_err(ort_err)?;
    let (_, presence) = out["Identity_1"]
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    let presence = presence.first().copied().unwrap_or(0.0);
    if presence < PRESENT {
        return Ok(None);
    }
    let (_, lm) = out["Identity"]
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    if lm.len() != 63 {
        return Err(anyhow!(
            "unexpected hand landmark output length {}",
            lm.len()
        ));
    }
    let mut pts = [[0.0; 2]; 21];
    for (k, p) in pts.iter_mut().enumerate() {
        *p = roi.to_upright(lm[3 * k], lm[3 * k + 1]);
    }
    Ok(Some(Hand { pts }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_turns_the_hand_upright() {
        // Forearm pointing right (elbow left of wrist): the crop's top should be to the right of the wrist.
        let roi = Roi::from_arm([100.0, 50.0], [60.0, 50.0]).unwrap();
        let top = roi.to_upright(112.0, 0.0);
        let bottom = roi.to_upright(112.0, 224.0);
        assert!(
            top[0] > bottom[0] + 50.0 && (top[1] - bottom[1]).abs() < 1e-3,
            "{top:?} {bottom:?}"
        );
        // Crop right is a quarter turn clockwise from up: with up = +x (right), crop right = +y (down).
        let right = roi.to_upright(224.0, 112.0);
        assert!(right[1] > roi.centre[1] + 20.0, "{right:?}");
    }

    #[test]
    fn pointing_is_index_out_and_others_curled() {
        // Wrist at the origin, fingers "up" along -y. Joints: knuckle (MCP) at 40, PIP at 55, tips at 85 if out or
        // 35 if curled back towards the palm.
        let mut pts = [[0.0f32; 2]; 21];
        let finger = |pts: &mut [[f32; 2]; 21], base: usize, x: f32, out: bool| {
            pts[base] = [x, -40.0];
            pts[base + 1] = [x, -55.0];
            pts[base + 2] = [x, if out { -70.0 } else { -45.0 }];
            pts[base + 3] = [x, if out { -85.0 } else { -35.0 }];
        };
        finger(&mut pts, 5, -10.0, true);
        for (b, x) in [(9, 0.0), (13, 10.0), (17, 20.0)] {
            finger(&mut pts, b, x, false);
        }
        let h = Hand { pts };
        assert!(h.pointing());
        let mut open = pts;
        for (b, x) in [(9, 0.0), (13, 10.0), (17, 20.0)] {
            finger(&mut open, b, x, true);
        }
        assert!(!Hand { pts: open }.pointing(), "open hand isn't pointing");
    }
}
