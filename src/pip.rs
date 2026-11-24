//! Picture-in-picture: a squircle crop of me, following my head, inset in a corner of the board feed.
//!
//! The head is found from the person matte (segmentation frame): the top of the matte, with the head's width from
//! the first rows below it. The crop is a square a few head-widths across, so it frames me the same at the board
//! or at the desk, and is smoothed so it glides rather than jitters.

use std::time::Instant;

use rayon::prelude::*;

use crate::image::{Rgb, from_upright, upright_size};
use crate::track::Kalman;

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomRight,
    BottomLeft,
}

impl Corner {
    pub fn next(self) -> Self {
        match self {
            Corner::TopLeft => Corner::TopRight,
            Corner::TopRight => Corner::BottomRight,
            Corner::BottomRight => Corner::BottomLeft,
            Corner::BottomLeft => Corner::TopLeft,
        }
    }
}

/// Squircle exponent (2 = circle/ellipse, higher = squarer).
const SQUIRCLE_N: f32 = 4.5;
/// Crop side in head widths, and where the head's top sits (fraction of the crop from its top).
const HEAD_WIDTHS: f32 = 3.2;
const HEAD_TOP: f32 = 0.18;
const BORDER: f32 = 3.0;
const SHADOW: f32 = 10.0;

pub struct Pip {
    pub on: bool,
    pub corner: Corner,
    /// Inset size as a fraction of the canvas width.
    size: f32,
    /// Crop in upright camera pixels (centre x, centre y, side), filtered. None until I've been seen.
    crop: Option<CropFilter>,
    /// A new measurement waiting to be folded in at the next frame.
    measured: Option<[f32; 3]>,
    /// 0..1: fades in when turned on / I'm seen, out when turned off / I'm gone.
    fade: f32,
    seen: Option<Instant>,
    last: Instant,
    /// Version of the matte the crop was last updated from (update once per new matte).
    matte_seen: u64,
    /// Pose results: the last one used, when pose last found me, and the last head width it measured (in upright
    /// camera pixels) for frames where it can't tell.
    head_seen: u64,
    pose_found: Option<Instant>,
    head_width: Option<f32>,
    patch: Rgb,
    /// Where the inset is drawn (top-left, canvas pixels) and how fast it's moving: it glides between corners.
    at: Option<[f32; 2]>,
    at_v: [f32; 2],
}

impl Pip {
    pub fn new(size: f32, corner: Corner, on: bool) -> Self {
        Pip {
            on,
            corner,
            size,
            crop: None,
            measured: None,
            fade: 0.0,
            seen: None,
            last: Instant::now(),
            matte_seen: u64::MAX,
            head_seen: u64::MAX,
            pose_found: None,
            head_width: None,
            patch: Rgb::new(1, 1),
            at: None,
            at_v: [0.0; 2],
        }
    }

    /// Follow my head in the matte (`mw` x `mh`, the upright frame scaled down); `version` changes with each new
    /// matte.
    pub fn track(
        &mut self,
        matte: &[f32],
        mw: usize,
        mh: usize,
        version: u64,
        upright: (usize, usize),
    ) {
        if version == self.matte_seen {
            return;
        }
        self.matte_seen = version;
        // Pose is better at finding my head (raised arms don't fool it); the matte is the fallback.
        if self
            .pose_found
            .is_some_and(|t| t.elapsed().as_secs_f32() < 0.6)
        {
            return;
        }
        let Some([cx, top, hw]) = find_head(matte, mw, mh) else {
            return;
        };
        let k = upright.0 as f32 / mw as f32;
        let side =
            (hw * k * HEAD_WIDTHS).clamp(upright.0 as f32 * 0.12, upright.0.min(upright.1) as f32);
        let target = [cx * k, top * k - HEAD_TOP * side + side / 2.0, side];
        self.seen = Some(Instant::now());
        self.measured = Some(target);
    }

    /// Follow my head as found by pose (`head` in the `seg` frame's pixels, `version` changes with each run).
    pub fn track_head(
        &mut self,
        version: u64,
        head: Option<crate::pose::Head>,
        seg: (usize, usize),
        upright: (usize, usize),
    ) {
        if version == self.head_seen {
            return;
        }
        self.head_seen = version;
        let Some(h) = head else { return };
        let k = upright.0 as f32 / seg.0 as f32;
        if let Some(w) = h.width {
            self.head_width = Some(w * k);
        }
        let Some(hw) = self.head_width else { return };
        let side =
            (hw * HEAD_WIDTHS).clamp(upright.0 as f32 * 0.12, upright.0.min(upright.1) as f32);
        // Eye level is about 0.55 head widths below the top of the head.
        let top = h.y * k - 0.55 * hw;
        self.pose_found = Some(Instant::now());
        self.seen = Some(Instant::now());
        self.measured = Some([h.x * k, top - HEAD_TOP * side + side / 2.0, side]);
    }

    /// Whether pose is currently finding me (for the status display).
    pub fn using_pose(&self) -> bool {
        self.pose_found
            .is_some_and(|t| t.elapsed().as_secs_f32() < 0.6)
    }

    /// Draw the inset onto `out`, cropping from the camera `frame` (turned `rot` to upright).
    pub fn draw(&mut self, frame: &Rgb, rot: u32, out: &mut Rgb) {
        let dt = self.last.elapsed().as_secs_f32();
        self.last = Instant::now();
        let present = self.seen.is_some_and(|t| t.elapsed().as_secs_f32() < 1.5);
        let goal = if self.on && present { 1.0 } else { 0.0 };
        let step = dt / 0.25; // quarter-second fades
        self.fade = if goal > self.fade {
            (self.fade + step).min(goal)
        } else {
            (self.fade - step).max(goal)
        };
        // Predict to this frame, then fold in a new measurement if there is one: smooth motion at the video's
        // frame rate even though measurements come at ~10 Hz.
        if let Some(z) = self.measured.take() {
            match &mut self.crop {
                None => self.crop = Some(CropFilter::new(z)),
                Some(f) => {
                    f.predict(dt);
                    f.update(z);
                }
            }
        } else if let Some(f) = &mut self.crop {
            f.predict(dt);
        }
        let Some([cx, cy, side]) = self.crop.as_ref().map(CropFilter::value) else {
            return;
        };
        if self.fade <= 0.0 {
            self.at = None; // reappear in place rather than gliding in from the last corner
            self.at_v = [0.0; 2];
            return;
        }
        let ease = self.fade * self.fade * (3.0 - 2.0 * self.fade);

        // Sample the crop into a square patch.
        let p = ((out.w as f32 * self.size).round() as usize).clamp(32, out.w.min(out.h));
        if (self.patch.w, self.patch.h) != (p, p) {
            self.patch = Rgb::new(p, p);
        }
        let up = upright_size((frame.w, frame.h), rot);
        // Keep the crop inside the picture.
        let side = side.min(up.0.min(up.1) as f32);
        let x0 = (cx - side / 2.0).clamp(0.0, up.0 as f32 - side);
        let y0 = (cy - side / 2.0).clamp(0.0, up.1 as f32 - side);
        // Each inset pixel covers k x k camera pixels: average an n x n grid of samples over that footprint
        // (one sample per camera pixel or so), or fine detail aliases and shimmers as the crop glides.
        let k = side / p as f32;
        if k <= 1.0 {
            // Enlarging (I'm far away): one bilinear sample per pixel is right.
            self.patch
                .data
                .par_chunks_mut(p * 3)
                .enumerate()
                .for_each(|(j, row)| {
                    for i in 0..p {
                        let u = x0 + (i as f32 + 0.5) * k;
                        let v = y0 + (j as f32 + 0.5) * k;
                        let [x, y] = from_upright([u as f64, v as f64], (frame.w, frame.h), rot);
                        let px = bilinear(frame, x as f32 - 0.5, y as f32 - 0.5);
                        for c in 0..3 {
                            row[i * 3 + c] = px[c].round().clamp(0.0, 255.0) as u8;
                        }
                    }
                });
        } else {
            // Shrinking: a box filter of point samples. The camera is only ever turned by a multiple of 90
            // degrees, so each sample column (u) and row (v) of the upright picture lands on one camera column or
            // row: work out their byte offsets once, then each sample is an add.
            let n = (k.ceil() as usize).clamp(1, 8);
            let sub = k / n as f32;
            let (fw, fh) = (frame.w, frame.h);
            let idx = |c: f32, len: usize| (c.floor().max(0.0) as usize).min(len - 1);
            // Offset contributed by an upright coordinate along x (u) or y (v).
            let u_off = |u: f32| match rot {
                90 => idx(fh as f32 - u, fh) * fw * 3,
                180 => idx(fw as f32 - u, fw) * 3,
                270 => idx(u, fh) * fw * 3,
                _ => idx(u, fw) * 3,
            };
            let v_off = |v: f32| match rot {
                90 => idx(v, fw) * 3,
                180 => idx(fh as f32 - v, fh) * fw * 3,
                270 => idx(fw as f32 - v, fw) * 3,
                _ => idx(v, fh) * fw * 3,
            };
            let us: Vec<usize> = (0..p * n)
                .map(|t| u_off(x0 + (t as f32 + 0.5) * sub))
                .collect();
            let vs: Vec<usize> = (0..p * n)
                .map(|t| v_off(y0 + (t as f32 + 0.5) * sub))
                .collect();
            let (nn, data) = ((n * n) as u32, &frame.data);
            self.patch
                .data
                .par_chunks_mut(p * 3)
                .enumerate()
                .for_each(|(j, row)| {
                    let rows = &vs[j * n..(j + 1) * n];
                    for i in 0..p {
                        let mut acc = [0u32; 3];
                        for &vo in rows {
                            for &uo in &us[i * n..(i + 1) * n] {
                                let o = vo + uo;
                                acc[0] += data[o] as u32;
                                acc[1] += data[o + 1] as u32;
                                acc[2] += data[o + 2] as u32;
                            }
                        }
                        for c in 0..3 {
                            row[i * 3 + c] = ((acc[c] + nn / 2) / nn) as u8;
                        }
                    }
                });
        }

        // Composite: soft shadow, light border, squircle-masked picture.
        let margin = (out.w.min(out.h) as f32 * 0.03).round() as isize + SHADOW as isize;
        let (right, bottom) = (
            out.w as isize - margin - p as isize,
            out.h as isize - margin - p as isize,
        );
        let goal = match self.corner {
            Corner::TopLeft => [margin, margin],
            Corner::TopRight => [right, margin],
            Corner::BottomRight => [right, bottom],
            Corner::BottomLeft => [margin, bottom],
        }
        .map(|v| v as f32);
        // Critically damped spring towards the corner: eases out and in, ~0.6 s across the feed.
        let at = self.at.get_or_insert(goal);
        let w = 9.0;
        let dt = dt.clamp(0.0, 0.1);
        for i in 0..2 {
            let a = w * w * (goal[i] - at[i]) - 2.0 * w * self.at_v[i];
            self.at_v[i] += a * dt;
            at[i] += self.at_v[i] * dt;
        }
        let (ox, oy) = (at[0].round() as isize, at[1].round() as isize);
        let r = p as f32 / 2.0;
        let pad = SHADOW as isize + 2;
        for j in -pad..p as isize + pad {
            let y = oy + j;
            if y < 0 || y >= out.h as isize {
                continue;
            }
            for i in -pad..p as isize + pad {
                let x = ox + i;
                if x < 0 || x >= out.w as isize {
                    continue;
                }
                // Signed distance (pixels, negative inside) to the squircle's edge, approximately.
                let nx = ((i as f32 + 0.5) - r) / r;
                let ny = ((j as f32 + 0.5) - r) / r;
                let d =
                    (nx.abs().powf(SQUIRCLE_N) + ny.abs().powf(SQUIRCLE_N)).powf(1.0 / SQUIRCLE_N);
                let dist = (d - 1.0) * r;
                let o = (y as usize * out.w + x as usize) * 3;
                let mut c = [
                    out.data[o] as f32,
                    out.data[o + 1] as f32,
                    out.data[o + 2] as f32,
                ];
                // Soft shadow just outside the edge.
                if dist > -1.0 {
                    let a = 0.35 * (1.0 - (dist / SHADOW).clamp(0.0, 1.0)).powi(2) * ease;
                    for v in &mut c {
                        *v *= 1.0 - a;
                    }
                }
                // Border ring, then the picture inside it (anti-aliased edges).
                let outer = (0.5 - dist).clamp(0.0, 1.0);
                let inner = (0.5 - (dist + BORDER)).clamp(0.0, 1.0);
                if outer > 0.0 {
                    let a = outer * ease;
                    for v in &mut c {
                        *v += a * (235.0 - *v);
                    }
                }
                if inner > 0.0 && i >= 0 && j >= 0 && (i as usize) < p && (j as usize) < p {
                    let a = inner * ease;
                    let q = (j as usize * p + i as usize) * 3;
                    for (k, v) in c.iter_mut().enumerate() {
                        *v += a * (self.patch.data[q + k] as f32 - *v);
                    }
                }
                for k in 0..3 {
                    out.data[o + k] = c[k].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    /// For the status display.
    pub fn describe(&self) -> String {
        let corner = format!("{:?}", self.corner)
            .to_lowercase()
            .replace("left", " left")
            .replace("right", " right");
        match (
            self.on,
            self.seen.is_some_and(|t| t.elapsed().as_secs_f32() < 1.5),
        ) {
            (false, _) => "off".into(),
            (true, true) => format!(
                "on, {corner}, following {}",
                if self.using_pose() {
                    "pose"
                } else {
                    "the matte"
                }
            ),
            (true, false) => format!("on, {corner} (waiting to see you)"),
        }
    }
}

/// The crop's centre and size, each filtered. Size gets far less freedom than position: a breathing zoom is
/// more distracting than a slight lag. What's shown follows the estimate through a critically damped spring, so a
/// correction at each new measurement (~10 Hz) is spread over the following frames instead of landing as a hop.
struct CropFilter {
    k: [Kalman; 3],
    shown: [f32; 3],
    shown_v: [f32; 3],
}

impl CropFilter {
    fn new(z: [f32; 3]) -> Self {
        // Noise scales with the crop: measurements of a big crop (me near the camera) wobble more in pixels.
        let s = z[2];
        let pos = Kalman::new(0.0, (0.5 * s).powi(2), (0.02 * s).powi(2));
        let size = Kalman::new(0.0, (0.3 * s).powi(2), (0.04 * s).powi(2));
        let mut k = [pos.clone(), pos, size];
        for i in 0..3 {
            k[i].x = z[i];
        }
        CropFilter {
            k,
            shown: z,
            shown_v: [0.0; 3],
        }
    }

    /// Spring stiffness (rad/s): settles in ~0.4 s.
    const OMEGA: f32 = 10.0;

    /// Advance one frame: predict the estimate, and move what's shown towards it.
    fn predict(&mut self, dt: f32) {
        self.k.iter_mut().for_each(|k| k.predict(dt));
        let dt = dt.clamp(0.0, 0.1);
        let w = Self::OMEGA;
        for i in 0..3 {
            let a = w * w * (self.k[i].x - self.shown[i]) - 2.0 * w * self.shown_v[i];
            self.shown_v[i] += a * dt;
            self.shown[i] += self.shown_v[i] * dt;
        }
    }

    fn update(&mut self, z: [f32; 3]) {
        for i in 0..3 {
            self.k[i].update(z[i]);
        }
    }

    fn value(&self) -> [f32; 3] {
        self.shown
    }
}

/// [head centre x, head top y, head width] in matte pixels, or None if there's no person.
fn find_head(m: &[f32], w: usize, h: usize) -> Option<[f32; 3]> {
    const ON: f32 = 0.5;
    let row_span = |y: usize| {
        let row = &m[y * w..(y + 1) * w];
        let first = row.iter().position(|&a| a > ON)?;
        let last = row.iter().rposition(|&a| a > ON)?;
        Some((first, last))
    };
    // Top of the person: the first row with a run of a few person pixels (ignores specks).
    let top = (0..h).find(|&y| row_span(y).is_some_and(|(a, b)| b - a >= 3))?;
    // Head: rows from the top down ~ one head height (about 6% of the frame), median width and centre.
    let band = ((h as f32 * 0.06) as usize).max(3);
    let mut widths = Vec::new();
    let mut centres = Vec::new();
    for y in top..(top + band).min(h) {
        if let Some((a, b)) = row_span(y) {
            widths.push((b - a + 1) as f32);
            centres.push((a + b + 1) as f32 / 2.0);
        }
    }
    if widths.len() < 2 {
        return None;
    }
    widths.sort_by(f32::total_cmp);
    centres.sort_by(f32::total_cmp);
    // The widest part of the band is the head proper (the crown is narrower).
    let hw = widths[widths.len() * 3 / 4];
    Some([centres[centres.len() / 2], top as f32, hw])
}

fn bilinear(img: &Rgb, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, (img.w - 1) as f32 - 1e-3);
    let y = y.clamp(0.0, (img.h - 1) as f32 - 1e-3);
    let (x0, y0) = (x as usize, y as usize);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (x1, y1) = ((x0 + 1).min(img.w - 1), (y0 + 1).min(img.h - 1));
    let at = |x: usize, y: usize, c: usize| img.data[(y * img.w + x) * 3 + c] as f32;
    [0, 1, 2].map(|c| {
        let t = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
        let b = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
        t * (1.0 - fy) + b * fy
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stick figure in a 100x160 matte: head a 12-wide disc at x=40, top at y=30; body below.
    fn figure() -> Vec<f32> {
        let (w, h) = (100, 160);
        let mut m = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let head = (x as f32 - 40.0).powi(2) + (y as f32 - 36.0).powi(2) < 36.0;
                let body = y >= 44 && (26..=54).contains(&x);
                if head || body {
                    m[y * w + x] = 1.0;
                }
            }
        }
        m
    }

    /// Simulate: measurements at 10 Hz with noise, frames at 25 Hz. Returns the filtered x per frame.
    fn simulate(
        truth: impl Fn(f32) -> f32,
        noise: f32,
        secs: f32,
        outlier_at: Option<f32>,
    ) -> Vec<(f32, f32)> {
        let mut seed = 7u32;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 2001) as f32 / 1000.0 - 1.0
        };
        let side = 400.0;
        let mut f = CropFilter::new([truth(0.0), 500.0, side]);
        let (mut out, mut next_meas) = (Vec::new(), 0.1);
        let dt = 0.04;
        let mut t = 0.0;
        while t < secs {
            t += dt;
            f.predict(dt);
            if t >= next_meas {
                next_meas += 0.1;
                let mut z = truth(t) + noise * rand();
                if outlier_at.is_some_and(|o| (t - o).abs() < 0.05) {
                    z += 300.0;
                }
                f.update([z, 500.0, side]);
            }
            out.push((t, f.value()[0]));
        }
        out
    }

    #[test]
    fn holds_still_when_i_do() {
        let xs = simulate(|_| 700.0, 8.0, 6.0, None);
        let tail: Vec<f32> = xs
            .iter()
            .filter(|(t, _)| *t > 2.0)
            .map(|(_, x)| *x)
            .collect();
        let max_step = tail
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max);
        let spread = tail.iter().cloned().fold(f32::MIN, f32::max)
            - tail.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max_step < 0.8,
            "frame-to-frame jitter {max_step:.2} px (measurement noise ±8 px)"
        );
        assert!(spread < 6.0, "wanders {spread:.1} px");
    }

    #[test]
    fn follows_a_move_smoothly() {
        // I walk 300 px in a second, then stop.
        let xs = simulate(
            |t| 700.0 + 300.0 * ((t - 1.0).clamp(0.0, 1.0)),
            4.0,
            4.0,
            None,
        );
        let at = |t: f32| xs.iter().find(|(s, _)| *s >= t).unwrap().1;
        assert!(
            (at(3.0) - 1000.0).abs() < 10.0,
            "not there after a second: {}",
            at(3.0)
        );
        let max_step = xs
            .windows(2)
            .map(|w| (w[1].1 - w[0].1).abs())
            .fold(0.0, f32::max);
        assert!(max_step < 20.0, "jumped {max_step:.1} px in one frame");
        let overshoot = xs.iter().map(|(_, x)| x - 1000.0).fold(f32::MIN, f32::max);
        assert!(overshoot < 15.0, "overshot by {overshoot:.1} px");
    }

    #[test]
    fn ignores_a_hand_raised_for_a_moment() {
        let xs = simulate(|_| 700.0, 4.0, 4.0, Some(2.0));
        let worst = xs
            .iter()
            .filter(|(t, _)| *t > 1.0)
            .map(|(_, x)| (x - 700.0).abs())
            .fold(0.0, f32::max);
        assert!(
            worst < 8.0,
            "a one-off 300 px outlier moved it {worst:.1} px"
        );
    }

    #[test]
    fn shrinking_averages_fine_detail_instead_of_aliasing() {
        // A 1-pixel black/white stripe pattern, shrunk ~5x: every inset pixel should come out mid-grey, not a
        // random mix of black and white.
        let (w, h) = (1000, 1000);
        let mut frame = Rgb::new(w, h);
        for y in 0..h {
            for x in 0..w {
                frame.data[(y * w + x) * 3..][..3].fill(if x % 2 == 0 { 0 } else { 255 });
            }
        }
        let mut pip = Pip::new(0.5, Corner::TopLeft, true);
        pip.crop = Some(CropFilter::new([500.0, 500.0, 1000.0]));
        pip.fade = 1.0;
        let mut out = Rgb::new(400, 400); // inset 200 px: 5 camera px per inset px
        pip.draw(&frame, 0, &mut out);
        let p = &pip.patch;
        let (lo, hi) = p
            .data
            .iter()
            .fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            lo > 90 && hi < 165,
            "stripes alias: inset values range {lo}..{hi}"
        );
    }

    #[test]
    fn turned_camera_gives_the_same_inset_as_an_upright_one() {
        // A random camera frame, and the same picture already turned upright (90 degrees clockwise).
        let (w, h) = (300, 200);
        let mut cam = Rgb::new(w, h);
        let mut seed = 99u32;
        for v in cam.data.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = seed as u8;
        }
        let mut upright = Rgb::new(h, w);
        for y in 0..h {
            for x in 0..w {
                // Turning clockwise: camera (x, y) -> upright (h - 1 - y, x).
                let (ux, uy) = (h - 1 - y, x);
                upright.data[(uy * h + ux) * 3..][..3]
                    .copy_from_slice(&cam.data[(y * w + x) * 3..][..3]);
            }
        }
        let inset = |frame: &Rgb, rot: u32| {
            let mut pip = Pip::new(0.2, Corner::TopLeft, true);
            pip.crop = Some(CropFilter::new([100.0, 150.0, 120.0])); // 120 px crop -> 40 px inset: 3x shrink
            pip.fade = 1.0;
            let mut out = Rgb::new(200, 200);
            pip.draw(frame, rot, &mut out);
            pip.patch.data
        };
        assert_eq!(inset(&cam, 90), inset(&upright, 0));
    }

    #[test]
    fn finds_the_head() {
        let [cx, top, hw] = find_head(&figure(), 100, 160).expect("no head");
        assert!((cx - 40.0).abs() < 1.5, "centre {cx}");
        assert!((top - 30.0).abs() < 1.5, "top {top}");
        assert!((9.0..=13.0).contains(&hw), "head width {hw}");
        assert!(find_head(&vec![0.0; 100 * 160], 100, 160).is_none());
    }

    #[test]
    fn inset_is_squircle_with_untouched_corners() {
        let mut frame = Rgb::new(200, 320);
        frame.data.fill(200);
        let mut pip = Pip::new(0.3, Corner::TopLeft, true);
        pip.track(&figure(), 100, 160, 1, (200, 320));
        pip.fade = 1.0;
        pip.last = Instant::now();
        let mut out = Rgb::new(400, 500);
        out.data.fill(40);
        pip.draw(&frame, 0, &mut out);
        let p = 120usize; // 0.3 * 400
        let margin = (400.0f32 * 0.03).round() as usize + SHADOW as usize;
        let at = |x: usize, y: usize| out.data[(y * 400 + x) * 3];
        // Centre shows the picture; the inset's own corner (outside the squircle) is board, maybe shadowed.
        assert_eq!(at(margin + p / 2, margin + p / 2), 200);
        assert!(at(margin + 1, margin + 1) <= 40, "corner was painted");
        // Far from the inset: untouched.
        assert_eq!(at(390, 490), 40);
    }
}
