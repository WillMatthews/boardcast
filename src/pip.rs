//! Picture-in-picture: a squircle crop of me, following my head, inset in a corner of the board feed.
//!
//! The head is found from the person matte (segmentation frame): the top of the matte, with the head's width from
//! the first rows below it. The crop is a square a few head-widths across, so it frames me the same at the board
//! or at the desk, and is smoothed so it glides rather than jitters.

use std::time::Instant;

use crate::image::{Rgb, from_upright, upright_size};

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
/// Smoothing per matte update (~10 Hz): fraction of the way to the new crop.
const FOLLOW: f32 = 0.25;
const BORDER: f32 = 3.0;
const SHADOW: f32 = 10.0;

pub struct Pip {
    pub on: bool,
    pub corner: Corner,
    /// Inset size as a fraction of the canvas width.
    size: f32,
    /// Crop in upright camera pixels: centre x, centre y, side. None until I've been seen.
    crop: Option<[f32; 3]>,
    /// 0..1: fades in when turned on / I'm seen, out when turned off / I'm gone.
    fade: f32,
    seen: Option<Instant>,
    last: Instant,
    /// Version of the matte the crop was last updated from (update once per new matte).
    matte_seen: u64,
    patch: Rgb,
}

impl Pip {
    pub fn new(size: f32, corner: Corner, on: bool) -> Self {
        Pip {
            on,
            corner,
            size,
            crop: None,
            fade: 0.0,
            seen: None,
            last: Instant::now(),
            matte_seen: u64::MAX,
            patch: Rgb::new(1, 1),
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
        let Some([cx, top, hw]) = find_head(matte, mw, mh) else {
            return;
        };
        let k = upright.0 as f32 / mw as f32;
        let side =
            (hw * k * HEAD_WIDTHS).clamp(upright.0 as f32 * 0.12, upright.0.min(upright.1) as f32);
        let target = [cx * k, top * k - HEAD_TOP * side + side / 2.0, side];
        self.seen = Some(Instant::now());
        self.crop = Some(match self.crop {
            None => target,
            Some(c) => [0, 1, 2].map(|i| c[i] + FOLLOW * (target[i] - c[i])),
        });
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
        let Some([cx, cy, side]) = self.crop else {
            return;
        };
        if self.fade <= 0.0 {
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
        let k = side / p as f32;
        for j in 0..p {
            for i in 0..p {
                let u = x0 + (i as f32 + 0.5) * k;
                let v = y0 + (j as f32 + 0.5) * k;
                let [x, y] = from_upright([u as f64, v as f64], (frame.w, frame.h), rot);
                let px = bilinear(frame, x as f32 - 0.5, y as f32 - 0.5);
                let o = (j * p + i) * 3;
                self.patch.data[o..o + 3].copy_from_slice(&px);
            }
        }

        // Composite: soft shadow, light border, squircle-masked picture.
        let margin = (out.w.min(out.h) as f32 * 0.03).round() as isize + SHADOW as isize;
        let (ox, oy) = match self.corner {
            Corner::TopLeft => (margin, margin),
            Corner::TopRight => (out.w as isize - margin - p as isize, margin),
            Corner::BottomRight => (
                out.w as isize - margin - p as isize,
                out.h as isize - margin - p as isize,
            ),
            Corner::BottomLeft => (margin, out.h as isize - margin - p as isize),
        };
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
            (true, true) => format!("on, {corner}"),
            (true, false) => format!("on, {corner} (waiting to see you)"),
        }
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

fn bilinear(img: &Rgb, x: f32, y: f32) -> [u8; 3] {
    let x = x.clamp(0.0, (img.w - 1) as f32 - 1e-3);
    let y = y.clamp(0.0, (img.h - 1) as f32 - 1e-3);
    let (x0, y0) = (x as usize, y as usize);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (x1, y1) = ((x0 + 1).min(img.w - 1), (y0 + 1).min(img.h - 1));
    let at = |x: usize, y: usize, c: usize| img.data[(y * img.w + x) * 3 + c] as f32;
    [0, 1, 2].map(|c| {
        let t = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
        let b = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
        (t * (1.0 - fy) + b * fy).round() as u8
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
