//! Find the board's edges in a camera frame, starting from the current corners.
//!
//! For points spaced along each edge of the current quad, look across the edge (along its normal) for the
//! strongest step from dark board (inside) to bright frame/wall (outside), using box averages either side so a
//! thin chalk stroke near the edge can't win. Fit a line to each edge's samples with RANSAC (sticky notes on the
//! edge, a person in front of it, glare are outliers), then intersect adjacent lines for the corners.

use crate::image::Rgb;

pub type Corners = [[f64; 2]; 4];

/// Samples per edge, and the fraction of each end skipped (corners are often shadowed or occluded).
const SAMPLES: usize = 48;
const END_SKIP: f64 = 0.06;
/// Box width either side of a candidate edge position.
const BOX: i32 = 6;
/// Minimum brightness step (luma, 0-255) for a sample to count.
const MIN_STEP: f32 = 12.0;
/// RANSAC inlier distance in pixels.
const INLIER: f64 = 1.5;

pub struct Fit {
    pub corners: Corners,
    /// Per edge (top, right, bottom, left): fraction of samples that agree with the fitted line.
    pub support: [f64; 4],
}

impl Fit {
    pub fn min_support(&self) -> f64 {
        self.support.iter().cloned().fold(1.0, f64::min)
    }
}

/// Bilinear luma sample, clamped to the frame (a wide search near the frame edge just sees edge pixels).
fn luma_at(img: &Rgb, x: f64, y: f64) -> Option<f32> {
    let x = x.clamp(0.0, (img.w - 2) as f64);
    let y = y.clamp(0.0, (img.h - 2) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let l = |x: usize, y: usize| {
        let i = (y * img.w + x) * 3;
        0.299 * img.data[i] as f32 + 0.587 * img.data[i + 1] as f32 + 0.114 * img.data[i + 2] as f32
    };
    let top = l(x0, y0) * (1.0 - fx) + l(x0 + 1, y0) * fx;
    let bot = l(x0, y0 + 1) * (1.0 - fx) + l(x0 + 1, y0 + 1) * fx;
    Some(top * (1.0 - fy) + bot * fy)
}

/// Find the edge offset (pixels along the outward normal) at point `p`, within ±`search`.
fn edge_offset(img: &Rgb, p: [f64; 2], out: [f64; 2], search: i32) -> Option<f64> {
    let n = 2 * (search + BOX) + 1;
    let mut prof = Vec::with_capacity(n as usize);
    for k in -(search + BOX)..=(search + BOX) {
        prof.push(luma_at(
            img,
            p[0] + out[0] * k as f64,
            p[1] + out[1] * k as f64,
        )?);
    }
    // score[o] = mean of BOX px just outside minus mean of BOX px just inside the boundary at o.
    let at = |o: i32| (o + search + BOX) as usize;
    let score = |o: i32| -> f32 {
        let inside: f32 = (o - BOX..o).map(|k| prof[at(k)]).sum();
        let outside: f32 = (o..o + BOX).map(|k| prof[at(k)]).sum();
        (outside - inside) / BOX as f32
    };
    let (best, s) = (-search + 1..search)
        .map(|o| (o, score(o)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if s < MIN_STEP {
        return None;
    }
    // Parabolic sub-pixel refinement.
    let (l, r) = (score(best - 1), score(best + 1));
    let den = l - 2.0 * s + r;
    let frac = if den.abs() > 1e-6 {
        (0.5 * (l - r) / den).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    // The boundary between the inside box [o-BOX, o) and outside box [o, o+BOX) sits at o - 0.5.
    Some(best as f64 + frac as f64 - 0.5)
}

/// RANSAC line fit of offsets d over positions t (both 1-D): returns (a, b) with d ≈ a + b t, and support.
fn fit_line(pts: &[(f64, f64)]) -> Option<(f64, f64, usize)> {
    if pts.len() < 4 {
        return None;
    }
    let mut best: Option<(Vec<usize>, f64, f64)> = None;
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let ((t0, d0), (t1, d1)) = (pts[i], pts[j]);
            if (t1 - t0).abs() < 0.1 {
                continue;
            }
            let b = (d1 - d0) / (t1 - t0);
            let a = d0 - b * t0;
            let inl: Vec<usize> = (0..pts.len())
                .filter(|&k| (pts[k].1 - a - b * pts[k].0).abs() < INLIER)
                .collect();
            if best.as_ref().is_none_or(|(bi, ..)| inl.len() > bi.len()) {
                best = Some((inl, a, b));
            }
        }
    }
    let (inl, ..) = best?;
    // Least squares on the inliers.
    let n = inl.len() as f64;
    let (st, sd) = inl
        .iter()
        .fold((0.0, 0.0), |(st, sd), &k| (st + pts[k].0, sd + pts[k].1));
    let (mt, md) = (st / n, sd / n);
    let (mut stt, mut std) = (0.0, 0.0);
    for &k in &inl {
        stt += (pts[k].0 - mt).powi(2);
        std += (pts[k].0 - mt) * (pts[k].1 - md);
    }
    let b = if stt > 1e-9 { std / stt } else { 0.0 };
    Some((md - b * mt, b, inl.len()))
}

fn intersect((p, d): ([f64; 2], [f64; 2]), (q, e): ([f64; 2], [f64; 2])) -> Option<[f64; 2]> {
    let den = d[0] * e[1] - d[1] * e[0];
    if den.abs() < 1e-9 {
        return None;
    }
    let s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den;
    Some([p[0] + s * d[0], p[1] + s * d[1]])
}

/// One refinement pass: search ±`search` px around each edge of `c` (TL, TR, BR, BL).
pub fn refine(img: &Rgb, c: &Corners, search: i32) -> Option<Fit> {
    let centre = [
        (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0,
        (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0,
    ];
    let mut lines = [([0.0; 2], [0.0; 2]); 4];
    let mut support = [0.0; 4];
    for e in 0..4 {
        let (p, q) = (c[e], c[(e + 1) % 4]);
        let dir = [q[0] - p[0], q[1] - p[1]];
        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let u = [dir[0] / len, dir[1] / len];
        // Outward normal: perpendicular to the edge, pointing away from the quad's centre.
        let mut out = [u[1], -u[0]];
        let mid = [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0];
        if (mid[0] - centre[0]) * out[0] + (mid[1] - centre[1]) * out[1] < 0.0 {
            out = [-out[0], -out[1]];
        }
        let pts: Vec<(f64, f64)> = (0..SAMPLES)
            .filter_map(|i| {
                let t = END_SKIP + (1.0 - 2.0 * END_SKIP) * i as f64 / (SAMPLES - 1) as f64;
                let s = [p[0] + dir[0] * t, p[1] + dir[1] * t];
                edge_offset(img, s, out, search).map(|d| (t, d))
            })
            .collect();
        let (a, b, n) = fit_line(&pts)?;
        support[e] = n as f64 / SAMPLES as f64;
        let p2 = [p[0] + out[0] * a, p[1] + out[1] * a];
        let q2 = [q[0] + out[0] * (a + b), q[1] + out[1] * (a + b)];
        lines[e] = (p2, [q2[0] - p2[0], q2[1] - p2[1]]);
    }
    // Corner i is where edge i-1 (ending at it) meets edge i (starting at it).
    let mut corners = [[0.0; 2]; 4];
    for i in 0..4 {
        corners[i] = intersect(lines[(i + 3) % 4], lines[i])?;
    }
    Some(Fit { corners, support })
}

/// Coarse-to-fine: a wide search, then narrow ones around the result.
pub fn calibrate(img: &Rgb, start: &Corners, search: i32) -> Option<Fit> {
    let mut fit = refine(img, start, search)?;
    for s in [search.min(24), 6] {
        fit = refine(img, &fit.corners, s)?;
    }
    Some(fit)
}

/// Watches for the board moving in the frame (a bumped camera) while streaming.
///
/// Every `PERIOD` it refines the corners with a narrow search. A move is accepted only when `AGREE` checks in a
/// row are well supported on all four edges, agree with each other, and differ from the current corners by more
/// than `MOVE`: a person in front of an edge, or noise, doesn't trigger it.
pub struct Tracker {
    pub corners: Corners,
    last: std::time::Instant,
    recent: Vec<Corners>,
    /// Running sum of the frames averaged for the current check, and how many so far.
    sum: Vec<u16>,
    summed: usize,
    avg: Option<Rgb>,
    /// Consecutive checks without a well-supported fit (big move, or something in front of the board).
    pub failures: usize,
    /// When the last check finished, and how far its fit was from the current corners (None: no fit).
    pub checked: Option<std::time::Instant>,
    pub shift: Option<f64>,
}

impl Tracker {
    const PERIOD: std::time::Duration = std::time::Duration::from_secs(2);
    const SEARCH: i32 = 15;
    const SUPPORT: f64 = 0.5;
    const AGREE: usize = 3;
    const MOVE: f64 = 2.0;
    /// Frames averaged per check (single frames are too noisy where the frame edge is in shadow).
    const AVERAGE: usize = 8;

    pub fn new(corners: Corners) -> Self {
        Tracker {
            corners,
            last: std::time::Instant::now(),
            recent: Vec::new(),
            sum: Vec::new(),
            summed: 0,
            avg: None,
            failures: 0,
            checked: None,
            shift: None,
        }
    }

    /// Call every frame; returns new corners when the board has moved.
    pub fn check(&mut self, frame: &Rgb) -> Option<Corners> {
        if self.last.elapsed() < Self::PERIOD {
            return None;
        }
        if self.summed == 0 {
            self.sum.clear();
            self.sum.resize(frame.data.len(), 0);
        }
        for (s, &v) in self.sum.iter_mut().zip(&frame.data) {
            *s += v as u16;
        }
        self.summed += 1;
        if self.summed < Self::AVERAGE {
            return None;
        }
        let avg = self.avg.get_or_insert_with(|| Rgb::new(frame.w, frame.h));
        let n = self.summed as u16;
        for (a, &s) in avg.data.iter_mut().zip(&self.sum) {
            *a = ((s + n / 2) / n) as u8;
        }
        self.summed = 0;
        self.last = std::time::Instant::now();
        let frame = &*avg;
        // Wide enough to catch a bump, then narrow so writing near an edge can't compete with it.
        let fit =
            refine(frame, &self.corners, Self::SEARCH).and_then(|f| refine(frame, &f.corners, 6));
        if std::env::var_os("BOARDVU_DEBUG_TRACK").is_some() {
            match &fit {
                Some(f) => crate::say!(
                    "track: shift {:.1} px, support {:?}",
                    max_shift(&f.corners, &self.corners),
                    f.support.map(|s| (s * 100.0).round() / 100.0)
                ),
                None => crate::say!("track: no fit"),
            }
        }
        self.checked = Some(std::time::Instant::now());
        self.shift = fit
            .as_ref()
            .filter(|f| f.min_support() >= Self::SUPPORT)
            .map(|f| max_shift(&f.corners, &self.corners));
        match fit {
            Some(fit) if fit.min_support() >= Self::SUPPORT => {
                self.failures = 0;
                self.recent.push(fit.corners);
            }
            _ => {
                self.recent.clear();
                self.failures += 1;
                if self.failures == 15 {
                    crate::say!(
                        "can't find the board's edges near where they should be (30 s). If the camera moved \
                         a lot, stop and run `boardvu calibrate`."
                    );
                }
                return None;
            }
        }
        if self.recent.len() > Self::AGREE {
            self.recent.remove(0);
        }
        if self.recent.len() < Self::AGREE {
            return None;
        }
        let mut mean = [[0.0; 2]; 4];
        for c in &self.recent {
            for i in 0..4 {
                for k in 0..2 {
                    mean[i][k] += c[i][k] / Self::AGREE as f64;
                }
            }
        }
        let consistent = self.recent.iter().all(|c| max_shift(c, &mean) < 1.0);
        let plausible = check_plausible(&mean, (frame.w, frame.h), &self.corners).is_ok();
        if consistent && plausible && max_shift(&mean, &self.corners) > Self::MOVE {
            self.corners = mean;
            self.recent.clear();
            return Some(mean);
        }
        None
    }
}

/// Reject outlines that can't be the board: corners outside the camera's view (the board doesn't fit in the
/// frame, so an edge can't really have been seen), a non-convex or badly skewed quad, or a big change of shape
/// from `prev`. Without this, a strong straight chalk line can pass for a missing edge.
pub fn check_plausible(c: &Corners, (w, h): (usize, usize), prev: &Corners) -> Result<(), String> {
    const MARGIN: f64 = 3.0;
    const NAMES: [&str; 4] = ["top-left", "top-right", "bottom-right", "bottom-left"];
    for (i, p) in c.iter().enumerate() {
        if p[0] < MARGIN || p[1] < MARGIN || p[0] > w as f64 - MARGIN || p[1] > h as f64 - MARGIN {
            return Err(format!(
                "the {} corner ({:.0},{:.0}) is outside the camera's view: the whole board must be in the frame",
                NAMES[i], p[0], p[1]
            ));
        }
    }
    for i in 0..4 {
        let (a, b, d) = (c[(i + 3) % 4], c[i], c[(i + 1) % 4]);
        let (u, v) = ([a[0] - b[0], a[1] - b[1]], [d[0] - b[0], d[1] - b[1]]);
        let cross = u[0] * v[1] - u[1] * v[0];
        let dot = u[0] * v[0] + u[1] * v[1];
        let angle = cross.abs().atan2(dot).to_degrees();
        if cross > 0.0 || !(55.0..=125.0).contains(&angle) {
            return Err(format!(
                "the outline isn't board-shaped (angle at the {} corner is {angle:.0}°)",
                NAMES[i]
            ));
        }
    }
    let side = |c: &Corners, i: usize| {
        let (a, b) = (c[i], c[(i + 1) % 4]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    };
    for i in 0..4 {
        let r = side(c, i) / side(prev, i).max(1.0);
        if !(0.6..=1.6).contains(&r) {
            return Err(format!(
                "an edge changed length by {r:.2}x from before; that's not the same board view"
            ));
        }
    }
    Ok(())
}

pub fn max_shift(a: &Corners, b: &Corners) -> f64 {
    (0..4)
        .map(|i| ((a[i][0] - b[i][0]).powi(2) + (a[i][1] - b[i][1]).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Homography;

    const TRUE: Corners = [
        [300.0, 120.0],
        [880.0, 70.0],
        [860.0, 760.0],
        [310.0, 700.0],
    ];

    /// Dark board in a bright frame on a mid wall, with chalk strokes near the edges, sticky notes overlapping
    /// the right edge, a "person" covering part of the bottom edge, and noise.
    fn scene() -> Rgb {
        let (w, h) = (1200, 900);
        let mut img = Rgb::new(w, h);
        // Board coordinates (u, v) in [0,1]^2 for each pixel via the inverse homography.
        let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let inv = Homography::from_points(&TRUE, &unit).unwrap();
        let mut seed = 12345u32;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 9) as f32 - 4.0
        };
        for y in 0..h {
            for x in 0..w {
                let [u, v] = inv.apply(x as f64 + 0.5, y as f64 + 0.5);
                let mut c = [150.0, 150.0, 140.0]; // wall
                let frame = 0.03;
                if (-frame..1.0 + frame).contains(&u) && (-frame..1.0 + frame).contains(&v) {
                    c = [200.0, 200.0, 205.0]; // aluminium frame
                }
                if (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v) {
                    c = [45.0, 95.0, 75.0]; // board
                    // chalk strokes parallel to edges, just inside
                    if (0.02..0.025).contains(&u)
                        || (0.03..0.035).contains(&v)
                        || (0.97..0.975).contains(&v)
                    {
                        c = [220.0, 225.0, 220.0];
                    }
                }
                // sticky notes over the right edge
                if (0.9..1.04).contains(&u)
                    && ((0.1..0.2).contains(&v)
                        || (0.25..0.35).contains(&v)
                        || (0.5..0.6).contains(&v))
                {
                    c = [235.0, 225.0, 130.0];
                }
                // a person in front of the bottom-left
                if (x as f64 - 380.0).powi(2) + (y as f64 - 800.0).powi(2) < 140.0f64.powi(2) {
                    c = [180.0, 130.0, 110.0];
                }
                let n = noise();
                let i = (y * w + x) * 3;
                for k in 0..3 {
                    img.data[i + k] = (c[k] + n).clamp(0.0, 255.0) as u8;
                }
            }
        }
        img
    }

    #[test]
    fn recovers_corners_after_a_bump() {
        let img = scene();
        // The camera "moved": every corner is off by up to ~60 px.
        let start: Corners = [
            [340.0, 90.0],
            [850.0, 110.0],
            [900.0, 720.0],
            [280.0, 740.0],
        ];
        let fit = calibrate(&img, &start, 120).expect("no fit");
        let err = max_shift(&fit.corners, &TRUE);
        assert!(
            err < 1.5,
            "corners off by {err:.2} px: {:?} (support {:?})",
            fit.corners,
            fit.support
        );
        assert!(fit.min_support() > 0.4, "support {:?}", fit.support);
    }

    /// One full check: skip the 2 s wait, feed the frames it averages.
    fn check_now(t: &mut Tracker, img: &Rgb) -> Option<Corners> {
        t.last = std::time::Instant::now() - Tracker::PERIOD;
        for _ in 0..Tracker::AVERAGE - 1 {
            assert!(t.check(img).is_none());
        }
        t.check(img)
    }

    #[test]
    fn tracker_realigns_after_agreeing_checks_and_ignores_a_still_board() {
        let img = scene();
        let mut still = Tracker::new(TRUE);
        for _ in 0..5 {
            assert!(
                check_now(&mut still, &img).is_none(),
                "moved without a move"
            );
        }
        let mut bumped = Tracker::new(TRUE.map(|[x, y]| [x + 6.0, y - 4.0]));
        assert!(check_now(&mut bumped, &img).is_none() && check_now(&mut bumped, &img).is_none());
        let c = check_now(&mut bumped, &img).expect("didn't re-align after 3 agreeing checks");
        assert!(max_shift(&c, &TRUE) < 1.5, "{c:?}");
        assert!(check_now(&mut bumped, &img).is_none(), "re-aligned twice");
    }

    #[test]
    fn implausible_outlines_are_rejected() {
        let frame = (1200, 900);
        assert!(check_plausible(&TRUE, frame, &TRUE).is_ok());
        // A corner above the image (the board's top is out of view).
        let off = [
            [300.0, -200.0],
            [880.0, 70.0],
            [860.0, 760.0],
            [310.0, 700.0],
        ];
        assert!(
            check_plausible(&off, frame, &TRUE)
                .unwrap_err()
                .contains("outside the camera's view")
        );
        // A chalk diagonal mistaken for the top edge: badly skewed.
        let skew = [
            [300.0, 20.0],
            [880.0, 500.0],
            [860.0, 760.0],
            [310.0, 700.0],
        ];
        assert!(check_plausible(&skew, frame, &TRUE).is_err());
        // Twisted (corners out of order).
        let twisted = [TRUE[1], TRUE[0], TRUE[2], TRUE[3]];
        assert!(check_plausible(&twisted, frame, &TRUE).is_err());
    }

    #[test]
    fn small_drift_is_tracked_by_one_narrow_pass() {
        let img = scene();
        let start: Corners = TRUE.map(|[x, y]| [x + 6.0, y - 4.0]);
        let fit = refine(&img, &start, 15).expect("no fit");
        assert!(max_shift(&fit.corners, &TRUE) < 1.5, "{:?}", fit.corners);
    }
}
