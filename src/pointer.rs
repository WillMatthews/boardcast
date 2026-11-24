//! A laser pointer on my index finger: the hand tracker finds the fingertip, I map it onto the board through the
//! same perspective transform as the board warp, and draw a red dot there in the board feed. Smoothed with the same
//! Kalman filter as the picture-in-picture (tuned quicker: hands move fast).

use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::geom::Homography;
use crate::image::{Rgb, from_upright};
use crate::track::Kalman;

const COLOUR: [f32; 3] = [255.0, 40.0, 40.0];
/// Display spring stiffness (rad/s): quick, but smooths the ~10 Hz measurements.
const OMEGA: f32 = 22.0;

pub struct Pointer {
    pub on: bool,
    /// Camera pixels -> board canvas pixels.
    to_board: Homography,
    canvas: (usize, usize),
    board: (usize, usize),
    filter: Option<[Kalman; 2]>,
    shown: [f32; 2],
    shown_v: [f32; 2],
    measured: Option<[f32; 2]>,
    found: Option<Instant>,
    fade: f32,
    last: Instant,
}

impl Pointer {
    pub fn new(
        corners: &[[f64; 2]; 4],
        canvas: (usize, usize),
        board: (usize, usize),
        on: bool,
    ) -> Result<Self> {
        Ok(Pointer {
            on,
            to_board: to_board(corners, canvas, board)?,
            canvas,
            board,
            filter: None,
            shown: [0.0; 2],
            shown_v: [0.0; 2],
            measured: None,
            found: None,
            fade: 0.0,
            last: Instant::now(),
        })
    }

    /// The board moved in the camera frame (re-aligned by the tracker).
    pub fn set_corners(&mut self, corners: &[[f64; 2]; 4]) -> Result<()> {
        self.to_board = to_board(corners, self.canvas, self.board)?;
        Ok(())
    }

    /// Point at `p` (upright camera pixels), e.g. my index fingertip from the hand tracker.
    pub fn point_at(&mut self, p: [f32; 2], cam: (usize, usize), rot: u32) {
        let [cx, cy] = from_upright([p[0] as f64, p[1] as f64], cam, rot);
        let [bx, by] = self.to_board.apply(cx, cy);
        // Only over the board.
        let ox = (self.canvas.0 - self.board.0) as f64 / 2.0;
        let oy = (self.canvas.1 - self.board.1) as f64 / 2.0;
        if bx < ox || by < oy || bx > ox + self.board.0 as f64 || by > oy + self.board.1 as f64 {
            return;
        }
        self.found = Some(Instant::now());
        self.measured = Some([bx as f32, by as f32]);
    }

    pub fn draw(&mut self, out: &mut Rgb) {
        let dt = self.last.elapsed().as_secs_f32().clamp(0.0, 0.1);
        self.last = Instant::now();
        let present = self.found.is_some_and(|t| t.elapsed().as_secs_f32() < 0.4);
        // Quick to appear, a little slower to go.
        let goal = if self.on && present { 1.0 } else { 0.0 };
        self.fade = if goal > self.fade {
            (self.fade + dt / 0.15).min(1.0)
        } else {
            (self.fade - dt / 0.3).max(0.0)
        };

        if let Some(z) = self.measured.take() {
            match &mut self.filter {
                Some(f) if self.fade > 0.0 => {
                    for i in 0..2 {
                        f[i].predict(dt);
                        f[i].update(z[i]);
                    }
                }
                // First sighting (or back after fading out): start where the hand is, don't sweep across.
                _ => {
                    let r = 6.0f32.powi(2);
                    let q = (4.0 * self.board.1 as f32).powi(2);
                    self.filter = Some([Kalman::new(z[0], q, r), Kalman::new(z[1], q, r)]);
                    self.shown = z;
                    self.shown_v = [0.0; 2];
                }
            }
        } else if let Some(f) = &mut self.filter {
            f.iter_mut().for_each(|k| k.predict(dt));
        }
        let Some(f) = &self.filter else { return };
        for i in 0..2 {
            let a = OMEGA * OMEGA * (f[i].x - self.shown[i]) - 2.0 * OMEGA * self.shown_v[i];
            self.shown_v[i] += a * dt;
            self.shown[i] += self.shown_v[i] * dt;
        }
        if self.fade <= 0.0 {
            return;
        }
        dot(out, self.shown, self.fade);
    }

    pub fn describe(&self) -> String {
        match (
            self.on,
            self.found.is_some_and(|t| t.elapsed().as_secs_f32() < 0.4),
        ) {
            (false, _) => "off".into(),
            (true, true) => "on, on the board".into(),
            (true, false) => "on (not pointing at the board)".into(),
        }
    }
}

/// A red dot with a soft glow, centred at `c` (canvas pixels).
fn dot(out: &mut Rgb, c: [f32; 2], fade: f32) {
    let r = (out.w.min(out.h) as f32 * 0.011).max(4.0);
    let glow = r * 3.0;
    let (x0, x1) = (
        (c[0] - glow).floor().max(0.0) as usize,
        ((c[0] + glow).ceil() as usize).min(out.w),
    );
    let (y0, y1) = (
        (c[1] - glow).floor().max(0.0) as usize,
        ((c[1] + glow).ceil() as usize).min(out.h),
    );
    for y in y0..y1 {
        for x in x0..x1 {
            let d = ((x as f32 + 0.5 - c[0]).powi(2) + (y as f32 + 0.5 - c[1]).powi(2)).sqrt();
            let core = (r + 0.5 - d).clamp(0.0, 1.0);
            let halo = 0.45 * (1.0 - (d / glow).clamp(0.0, 1.0)).powi(2);
            let a = core.max(halo) * fade;
            if a <= 0.0 {
                continue;
            }
            // A bright centre so it reads as a light, not a red blob.
            let centre = (1.0 - d / (0.5 * r)).clamp(0.0, 1.0) * 0.7;
            let i = (y * out.w + x) * 3;
            for k in 0..3 {
                let col = COLOUR[k] + centre * (255.0 - COLOUR[k]);
                let v = out.data[i + k] as f32;
                out.data[i + k] = (v + a * (col - v)).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Camera pixels -> board canvas pixels: the inverse of the board warp (the board centred in the canvas).
fn to_board(
    corners: &[[f64; 2]; 4],
    (cw, ch): (usize, usize),
    (bw, bh): (usize, usize),
) -> Result<Homography> {
    let (ox, oy) = (((cw - bw) / 2) as f64, ((ch - bh) / 2) as f64);
    let (bw, bh) = (bw as f64, bh as f64);
    let rect = [[ox, oy], [ox + bw, oy], [ox + bw, oy + bh], [ox, oy + bh]];
    Homography::from_points(corners, &rect).ok_or_else(|| anyhow!("board corners are degenerate"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORNERS: [[f64; 2]; 4] = [
        [200.0, 100.0],
        [800.0, 120.0],
        [780.0, 700.0],
        [220.0, 680.0],
    ];

    #[test]
    fn camera_corners_land_on_the_board_corners() {
        let h = to_board(&CORNERS, (500, 400), (300, 400)).unwrap();
        let [x, y] = h.apply(800.0, 120.0); // top-right of the board in the camera
        assert!((x - 400.0).abs() < 1e-6 && y.abs() < 1e-6, "{x} {y}");
    }

    #[test]
    fn dot_goes_where_i_point_and_only_over_the_board() {
        // Upright camera (rot 0): camera pixels are upright pixels.
        let mut p = Pointer::new(&CORNERS, (600, 600), (600, 600), true).unwrap();
        p.point_at([514.0, 500.0], (1000, 800), 0);
        let z = p.measured.expect("a point over the board wasn't used");
        let want = to_board(&CORNERS, (600, 600), (600, 600))
            .unwrap()
            .apply(514.0, 500.0);
        assert!((z[0] as f64 - want[0]).abs() < 0.01 && (z[1] as f64 - want[1]).abs() < 0.01);
        p.point_at([40.0, 40.0], (1000, 800), 0);
        assert!(
            p.measured.is_some_and(|m| m == z),
            "a point off the board moved the dot"
        );
    }
}
