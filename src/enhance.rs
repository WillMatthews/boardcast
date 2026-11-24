//! Chalk enhancement: flatten the board's uneven lighting and make chalk stand out.
//!
//! Two estimates, from luma `Y` and colour:
//! - **Strokes** (every frame): the white top-hat `t = Y - opening(Y)`. A morphological opening (min filter then
//!   max filter, radius `STROKE_R`) removes bright features thinner than about `2 * STROKE_R + 1` pixels, i.e.
//!   chalk strokes, but keeps big bright areas (sticky notes, chalk-dust haze, lighting). So `t` is chalk only.
//! - **Lighting** (refreshed every `REFRESH` frames): per-channel median over `CELL`-pixel cells, then a 7x7-cell
//!   median (robust to writing and sticky notes), a blur, and bilinear upsampling.
//!
//! Looks:
//! Lighting multiplies brightness, so both looks divide by it: `t` is rescaled to the board's average lighting.
//! - `Natural`: an even board colour plus each pixel's difference from the local board, scaled by the lighting
//!   gain (capped at `MAX_GAIN`), plus boosted strokes. Colours kept.
//! - `Paper`: dark ink on white, ink strength from `t` between an adaptive noise floor and a high percentile.
//!   Only chalk survives; sticky notes and haze become paper.

use rayon::prelude::*;

use crate::image::{ROWS_PER_TASK, Rgb};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Look {
    /// The deskewed board as the camera sees it.
    Plain,
    /// Even lighting, boosted chalk, colours kept.
    Natural,
    /// Dark ink on white paper.
    Paper,
}

impl Look {
    pub fn next(self) -> Self {
        match self {
            Look::Plain => Look::Natural,
            Look::Natural => Look::Paper,
            Look::Paper => Look::Plain,
        }
    }
}

/// Opening radius: strokes up to ~2 * STROKE_R + 1 px wide count as chalk.
const STROKE_R: usize = 5;
/// Lighting grid cell size in pixels.
const CELL: usize = 16;
/// Recompute lighting and noise statistics every this many frames.
const REFRESH: u32 = 15;
/// Most the lighting correction may amplify a pixel's difference from the local board (dark corners).
const MAX_GAIN: f32 = 3.0;
/// Natural: the board's brightness (luma) after flattening.
const BOARD_LUMA: f32 = 80.0;
/// Natural: how much extra brightness strokes get, per unit of top-hat above the noise floor.
const STROKE_BOOST: f32 = 0.8;

pub struct Enhancer {
    pub look: Look,
    w: usize,
    h: usize,
    /// Per-pixel lighting colour.
    light: Vec<f32>,
    /// The even board colour Natural flattens to.
    board: [f32; 3],
    /// Top-hat noise floor and Paper's full-ink level.
    floor: f32,
    hi: f32,
    y: Vec<f32>,
    tmp: Vec<f32>,
    open: Vec<f32>,
    frames: u32,
}

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

impl Enhancer {
    pub fn new(look: Look, w: usize, h: usize) -> Self {
        let n = w * h;
        Enhancer {
            look,
            w,
            h,
            light: vec![0.0; n * 3],
            board: [0.0; 3],
            floor: 6.0,
            hi: 40.0,
            y: vec![0.0; n],
            tmp: vec![0.0; n],
            open: vec![0.0; n],
            frames: 0,
        }
    }

    /// Enhance `src` into `out` (same size). Plain copies.
    pub fn apply(&mut self, src: &Rgb, out: &mut Rgb) {
        assert_eq!((src.w, src.h), (self.w, self.h));
        if self.look == Look::Plain {
            out.data.copy_from_slice(&src.data);
            return;
        }
        let (w, h) = (self.w, self.h);
        self.y
            .par_iter_mut()
            .zip(src.data.par_chunks(3))
            .with_min_len(4096)
            .for_each(|(y, p)| {
                *y = luma(p[0] as f32, p[1] as f32, p[2] as f32);
            });
        // Opening = dilate(erode(Y)), each a separable square filter.
        minmax(&self.y, &mut self.open, w, h, STROKE_R, false, true);
        minmax(&self.open, &mut self.tmp, w, h, STROKE_R, false, false);
        minmax(&self.tmp, &mut self.open, w, h, STROKE_R, true, true);
        minmax(&self.open, &mut self.tmp, w, h, STROKE_R, true, false);
        std::mem::swap(&mut self.open, &mut self.tmp);
        if self.frames % REFRESH == 0 {
            self.refresh(src);
        }
        self.frames = self.frames.wrapping_add(1);

        let (look, board, floor, hi) = (self.look, self.board, self.floor, self.hi);
        let board_y = BOARD_LUMA;
        out.data
            .par_chunks_mut(w * 3)
            .zip(src.data.par_chunks(w * 3))
            .zip(self.light.par_chunks(w * 3))
            .zip(self.y.par_chunks(w))
            .zip(self.open.par_chunks(w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|((((out, src), light), y), open)| {
                for x in 0..w {
                    let i = x * 3;
                    let s = [src[i] as f32, src[i + 1] as f32, src[i + 2] as f32];
                    let l = [light[i], light[i + 1], light[i + 2]];
                    let ly = luma(l[0], l[1], l[2]).max(1.0);
                    // Lighting multiplies brightness, so normalise by it (the top-hat as it would be under the
                    // reference lighting), capped so nearly unlit corners don't turn into amplified noise.
                    let g = (board_y / ly).min(MAX_GAIN);
                    let t = ((y[x] - open[x]) * g - floor).max(0.0);
                    match look {
                        Look::Natural => {
                            for c in 0..3 {
                                // Bare board -> exactly the board colour; deviations (chalk) scaled by the gain.
                                let v = board[c] + g * (s[c] - l[c]) + STROKE_BOOST * t;
                                out[i + c] = v.clamp(0.0, 255.0) as u8;
                            }
                        }
                        Look::Paper => {
                            let v = (t / (hi - floor)).clamp(0.0, 1.0).powf(0.8);
                            const PAPER: [f32; 3] = [250.0, 250.0, 246.0];
                            for c in 0..3 {
                                // Near-black ink, tinted by the chalk's colour: how its colour differs from the
                                // board's here, beyond the brightness difference (white chalk -> neutral).
                                let tint = (s[c] - l[c]) - (y[x] - ly);
                                let ink = (30.0 + 1.5 * tint).clamp(0.0, 200.0);
                                out[i + c] = (PAPER[c] * (1.0 - v) + ink * v) as u8;
                            }
                        }
                        Look::Plain => unreachable!(),
                    }
                }
            });
    }

    /// Recompute the lighting estimate, the board colour, and the top-hat noise floor / ink level.
    fn refresh(&mut self, src: &Rgb) {
        let (w, h) = (self.w, self.h);
        let (gw, gh) = (w.div_ceil(CELL), h.div_ceil(CELL));
        // Per-cell, per-channel median.
        let cells: Vec<[f32; 3]> = (0..gw * gh)
            .into_par_iter()
            .map(|ci| {
                let (cx, cy) = (ci % gw, ci / gw);
                let mut ch: [Vec<u8>; 3] = Default::default();
                for y in cy * CELL..((cy + 1) * CELL).min(h) {
                    for x in cx * CELL..((cx + 1) * CELL).min(w) {
                        let i = (y * w + x) * 3;
                        for c in 0..3 {
                            ch[c].push(src.data[i + c]);
                        }
                    }
                }
                ch.map(|mut v| {
                    let m = v.len() / 2;
                    *v.select_nth_unstable(m).1 as f32
                })
            })
            .collect();
        // 7x7-cell median per channel: sticky notes and dense writing cover well under half of it.
        let at = |g: &[[f32; 3]], x: isize, y: isize| {
            g[y.clamp(0, gh as isize - 1) as usize * gw + x.clamp(0, gw as isize - 1) as usize]
        };
        let med: Vec<[f32; 3]> = (0..gw * gh)
            .into_par_iter()
            .map(|ci| {
                let (cx, cy) = ((ci % gw) as isize, (ci / gw) as isize);
                let mut out = [0.0; 3];
                for c in 0..3 {
                    let mut v: Vec<f32> = Vec::with_capacity(49);
                    for dy in -3..=3 {
                        for dx in -3..=3 {
                            v.push(at(&cells, cx + dx, cy + dy)[c]);
                        }
                    }
                    let m = v.len() / 2;
                    out[c] = *v.select_nth_unstable_by(m, f32::total_cmp).1;
                }
                out
            })
            .collect();
        // 3x3-cell box blur.
        let grid: Vec<[f32; 3]> = (0..gw * gh)
            .map(|ci| {
                let (cx, cy) = ((ci % gw) as isize, (ci / gw) as isize);
                let mut s = [0.0; 3];
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let v = at(&med, cx + dx, cy + dy);
                        for c in 0..3 {
                            s[c] += v[c] / 9.0;
                        }
                    }
                }
                s
            })
            .collect();
        let mut board = [0.0; 3];
        for g in &grid {
            for c in 0..3 {
                board[c] += g[c] / grid.len() as f32;
            }
        }
        // Keep the board's hue but set a fixed, comfortable brightness, so Natural looks the same under any light.
        let by = luma(board[0], board[1], board[2]).max(1.0);
        self.board = board.map(|v| v * BOARD_LUMA / by);
        // Bilinear upsample (cell centres at (i + 0.5) * CELL).
        let gat = |x: usize, y: usize| grid[y.min(gh - 1) * gw + x.min(gw - 1)];
        self.light
            .par_chunks_mut(w * 3)
            .enumerate()
            .with_min_len(ROWS_PER_TASK)
            .for_each(|(y, row)| {
                let gy = ((y as f32 + 0.5) / CELL as f32 - 0.5).max(0.0);
                let (y0, fy) = (gy as usize, gy.fract());
                for x in 0..w {
                    let gx = ((x as f32 + 0.5) / CELL as f32 - 0.5).max(0.0);
                    let (x0, fx) = (gx as usize, gx.fract());
                    let (a, b) = (gat(x0, y0), gat(x0 + 1, y0));
                    let (c, d) = (gat(x0, y0 + 1), gat(x0 + 1, y0 + 1));
                    for k in 0..3 {
                        row[x * 3 + k] = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy)
                            + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
                    }
                }
            });
        // Top-hat statistics. Most pixels are bare board, so the median top-hat measures noise; the floor sits
        // a few times above it. Full ink is the 98th percentile of what's above the floor (the chalk).
        let board_y = BOARD_LUMA; // same reference brightness as `apply` uses
        let mut t: Vec<f32> = (0..w * h)
            .step_by(5)
            .map(|i| {
                let ly = luma(
                    self.light[i * 3],
                    self.light[i * 3 + 1],
                    self.light[i * 3 + 2],
                )
                .max(1.0);
                (self.y[i] - self.open[i]) * (board_y / ly).min(MAX_GAIN)
            })
            .collect();
        let m = t.len() / 2;
        let noise = *t.select_nth_unstable_by(m, f32::total_cmp).1;
        self.floor = (3.0 * noise).clamp(3.0, 40.0);
        let mut above: Vec<f32> = t.into_iter().filter(|&v| v > self.floor).collect();
        self.hi = if above.len() < 100 {
            self.floor + 30.0
        } else {
            let k = above.len() * 98 / 100;
            (*above.select_nth_unstable_by(k, f32::total_cmp).1).max(self.floor + 15.0)
        };
    }
}

/// Separable square min/max filter, one direction per call: `horizontal` filters along rows.
fn minmax(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize, max: bool, horizontal: bool) {
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let init = if max { f32::MIN } else { f32::MAX };
    dst.par_chunks_mut(w)
        .enumerate()
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(y, o)| {
            if horizontal {
                let s = &src[y * w..(y + 1) * w];
                for x in 0..w {
                    o[x] = s[x.saturating_sub(r)..=(x + r).min(w - 1)]
                        .iter()
                        .fold(init, |m, &v| pick(m, v));
                }
            } else {
                o.fill(init);
                for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    for (d, &v) in o.iter_mut().zip(&src[yy * w..(yy + 1) * w]) {
                        *d = pick(*d, v);
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 200;
    const H: usize = 240;

    /// A board lit brighter at the top, with chalk strokes near the top and bottom, and a sticky note.
    fn board() -> Rgb {
        let mut f = Rgb::new(W, H);
        for y in 0..H {
            for x in 0..W {
                let light = 1.4 - 0.8 * y as f32 / H as f32; // hot spot at the top
                let mut c = [40.0 * light, 90.0 * light, 70.0 * light];
                let stroke =
                    ((20..23).contains(&y) || (200..203).contains(&y)) && (30..130).contains(&x);
                if stroke {
                    c = c.map(|v| v + 110.0 * light);
                }
                if (150..190).contains(&x) && (80..120).contains(&y) {
                    c = [230.0, 220.0, 120.0]; // sticky note
                }
                f.data[(y * W + x) * 3..][..3].copy_from_slice(&c.map(|v| v.min(255.0) as u8));
            }
        }
        f
    }

    fn px(f: &Rgb, x: usize, y: usize) -> [u8; 3] {
        let i = (y * f.w + x) * 3;
        [f.data[i], f.data[i + 1], f.data[i + 2]]
    }

    fn run(look: Look) -> Rgb {
        let src = board();
        let mut e = Enhancer::new(look, W, H);
        let mut out = Rgb::new(W, H);
        e.apply(&src, &mut out);
        out
    }

    #[test]
    fn natural_flattens_lighting_and_keeps_strokes() {
        let out = run(Look::Natural);
        let (top, bottom) = (px(&out, 10, 8), px(&out, 10, 230));
        for c in 0..3 {
            assert!(
                (top[c] as i32 - bottom[c] as i32).abs() <= 8,
                "uneven board: {top:?} vs {bottom:?}"
            );
        }
        assert!(
            px(&out, 80, 21)[1] > top[1] + 60,
            "top stroke {:?}",
            px(&out, 80, 21)
        );
        assert!(
            px(&out, 80, 201)[1] > bottom[1] + 60,
            "bottom stroke {:?}",
            px(&out, 80, 201)
        );
    }

    #[test]
    fn paper_is_chalk_only_ink_on_white() {
        let out = run(Look::Paper);
        for (x, y, what) in [
            (10, 8, "top paper"),
            (10, 230, "bottom paper"),
            (170, 100, "sticky note"),
        ] {
            assert!(
                px(&out, x, y).iter().all(|&v| v > 230),
                "{what} {:?}",
                px(&out, x, y)
            );
        }
        assert!(
            px(&out, 80, 21).iter().all(|&v| v < 90),
            "top ink {:?}",
            px(&out, 80, 21)
        );
        assert!(
            px(&out, 80, 201).iter().all(|&v| v < 120),
            "bottom ink {:?}",
            px(&out, 80, 201)
        );
    }
}
