//! Frame buffers, precomputed resampling maps, and RGB -> YUYV conversion.

use rayon::prelude::*;

use crate::geom::Homography;

/// Minimum image rows per parallel task: one-row tasks cost more in scheduling than in work.
pub const ROWS_PER_TASK: usize = 16;

/// Packed 8-bit RGB image.
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn new(w: usize, h: usize) -> Self {
        Rgb {
            w,
            h,
            data: vec![0; w * h * 3],
        }
    }
}

/// One output pixel's bilinear source: top-left source pixel index and 8-bit fractional weights.
/// `idx == u32::MAX` means "outside the source": fill with the background colour.
#[derive(Clone, Copy)]
struct Tap {
    idx: u32,
    fx: u16, // 0..=256
    fy: u16,
}

/// A fixed geometric mapping output pixel -> source position, computed once and applied per frame.
pub struct ResampleMap {
    pub out_w: usize,
    pub out_h: usize,
    src_w: usize,
    src_h: usize,
    taps: Vec<Tap>,
    bg: [u8; 3],
}

impl ResampleMap {
    /// Build from a function giving the source (x, y) for an output pixel centre, or None for background.
    fn build(
        out_w: usize,
        out_h: usize,
        src_w: usize,
        src_h: usize,
        bg: [u8; 3],
        f: impl Fn(usize, usize) -> Option<[f64; 2]> + Sync,
    ) -> Self {
        let mut taps = vec![
            Tap {
                idx: u32::MAX,
                fx: 0,
                fy: 0
            };
            out_w * out_h
        ];
        taps.par_chunks_mut(out_w).enumerate().for_each(|(y, row)| {
            for (x, t) in row.iter_mut().enumerate() {
                let Some([sx, sy]) = f(x, y) else { continue };
                // Outside the camera frame -> background.
                if sx < -0.5 || sy < -0.5 || sx > src_w as f64 - 0.5 || sy > src_h as f64 - 0.5 {
                    continue;
                }
                let sx = sx.clamp(0.0, (src_w - 1) as f64);
                let sy = sy.clamp(0.0, (src_h - 1) as f64);
                // Top-left tap of the 2x2 neighbourhood, kept one short of the last row/column so the
                // other three taps stay in bounds; the weight then reaches 256 on the last pixel.
                let x0 = (sx.floor() as usize).min(src_w - 2);
                let y0 = (sy.floor() as usize).min(src_h - 2);
                *t = Tap {
                    idx: (y0 * src_w + x0) as u32,
                    fx: ((sx - x0 as f64) * 256.0).round() as u16,
                    fy: ((sy - y0 as f64) * 256.0).round() as u16,
                };
            }
        });
        ResampleMap {
            out_w,
            out_h,
            src_w,
            src_h,
            taps,
            bg,
        }
    }

    /// Perspective warp of the quad `corners` (TL, TR, BR, BL in source pixels) into a `board_w` x `board_h`
    /// rectangle, centred on an `out_w` x `out_h` canvas filled with `bg`.
    pub fn board(
        corners: &[[f64; 2]; 4],
        src_w: usize,
        src_h: usize,
        out_w: usize,
        out_h: usize,
        board_w: usize,
        board_h: usize,
        bg: [u8; 3],
    ) -> anyhow::Result<Self> {
        let (bw, bh) = (board_w as f64, board_h as f64);
        // Map the board rectangle's outer edges (pixel corners, not centres) onto the quad.
        let rect = [[0.0, 0.0], [bw, 0.0], [bw, bh], [0.0, bh]];
        let h = Homography::from_points(&rect, corners)
            .ok_or_else(|| anyhow::anyhow!("board corners are degenerate"))?;
        let ox = (out_w - board_w) / 2;
        let oy = (out_h - board_h) / 2;
        Ok(Self::build(out_w, out_h, src_w, src_h, bg, |x, y| {
            if x < ox || y < oy || x >= ox + board_w || y >= oy + board_h {
                return None;
            }
            let p = h.apply((x - ox) as f64 + 0.5, (y - oy) as f64 + 0.5);
            Some([p[0] - 0.5, p[1] - 0.5])
        }))
    }

    /// Plain rescale of the whole frame (for the raw passthrough).
    pub fn scale(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
        let (kx, ky) = (src_w as f64 / out_w as f64, src_h as f64 / out_h as f64);
        Self::build(out_w, out_h, src_w, src_h, [0; 3], |x, y| {
            Some([(x as f64 + 0.5) * kx - 0.5, (y as f64 + 0.5) * ky - 0.5])
        })
    }

    /// Single-channel version of `apply` (e.g. a mask); outside the source reads as 0.
    pub fn apply_gray(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), self.src_w * self.src_h, "source size changed");
        assert_eq!(dst.len(), self.out_w * self.out_h);
        let stride = self.src_w;
        dst.par_chunks_mut(self.out_w)
            .zip(self.taps.par_chunks(self.out_w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|(out, taps)| {
                for (o, t) in out.iter_mut().zip(taps) {
                    if t.idx == u32::MAX {
                        *o = 0.0;
                        continue;
                    }
                    let i = t.idx as usize;
                    let (fx, fy) = (t.fx as f32 / 256.0, t.fy as f32 / 256.0);
                    let top = src[i] * (1.0 - fx) + src[i + 1] * fx;
                    let bot = src[i + stride] * (1.0 - fx) + src[i + stride + 1] * fx;
                    *o = top * (1.0 - fy) + bot * fy;
                }
            });
    }

    pub fn apply(&self, src: &Rgb, dst: &mut Rgb) {
        assert_eq!(
            (src.w, src.h),
            (self.src_w, self.src_h),
            "source frame size changed"
        );
        assert_eq!((dst.w, dst.h), (self.out_w, self.out_h));
        let stride = self.src_w * 3;
        let s = &src.data;
        dst.data
            .par_chunks_mut(self.out_w * 3)
            .zip(self.taps.par_chunks(self.out_w))
            .with_min_len(ROWS_PER_TASK)
            .for_each(|(out, taps)| {
                for (px, t) in out.chunks_exact_mut(3).zip(taps) {
                    if t.idx == u32::MAX {
                        px.copy_from_slice(&self.bg);
                        continue;
                    }
                    let i = t.idx as usize * 3;
                    let (fx, fy) = (t.fx as u32, t.fy as u32);
                    let (w00, w10) = ((256 - fx) * (256 - fy), fx * (256 - fy));
                    let (w01, w11) = ((256 - fx) * fy, fx * fy);
                    for c in 0..3 {
                        let v = s[i + c] as u32 * w00
                            + s[i + 3 + c] as u32 * w10
                            + s[i + stride + c] as u32 * w01
                            + s[i + stride + 3 + c] as u32 * w11;
                        px[c] = ((v + 32768) >> 16) as u8;
                    }
                }
            });
    }
}

/// RGB -> YUYV 4:2:2, BT.601 limited range (what webcams and browsers expect for YUYV).
pub fn rgb_to_yuyv(src: &Rgb, dst: &mut [u8]) {
    assert!(src.w % 2 == 0);
    assert_eq!(dst.len(), src.w * src.h * 2);
    dst.par_chunks_mut(src.w * 2)
        .zip(src.data.par_chunks(src.w * 3))
        .with_min_len(ROWS_PER_TASK)
        .for_each(|(out, row)| {
            for (o, p) in out.chunks_exact_mut(4).zip(row.chunks_exact(6)) {
                let y = |r: i32, g: i32, b: i32| ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                let (r0, g0, b0) = (p[0] as i32, p[1] as i32, p[2] as i32);
                let (r1, g1, b1) = (p[3] as i32, p[4] as i32, p[5] as i32);
                let (r, g, b) = ((r0 + r1) / 2, (g0 + g1) / 2, (b0 + b1) / 2);
                o[0] = y(r0, g0, b0) as u8;
                o[1] = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
                o[2] = y(r1, g1, b1) as u8;
                o[3] = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_scale_is_lossless() {
        let mut src = Rgb::new(8, 6);
        for (i, v) in src.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        let map = ResampleMap::scale(8, 6, 8, 6);
        let mut dst = Rgb::new(8, 6);
        map.apply(&src, &mut dst);
        assert_eq!(src.data, dst.data);
    }

    #[test]
    fn yuyv_grey_levels() {
        let mut src = Rgb::new(2, 1);
        src.data = vec![255, 255, 255, 0, 0, 0];
        let mut out = [0u8; 4];
        rgb_to_yuyv(&src, &mut out);
        assert_eq!(out[0], 235);
        assert_eq!(out[2], 16);
        assert_eq!((out[1], out[3]), (128, 128));
    }
}
