//! Where my head is, from a YOLO pose model (yolo11n-pose, ONNX; see scripts/fetch-pose-model.sh).
//!
//! Pose rather than face detection: when I'm writing I have my back to the camera, and a face detector loses me
//! exactly then. Pose keypoints still find my ears and shoulders from behind.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;

use crate::image::{ResampleMap, Rgb};

/// YOLO's letterbox padding grey (what it was trained with).
const PAD: f32 = 114.0 / 255.0;

/// My head, in the pixels of the frame given to `Pose::run`.
#[derive(Clone, Copy, Debug)]
pub struct Head {
    pub x: f32,
    /// Eye level (or the equivalent from behind).
    pub y: f32,
    /// Head width, when there were the keypoints to tell.
    pub width: Option<f32>,
}

/// What pose found of me (in `Pose::run`'s frame pixels).
#[derive(Clone, Copy, Debug, Default)]
pub struct Person {
    pub head: Option<Head>,
}

pub struct Pose {
    session: Session,
    _env: ort::environment::Environment,
    /// Model input size, and the input tensor (NCHW, padded with `PAD`).
    in_w: usize,
    in_h: usize,
    input: Vec<f32>,
    /// The frame is scaled by `scale` (keeping its shape: YOLO finds no one in a squashed picture) to
    /// `fit_w` wide, placed at (`ox`, `oy`) in the input; `resize` does the scaling when it isn't 1.
    scale: f32,
    fit_w: usize,
    ox: usize,
    oy: usize,
    resize: Option<(ResampleMap, Rgb)>,
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("onnxruntime: {e}")
}

impl Pose {
    /// `w` x `h`: the size of the frames that will be passed to `run` (resized to the model's input if different).
    pub fn new(model: &Path, w: usize, h: usize, threads: usize, gpu: bool) -> Result<Self> {
        let env = crate::segment::ort_env()?;
        let (session, _) = crate::segment::load(&env, model, threads, gpu).with_context(|| {
            format!(
                "loading {} (run scripts/fetch-pose-model.sh)",
                model.display()
            )
        })?;
        // A fixed-size model (exported for one picture shape) gets the frame letterboxed into it; a dynamic one
        // gets it as it is, padded up to the multiple of 32 YOLO needs.
        let dims = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let (in_w, in_h) = match dims[..] {
            [_, _, h, w] if h > 0 && w > 0 => (w as usize, h as usize),
            _ => (w.next_multiple_of(32), h.next_multiple_of(32)),
        };
        let (fit, scale) = letterbox((w, h), (in_w, in_h));
        let resize = (fit != (w, h)).then(|| (ResampleMap::scale(w, h, fit.0, fit.1), Rgb::new(fit.0, fit.1)));
        Ok(Pose {
            session,
            _env: env,
            in_w,
            in_h,
            input: vec![PAD; 3 * in_w * in_h],
            scale,
            fit_w: fit.0,
            ox: (in_w - fit.0) / 2,
            oy: (in_h - fit.1) / 2,
            resize,
        })
    }

    /// My head in `frame`, or None if nobody's there. Coordinates are in `frame`'s pixels.
    pub fn run(&mut self, frame: &Rgb) -> Result<Person> {
        let img = match &mut self.resize {
            Some((map, small)) => {
                map.apply(frame, small);
                &*small
            }
            None => frame,
        };
        let (n, iw) = (self.in_w * self.in_h, self.in_w);
        let (r, gb) = self.input.split_at_mut(n);
        let (g, b) = gb.split_at_mut(n);
        for (y, row) in img.data.chunks_exact(self.fit_w * 3).enumerate() {
            let o = (self.oy + y) * iw + self.ox;
            for (x, px) in row.chunks_exact(3).enumerate() {
                r[o + x] = px[0] as f32 / 255.0;
                g[o + x] = px[1] as f32 / 255.0;
                b[o + x] = px[2] as f32 / 255.0;
            }
        }
        let t = Tensor::from_array((
            vec![1i64, 3, self.in_h as i64, self.in_w as i64],
            self.input.clone(),
        ))
        .map_err(ort_err)?;
        let out = self
            .session
            .run(ort::inputs!["images" => t])
            .map_err(ort_err)?;
        let (shape, data) = out["output0"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if shape.len() != 3 || shape[1] != 56 {
            return Err(anyhow!("unexpected pose output shape {shape}"));
        }
        let n = shape[2] as usize;
        Ok(Person {
            head: best_head(data, n).map(|h| Head {
                x: (h.x - self.ox as f32) / self.scale,
                y: (h.y - self.oy as f32) / self.scale,
                width: h.width.map(|w| w / self.scale),
            }),
        })
    }
}

/// The size a `w` x `h` picture is scaled to so it fits in `in_w` x `in_h` keeping its shape, and the scale.
fn letterbox((w, h): (usize, usize), (in_w, in_h): (usize, usize)) -> ((usize, usize), f32) {
    let scale = (in_w as f32 / w as f32).min(in_h as f32 / h as f32);
    let fit = |v: usize, max: usize| ((v as f32 * scale).round() as usize).clamp(1, max);
    ((fit(w, in_w), fit(h, in_h)), scale)
}

/// Keypoint confidence needed to use a point.
const SEEN: f32 = 0.5;
/// Person score needed.
const PERSON: f32 = 0.5;

/// From the [56, n] output (box cx, cy, w, h, score, then 17 keypoints x, y, conf; channel-major): the most
/// confident person's head.
fn best_head(out: &[f32], n: usize) -> Option<Head> {
    let at = |c: usize, i: usize| out[c * n + i];
    let best = (0..n).max_by(|&a, &b| at(4, a).total_cmp(&at(4, b)))?;
    if at(4, best) < PERSON {
        return None;
    }
    let kp = |k: usize| {
        let (x, y, c) = (
            at(5 + 3 * k, best),
            at(6 + 3 * k, best),
            at(7 + 3 * k, best),
        );
        (c >= SEEN).then_some((x, y))
    };
    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    // COCO: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders.
    let face: Vec<(f32, f32)> = (0..5).filter_map(kp).collect();
    let (ears, shoulders, eyes) = ((kp(3), kp(4)), (kp(5), kp(6)), (kp(1), kp(2)));
    // Each pair only shrinks as I turn (ears close up in profile, shoulders when I stand side-on), so the largest
    // estimate is the least foreshortened one.
    type Pt = Option<(f32, f32)>;
    let pair = |(a, b): (Pt, Pt), k: f32| Some(dist(a?, b?) * k);
    let width = [pair(ears, 1.1), pair(shoulders, 0.45), pair(eyes, 2.6)]
        .into_iter()
        .flatten()
        .reduce(f32::max);
    if !face.is_empty() {
        let mean = |pts: &[(f32, f32)]| {
            let m = pts.len() as f32;
            pts.iter()
                .fold((0.0, 0.0), |s, p| (s.0 + p.0 / m, s.1 + p.1 / m))
        };
        let (x, mut y) = mean(&face);
        // Height from the ears when they're visible: they stay at eye level when I nod, while the nose and eyes
        // drop as I look down (which would put the crop too low and cut off the top of my head).
        let ear_pts: Vec<(f32, f32)> = [ears.0, ears.1].into_iter().flatten().collect();
        if !ear_pts.is_empty() {
            y = mean(&ear_pts).1;
        }
        return Some(Head { x, y, width });
    }
    // No head points (looking down, or cut off): put the head above the shoulders.
    if let (Some(a), Some(b)) = shoulders {
        let span = dist(a, b);
        return Some(Head {
            x: (a.0 + b.0) / 2.0,
            y: (a.1 + b.1) / 2.0 - 0.55 * span,
            width: Some(0.45 * span),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One detection among `n`, with the given keypoints (index, x, y, conf).
    fn output(n: usize, i: usize, score: f32, kps: &[(usize, f32, f32, f32)]) -> Vec<f32> {
        let mut out = vec![0.0; 56 * n];
        out[4 * n + i] = score;
        for &(k, x, y, c) in kps {
            out[(5 + 3 * k) * n + i] = x;
            out[(6 + 3 * k) * n + i] = y;
            out[(7 + 3 * k) * n + i] = c;
        }
        out
    }

    #[test]
    fn letterbox_keeps_the_picture_shape() {
        // A landscape picture into the old portrait model: fits the width, centred, same shape.
        let ((w, h), s) = letterbox((512, 288), (288, 512));
        assert_eq!((w, h), (288, 162));
        assert!((s - 0.5625).abs() < 1e-6);
        // Same shape: untouched.
        assert_eq!(letterbox((512, 288), (512, 288)), ((512, 288), 1.0));
    }

    #[test]
    fn head_from_behind_uses_the_ears() {
        // Back to the camera: no nose or eyes, both ears and shoulders.
        let out = output(
            10,
            3,
            0.9,
            &[
                (3, 100.0, 50.0, 0.9),
                (4, 130.0, 52.0, 0.9),
                (5, 80.0, 100.0, 0.9),
                (6, 150.0, 100.0, 0.9),
            ],
        );
        let h = best_head(&out, 10).unwrap();
        assert!(
            (h.x - 115.0).abs() < 0.1 && (h.y - 51.0).abs() < 0.1,
            "{h:?}"
        );
        assert!((h.width.unwrap() - 33.0).abs() < 0.5, "{h:?}");
    }

    #[test]
    fn turning_my_head_does_not_shrink_it() {
        // In profile the ears nearly overlap; the shoulders still give the size.
        let out = output(
            4,
            1,
            0.9,
            &[
                (3, 100.0, 50.0, 0.9),
                (4, 106.0, 50.0, 0.9),
                (5, 80.0, 100.0, 0.9),
                (6, 150.0, 100.0, 0.9),
            ],
        );
        let h = best_head(&out, 4).unwrap();
        assert!((h.width.unwrap() - 31.5).abs() < 0.5, "{h:?}");
    }

    #[test]
    fn falls_back_to_the_shoulders_then_to_nothing() {
        let out = output(4, 0, 0.8, &[(5, 80.0, 100.0, 0.9), (6, 160.0, 100.0, 0.9)]);
        let h = best_head(&out, 4).unwrap();
        assert!(
            (h.x - 120.0).abs() < 0.1 && h.y < 100.0 && (h.width.unwrap() - 36.0).abs() < 0.1,
            "{h:?}"
        );
        assert!(
            best_head(&output(4, 0, 0.8, &[]), 4).is_none(),
            "no keypoints, no head"
        );
        assert!(
            best_head(&output(4, 0, 0.2, &[(0, 1.0, 1.0, 0.9)]), 4).is_none(),
            "low person score"
        );
    }
}
