//! Where my head is, from a YOLO pose model (yolo11n-pose, ONNX; see scripts/fetch-pose-model.sh).
//!
//! Pose rather than face detection: when I'm writing I have my back to the camera, and a face detector loses me
//! exactly then. Pose keypoints still find my ears and shoulders from behind.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;

use crate::image::{ResampleMap, Rgb};

/// The model's fixed input size (exported at 512x288, portrait: the upright camera picture).
pub const W: usize = 288;
pub const H: usize = 512;

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
    input: Vec<f32>,
    resize: Option<(ResampleMap, Rgb)>,
}

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("onnxruntime: {e}")
}

impl Pose {
    /// `w` x `h`: the size of the frames that will be passed to `run` (resized to the model's input if different).
    pub fn new(model: &Path, w: usize, h: usize, threads: usize) -> Result<Self> {
        let env = crate::segment::ort_env()?;
        let session = Session::builder(&env)
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
                    "loading {} (run scripts/fetch-pose-model.sh)",
                    model.display()
                )
            })?;
        let resize = ((w, h) != (W, H)).then(|| (ResampleMap::scale(w, h, W, H), Rgb::new(W, H)));
        Ok(Pose {
            session,
            _env: env,
            input: vec![0.0; 3 * W * H],
            resize,
        })
    }

    /// My head in `frame`, or None if nobody's there. Coordinates are in `frame`'s pixels.
    pub fn run(&mut self, frame: &Rgb) -> Result<Person> {
        let (img, kx, ky) = match &mut self.resize {
            Some((map, small)) => {
                map.apply(frame, small);
                (
                    &*small,
                    frame.w as f32 / W as f32,
                    frame.h as f32 / H as f32,
                )
            }
            None => (frame, 1.0, 1.0),
        };
        let n = W * H;
        let (r, gb) = self.input.split_at_mut(n);
        let (g, b) = gb.split_at_mut(n);
        for (i, px) in img.data.chunks_exact(3).enumerate() {
            r[i] = px[0] as f32 / 255.0;
            g[i] = px[1] as f32 / 255.0;
            b[i] = px[2] as f32 / 255.0;
        }
        let t = Tensor::from_array((vec![1i64, 3, H as i64, W as i64], self.input.clone()))
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
                x: h.x * kx,
                y: h.y * ky,
                width: h.width.map(|w| w * kx),
            }),
        })
    }
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
    let width = match (ears, shoulders, eyes) {
        ((Some(a), Some(b)), _, _) => Some(dist(a, b) * 1.1),
        (_, (Some(a), Some(b)), _) => Some(dist(a, b) * 0.45),
        (_, _, (Some(a), Some(b))) => Some(dist(a, b) * 2.6),
        _ => None,
    };
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
