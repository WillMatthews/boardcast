//! Person segmentation with Robust Video Matting (RVM, MobileNetV3) through ONNX Runtime.
//!
//! RVM is recurrent: each call feeds back four hidden states, which keeps the matte temporally stable.
//! Model: https://github.com/PeterL1n/RobustVideoMatting (rvm_mobilenetv3_fp32.onnx, see scripts/fetch-model.sh).

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;

use crate::image::Rgb;

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("onnxruntime: {e}")
}

/// The process's ONNX Runtime environment (ORT allows only one), shared by every session.
pub fn ort_env() -> Result<ort::environment::Environment> {
    static ENV: std::sync::Mutex<Option<ort::environment::Environment>> =
        std::sync::Mutex::new(None);
    let mut env = ENV.lock().unwrap();
    if env.is_none() {
        *env = Some(ort::init().with_name("boardvu").build().map_err(ort_err)?);
    }
    Ok(env.clone().unwrap())
}

pub struct Matting {
    session: Session,
    _env: ort::environment::Environment,
    /// Recurrent states r1..r4 (shape, data), fed back each frame.
    rec: Vec<(Vec<i64>, Vec<f32>)>,
    src: Vec<f32>,
    ratio: f32,
    pub w: usize,
    pub h: usize,
}

impl Matting {
    /// `w` x `h` is the size of the frames passed to `run`. `ratio` is RVM's internal downsample ratio:
    /// pick it so `ratio * max(w, h)` is roughly 256-512.
    pub fn new(model: &Path, w: usize, h: usize, ratio: f32, threads: usize) -> Result<Self> {
        let env = ort_env()?;
        let session = Session::builder(&env)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            // Idle ORT threads busy-wait by default, burning whole cores between our ~10 Hz runs.
            .with_intra_op_spinning(false)
            .map_err(ort_err)?
            .with_inter_op_spinning(false)
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)
            .with_context(|| format!("loading {} (run scripts/fetch-model.sh)", model.display()))?;
        Ok(Matting {
            session,
            _env: env,
            rec: vec![(vec![1, 1, 1, 1], vec![0.0]); 4],
            src: vec![0.0; 3 * w * h],
            ratio,
            w,
            h,
        })
    }

    /// Forget the recurrent state (e.g. after a long gap between frames).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.rec = vec![(vec![1, 1, 1, 1], vec![0.0]); 4];
    }

    /// Alpha matte of the people in `frame` (which must be `w` x `h`), written to `alpha` (0 = background, 1 = person).
    pub fn run(&mut self, frame: &Rgb, alpha: &mut [f32]) -> Result<()> {
        let (w, h) = (self.w, self.h);
        assert_eq!((frame.w, frame.h), (w, h));
        assert_eq!(alpha.len(), w * h);
        // Packed RGB u8 -> planar f32 [1, 3, h, w] in 0..1.
        let n = w * h;
        let (r, gb) = self.src.split_at_mut(n);
        let (g, b) = gb.split_at_mut(n);
        for (i, px) in frame.data.chunks_exact(3).enumerate() {
            r[i] = px[0] as f32 / 255.0;
            g[i] = px[1] as f32 / 255.0;
            b[i] = px[2] as f32 / 255.0;
        }
        let t =
            |shape: Vec<i64>, data: Vec<f32>| Tensor::from_array((shape, data)).map_err(ort_err);
        let inputs = ort::inputs![
            "src" => t(vec![1, 3, h as i64, w as i64], self.src.clone())?,
            "r1i" => t(self.rec[0].0.clone(), self.rec[0].1.clone())?,
            "r2i" => t(self.rec[1].0.clone(), self.rec[1].1.clone())?,
            "r3i" => t(self.rec[2].0.clone(), self.rec[2].1.clone())?,
            "r4i" => t(self.rec[3].0.clone(), self.rec[3].1.clone())?,
            "downsample_ratio" => t(vec![1], vec![self.ratio])?,
        ];
        let out = self.session.run(inputs).map_err(ort_err)?;
        let (shape, pha) = out["pha"].try_extract_tensor::<f32>().map_err(ort_err)?;
        if pha.len() != n {
            return Err(anyhow!(
                "RVM returned alpha of shape {shape}, expected {h}x{w}"
            ));
        }
        alpha.copy_from_slice(pha);
        for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
            let (shape, data) = out[*name].try_extract_tensor::<f32>().map_err(ort_err)?;
            self.rec[i] = (shape.to_vec(), data.to_vec());
        }
        Ok(())
    }
}
