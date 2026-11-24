//! How sharp the board looks: the RMS of the luma Laplacian over the board, in board space.
//!
//! Chalk edges dominate it, so it rises as focus improves. Camera noise adds to it too, so compare numbers taken in
//! the same light, and average a few frames first where possible (`focus` does). In-camera sharpening also raises
//! it (halos are "edges"), so don't use it to pick the camera's sharpness setting.

use crate::image::Rgb;

/// Sharpness of `img` inside `rect` = (x, y, w, h), with a margin trimmed so the board's frame isn't counted.
pub fn sharpness(img: &Rgb, (x0, y0, w, h): (usize, usize, usize, usize)) -> f32 {
    let m = (w.min(h) / 25).max(2);
    let (x0, y0, x1, y1) = (
        x0 + m,
        y0 + m,
        (x0 + w).saturating_sub(m),
        (y0 + h).saturating_sub(m),
    );
    let (x1, y1) = (x1.min(img.w - 1), y1.min(img.h - 1));
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return 0.0;
    }
    let y = |x: usize, y: usize| {
        let i = (y * img.w + x) * 3;
        0.299 * img.data[i] as f32 + 0.587 * img.data[i + 1] as f32 + 0.114 * img.data[i + 2] as f32
    };
    let (mut sum, mut n) = (0.0f64, 0usize);
    for j in y0.max(1)..y1 {
        for i in x0.max(1)..x1 {
            let l = 4.0 * y(i, j) - y(i - 1, j) - y(i + 1, j) - y(i, j - 1) - y(i, j + 1);
            sum += (l * l) as f64;
            n += 1;
        }
    }
    (sum / n as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vertical stripes, blurred by a box of `blur` pixels.
    fn stripes(blur: usize) -> Rgb {
        let (w, h) = (80, 60);
        let sharp: Vec<f32> = (0..w)
            .map(|x| if (x / 4) % 2 == 0 { 40.0 } else { 220.0 })
            .collect();
        let mut img = Rgb::new(w, h);
        for x in 0..w {
            let lo = x.saturating_sub(blur / 2);
            let hi = (x + blur / 2 + 1).min(w);
            let v = sharp[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            for y in 0..h {
                img.data[(y * w + x) * 3..][..3].fill(v as u8);
            }
        }
        img
    }

    #[test]
    fn sharper_scores_higher() {
        let r = (0, 0, 80, 60);
        let (a, b, c) = (
            sharpness(&stripes(1), r),
            sharpness(&stripes(3), r),
            sharpness(&stripes(5), r),
        );
        assert!(a > b && b > c && c > 0.0, "{a} {b} {c}");
        assert_eq!(sharpness(&Rgb::new(80, 60), r), 0.0);
    }
}
