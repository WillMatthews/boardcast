//! Perspective geometry: a 3x3 homography from four point pairs.

pub type Pt = [f64; 2];

/// Row-major 3x3 homography mapping (x, y, 1) -> (u, v, w), result (u/w, v/w).
#[derive(Clone, Copy, Debug)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    /// Solve for H with H * src[i] ~ dst[i] (h33 fixed to 1). Returns None for degenerate input.
    pub fn from_points(src: &[Pt; 4], dst: &[Pt; 4]) -> Option<Self> {
        // 8 unknowns h11..h32; each pair gives two rows of A h = b.
        let mut a = [[0.0f64; 9]; 8]; // augmented [A | b]
        for i in 0..4 {
            let [x, y] = src[i];
            let [u, v] = dst[i];
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..8 {
            let piv = (col..8).max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))?;
            if a[piv][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, piv);
            for r in 0..8 {
                if r != col {
                    let f = a[r][col] / a[col][col];
                    if f != 0.0 {
                        for c in col..9 {
                            a[r][c] -= f * a[col][c];
                        }
                    }
                }
            }
        }
        let mut h = [0.0; 9];
        for i in 0..8 {
            h[i] = a[i][8] / a[i][i];
        }
        h[8] = 1.0;
        Some(Homography(h))
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> Pt {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        [
            (h[0] * x + h[1] * y + h[2]) / w,
            (h[3] * x + h[4] * y + h[5]) / w,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_corners_exactly() {
        let src = [
            [1168.0, 302.0],
            [1775.0, 232.0],
            [1771.0, 1163.0],
            [1160.0, 1091.0],
        ];
        let dst = [[0.0, 0.0], [809.0, 0.0], [809.0, 1079.0], [0.0, 1079.0]];
        let h = Homography::from_points(&dst, &src).unwrap();
        for i in 0..4 {
            let p = h.apply(dst[i][0], dst[i][1]);
            assert!(
                (p[0] - src[i][0]).abs() < 1e-6 && (p[1] - src[i][1]).abs() < 1e-6,
                "{p:?}"
            );
        }
    }

    #[test]
    fn degenerate_is_none() {
        let p = [[0.0, 0.0]; 4];
        assert!(Homography::from_points(&p, &p).is_none());
    }
}
