//! Smoothing for things that follow me (the picture-in-picture crop, the pointer).

/// Constant-velocity Kalman filter for one coordinate, with velocity damping (so it eases to a stop rather than
/// overshooting) and gating (a measurement far from the prediction is ignored unless they keep coming).
#[derive(Clone)]
pub struct Kalman {
    pub x: f32,
    v: f32,
    /// Covariance [[pp, pv], [pv, vv]].
    p: [f32; 3],
    /// Acceleration noise (px/s^2)^2: how freely it may change course.
    q: f32,
    /// Measurement noise (px)^2.
    r: f32,
    rejected: u32,
}

impl Kalman {
    /// Velocity half-life, seconds.
    const DAMP: f32 = 0.35;
    /// Gate: an innovation beyond this many standard deviations is an outlier. The first is ignored; if more
    /// follow (a real move, not a hand flicked up), they're let through clipped to a growing allowance, so a move
    /// is followed smoothly instead of ignored and then jumped to.
    const GATE: f32 = 3.5;

    pub fn new(x: f32, q: f32, r: f32) -> Self {
        Kalman {
            x,
            v: 0.0,
            p: [r, 0.0, q * 0.05],
            q,
            r,
            rejected: 0,
        }
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.5);
        let d = 0.5f32.powf(dt / Self::DAMP);
        self.x += self.v * dt;
        self.v *= d;
        let [pp, pv, vv] = self.p;
        // P = F P F' + Q, F = [[1, dt], [0, d]], Q from white acceleration noise.
        let (dt2, dt3, dt4) = (dt * dt, dt * dt * dt, dt * dt * dt * dt);
        self.p = [
            pp + 2.0 * dt * pv + dt2 * vv + self.q * dt4 / 4.0,
            d * (pv + dt * vv) + self.q * dt3 / 2.0,
            d * d * vv + self.q * dt2,
        ];
    }

    pub fn update(&mut self, z: f32) {
        let mut y = z - self.x;
        let limit = Self::GATE * (self.p[0] + self.r).sqrt();
        if y.abs() > limit {
            self.rejected += 1;
            if self.rejected == 1 {
                return;
            }
            y = y.clamp(-limit * self.rejected as f32, limit * self.rejected as f32);
        } else {
            self.rejected = 0;
        }
        let [pp, pv, vv] = self.p;
        let s = pp + self.r;
        let (k0, k1) = (pp / s, pv / s);
        self.x += k0 * y;
        self.v += k1 * y;
        self.p = [(1.0 - k0) * pp, (1.0 - k0) * pv, vv - k1 * pv];
    }
}
