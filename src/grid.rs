//! A labelled pixel grid over a camera frame, for reading off rough board corners by eye (`boardvu setup`).

use crate::image::Rgb;

/// 3x5 digit glyphs, one row per entry, high bit on the left.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

fn put(img: &mut Rgb, x: usize, y: usize, col: [u8; 3]) {
    if x < img.w && y < img.h {
        let i = (y * img.w + x) * 3;
        img.data[i..i + 3].copy_from_slice(&col);
    }
}

/// `n` in digits `scale` pixels per dot, on a dark box, top-left at (x, y).
fn number(img: &mut Rgb, x: usize, y: usize, n: usize, scale: usize) {
    let text = n.to_string();
    let (bw, bh) = ((text.len() * 4 + 1) * scale, 7 * scale);
    for dy in 0..bh {
        for dx in 0..bw {
            put(img, x + dx, y + dy, [0, 0, 0]);
        }
    }
    for (k, c) in text.bytes().enumerate() {
        let glyph = DIGITS[(c - b'0') as usize];
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put(
                                img,
                                x + (1 + k * 4 + col) * scale + sx,
                                y + (1 + row) * scale + sy,
                                [255, 255, 0],
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Grid lines every `step` pixels (brighter every second one), labelled with their pixel coordinate along the top
/// and left edges.
pub fn draw(img: &mut Rgb, step: usize) {
    let scale = (img.w.max(img.h) / 640).max(2);
    for x in (step..img.w).step_by(step) {
        let col = if (x / step).is_multiple_of(2) { [255, 255, 0] } else { [128, 128, 0] };
        for y in 0..img.h {
            put(img, x, y, col);
        }
    }
    for y in (step..img.h).step_by(step) {
        let col = if (y / step).is_multiple_of(2) { [255, 255, 0] } else { [128, 128, 0] };
        for x in 0..img.w {
            put(img, x, y, col);
        }
    }
    for x in (2 * step..img.w).step_by(2 * step) {
        number(img, x + 2, 2, x, scale);
    }
    for y in (2 * step..img.h).step_by(2 * step) {
        number(img, 2, y + 2, y, scale);
    }
}
