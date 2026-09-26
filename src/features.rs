//! Phase 2: turn a 64x64 RGB tile into 38 numbers a model can learn from.
//!
//!   colour  : per channel (R, G, B) -> mean, standard deviation, 8-bin histogram = 3 x 10 = 30
//!   texture : mean |dx|, mean |dy|, 6-bin histogram of edge strength           = 2 + 6  = 8
//!
//! Water is dark and smooth, forest is dark green, cities have lots of edges, and so on.

use image::RgbImage;

pub const FEATURE_DIM: usize = 38;
const BINS: usize = 8;
const EDGE_BINS: usize = 6;
const EDGE_MAX: f32 = 0.3; // edge strengths above this all land in the last bin

/// `&RgbImage` = we *borrow* the image: we can read it, but the caller still owns it.
pub fn extract(img: &RgbImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let n = (w * h) as f32;
    let mut out = Vec::with_capacity(FEATURE_DIM);

    // --- colour ---
    for c in 0..3 {
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        let mut hist = [0.0f32; BINS]; // fixed-size array on the stack
        for p in img.pixels() {
            let v = p[c] as f32 / 255.0; // 0..=255 -> 0.0..=1.0
            sum += v;
            sum_sq += v * v;
            hist[((v * BINS as f32) as usize).min(BINS - 1)] += 1.0; // min(): v == 1.0 would overflow
        }
        let mean = sum / n;
        out.push(mean);
        out.push((sum_sq / n - mean * mean).max(0.0).sqrt()); // max(0): float rounding can go below 0
        out.extend(hist.iter().map(|count| count / n)); // counts -> fractions
    }

    // --- texture ---
    let gray: Vec<f32> = img.pixels().map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 765.0).collect();
    let at = |x: u32, y: u32| gray[(y * w + x) as usize]; // closure that borrows `gray`

    let (mut dx_sum, mut dy_sum) = (0.0, 0.0);
    let mut edges = [0.0f32; EDGE_BINS];
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let dx = at(x + 1, y) - at(x, y); // horizontal change
            let dy = at(x, y + 1) - at(x, y); // vertical change
            dx_sum += dx.abs();
            dy_sum += dy.abs();
            let strength = (dx * dx + dy * dy).sqrt();
            edges[((strength / EDGE_MAX * EDGE_BINS as f32) as usize).min(EDGE_BINS - 1)] += 1.0;
        }
    }
    let m = ((w - 1) * (h - 1)) as f32;
    out.push(dx_sum / m);
    out.push(dy_sum / m);
    out.extend(edges.iter().map(|e| e / m));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn always_38_numbers() {
        assert_eq!(extract(&RgbImage::from_pixel(64, 64, Rgb([10, 200, 30]))).len(), FEATURE_DIM);
    }

    #[test]
    fn flat_image_has_no_edges() {
        let f = extract(&RgbImage::from_pixel(64, 64, Rgb([128, 128, 128])));
        assert_eq!((f[30], f[31]), (0.0, 0.0)); // mean |dx|, mean |dy|
        assert_eq!(f[32], 1.0); // every pixel in the weakest-edge bin
    }
}
