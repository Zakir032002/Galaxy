//! Phase 3: the classifier. Softmax (multinomial logistic) regression on the 38 features.
//!
//! score_k = bias_k + sum_j weight_kj * z_j        (one score per class, z = standardised features)
//! prob_k  = exp(score_k) / sum_i exp(score_i)     (softmax: scores -> probabilities summing to 1)
//! Training nudges the weights so the true class gets a higher probability (gradient descent).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::features::{FEATURE_DIM, extract};

/// Anything that can classify a tile. The HTTP layer only knows this trait, so a CNN could
/// replace `SoftmaxModel` without touching api.rs.
/// `Send + Sync`: safe to share one model between many request threads.
pub trait Classifier: Send + Sync {
    fn version(&self) -> &str;
    fn predict(&self, img: &RgbImage) -> Prediction;
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub label: String,
    pub confidence: f32,          // probability of the top class
    pub margin: f32,              // top1 - top2: small = torn between two classes
    pub probs: Vec<(String, f32)>, // all classes, most likely first
}

#[derive(Serialize, Deserialize)]
pub struct SoftmaxModel {
    pub classes: Vec<String>,
    mean: Vec<f32>, // training-set mean/std per feature, used to standardise new tiles
    std: Vec<f32>,
    weights: Vec<Vec<f32>>, // [class][feature]
    bias: Vec<f32>,
    #[serde(skip)] // not stored in the file: it's the hash of the file itself
    version: String,
}

impl SoftmaxModel {
    /// `x[i]` = features of tile i, `y[i]` = index of its class in `classes`.
    pub fn train(classes: Vec<String>, x: &[Vec<f32>], y: &[usize], epochs: usize) -> Self {
        let (k, d, n) = (classes.len(), FEATURE_DIM, x.len() as f32);
        let (lr, l2) = (0.5, 1e-3); // step size; small pull towards 0 so it doesn't get over-confident

        // Standardise: every feature gets mean 0, std 1, so no feature dominates just by scale.
        let mean: Vec<f32> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f32>() / n).collect();
        let std: Vec<f32> = (0..d)
            .map(|j| (x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f32>() / n).sqrt().max(1e-6))
            .collect();
        let z: Vec<Vec<f32>> = x.iter().map(|r| standardise(r, &mean, &std)).collect();

        let mut m = SoftmaxModel {
            classes,
            mean,
            std,
            weights: vec![vec![0.0; d]; k],
            bias: vec![0.0; k],
            version: "untrained".into(),
        };
        for _ in 0..epochs {
            let (mut gw, mut gb) = (vec![vec![0.0; d]; k], vec![0.0; k]);
            for (zi, &yi) in z.iter().zip(y) {
                let p = m.probs(zi);
                for c in 0..k {
                    let err = p[c] - if c == yi { 1.0 } else { 0.0 }; // gradient of cross-entropy
                    gb[c] += err;
                    for j in 0..d {
                        gw[c][j] += err * zi[j];
                    }
                }
            }
            for c in 0..k {
                m.bias[c] -= lr * gb[c] / n;
                for j in 0..d {
                    m.weights[c][j] -= lr * (gw[c][j] / n + l2 * m.weights[c][j]);
                }
            }
        }
        m
    }

    /// Softmax over the class scores. Subtracting the max first keeps exp() from overflowing.
    fn probs(&self, z: &[f32]) -> Vec<f32> {
        let scores: Vec<f32> = (self.weights.iter().zip(&self.bias))
            .map(|(w, b)| b + w.iter().zip(z).map(|(wj, zj)| wj * zj).sum::<f32>())
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        exps.iter().map(|e| e / total).collect()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string(self)?).with_context(|| format!("writing {}", path.display()))
    }

    /// Load and validate. A bad model file should stop the service at startup, not serve garbage.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let mut m: SoftmaxModel = serde_json::from_slice(&bytes).context("parsing model JSON")?;
        let k = m.classes.len();
        ensure!(k >= 2 && m.weights.len() == k && m.bias.len() == k, "model shape doesn't match its classes");
        ensure!(
            m.mean.len() == FEATURE_DIM && m.weights.iter().all(|w| w.len() == FEATURE_DIM),
            "model expects a different number of features than this code extracts: retrain"
        );
        ensure!(m.weights.iter().flatten().all(|v| v.is_finite()), "model contains NaN/inf");
        m.version = format!("softmax-{}", &sha256_hex(&bytes)[..12]); // version = hash of the file
        Ok(m)
    }
}

impl Classifier for SoftmaxModel {
    fn version(&self) -> &str {
        &self.version
    }

    fn predict(&self, img: &RgbImage) -> Prediction {
        let p = self.probs(&standardise(&extract(img), &self.mean, &self.std));
        let mut probs: Vec<(String, f32)> = self.classes.iter().cloned().zip(p).collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1)); // f32 isn't `Ord` (NaN!), so use total_cmp
        let (label, confidence) = probs[0].clone();
        Prediction { label, confidence, margin: confidence - probs[1].1, probs }
    }
}

fn standardise(x: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    x.iter().zip(mean).zip(std).map(|((v, m), s)| (v - m) / s).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_and_sums_to_one() {
        let m = SoftmaxModel {
            classes: vec!["a".into(), "b".into()],
            mean: vec![],
            std: vec![],
            weights: vec![vec![1000.0], vec![1001.0]], // huge scores would overflow a naive exp()
            bias: vec![0.0, 0.0],
            version: String::new(),
        };
        let p = m.probs(&[1.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p[1] > p[0]);
    }
}
