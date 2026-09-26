//! Phase 3: `tilesvc train` and `tilesvc classify <file.png>`.

use std::path::Path;

use anyhow::Result;
use tilesvc::features::extract;
use tilesvc::model::{Classifier, SoftmaxModel};

const DATA: &str = "be-mlsys-assignment-dataset";
const MODEL: &str = "model.json";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    // Slice patterns: match on the *shape* of the argument list.
    match args.as_slice() {
        ["train"] => train(),
        ["classify", files @ ..] => {
            let model = SoftmaxModel::load(Path::new(MODEL))?;
            for f in files {
                let p = model.predict(&image::open(f)?.to_rgb8());
                println!("{f}\t{}\t{:.2}", p.label, p.confidence);
            }
            Ok(())
        }
        _ => {
            eprintln!("usage: tilesvc train | classify <file.png>...");
            std::process::exit(2);
        }
    }
}

fn train() -> Result<()> {
    let (classes, x, y) = load_training_set()?;
    let model = SoftmaxModel::train(classes, &x, &y, 2000);
    model.save(Path::new(MODEL))?;
    let loaded = SoftmaxModel::load(Path::new(MODEL))?; // round-trip check + computes the version
    println!("trained on {} tiles -> {MODEL} ({})", x.len(), loaded.version());
    Ok(())
}

/// Folder name = label. Returns (class names, feature rows, class index per row).
fn load_training_set() -> Result<(Vec<String>, Vec<Vec<f32>>, Vec<usize>)> {
    let root = format!("{DATA}/candidate_tiles");
    let mut classes: Vec<String> =
        std::fs::read_dir(&root)?.map(|e| Ok(e?.file_name().to_string_lossy().into_owned())).collect::<Result<_>>()?;
    classes.sort(); // class order decides which weight row is which class: keep it deterministic

    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (idx, class) in classes.iter().enumerate() {
        for entry in std::fs::read_dir(format!("{root}/{class}"))? {
            x.push(extract(&image::open(entry?.path())?.to_rgb8()));
            y.push(idx);
        }
    }
    Ok((classes, x, y))
}
