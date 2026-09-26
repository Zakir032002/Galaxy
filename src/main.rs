//! Phase 2: print the 38 features of one tile.

use tilesvc::features;

fn main() -> Result<(), image::ImageError> {
    let img = image::open("be-mlsys-assignment-dataset/eval_set/tile_001.png")?.to_rgb8();
    let f = features::extract(&img);
    println!("{} features: {:.3?}", f.len(), f);
    Ok(())
}
