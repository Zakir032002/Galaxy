//! Phase 0: look at the data before designing anything.
//! Prints each class folder and how many PNG tiles it holds.

use std::fs;

// `Result<(), std::io::Error>` lets us use `?`: if read_dir fails, main returns the error.
fn main() -> std::io::Result<()> {
    let root = "be-mlsys-assignment-dataset/candidate_tiles";

    // Collect the class folder names, sorted so the order is always the same.
    let mut classes: Vec<String> = fs::read_dir(root)?
        .filter_map(|entry| entry.ok()) // skip entries we can't read
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    classes.sort();

    for class in &classes {
        // `&classes` borrows the Vec; we only read it, so we don't need to own it.
        let count = fs::read_dir(format!("{root}/{class}"))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .count();
        println!("{class:<12} {count}");
    }
    println!("{} classes", classes.len());
    Ok(())
}
