//! tilesvc: offline land-use classification for satellite tiles.
//! Each `pub mod` line compiles `src/<name>.rs` and makes it usable from main.rs and tests.

pub mod api;
pub mod db;
pub mod features;
pub mod model;
