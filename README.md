# tilesvc: offline satellite tile classifier

A single Rust binary that classifies 64x64 satellite tiles into 7 land-use classes (AnnualCrop, Forest, Highway, Industrial, Residential, River, SeaLake), stores every result in SQLite, and lets an analyst query them. It runs fully offline

| Part | Where |
|---|---|
| 1. Design note | [`DESIGN.md`](DESIGN.md) |
| 2. Working slice | this repo (below) |
| 3. Answers | [`ANSWERS.md`](ANSWERS.md) |

## Run it

You need Rust 1.88 or newer (install from https://rustup.rs).

```bash
# put the dataset next to Cargo.toml
unzip Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip     # -> be-mlsys-assignment-dataset/

cargo build --release
./target/release/tilesvc train     # optional: model.json is already committed (< 1 s to retrain)
./target/release/tilesvc eval      # accuracy, confusion matrix, coverage table, confident mistakes
./target/release/tilesvc serve     # http://127.0.0.1:8080 (set ADDR=0.0.0.0:8080 to expose)
```

## Use it

```bash
# classify + store one tile (body = raw PNG bytes)
curl -X POST --data-binary @be-mlsys-assignment-dataset/eval_set/tile_043.png \
     "localhost:8080/tiles?filename=tile_043.png"
# 201 {"duplicate":false,"prediction":{"label":"River","confidence":0.54,"margin":0.14,
#      "status":"needs_review","probs":[["River",0.54],["Highway",0.40],...],
#      "model_version":"softmax-28ba3f698447","sha256":"db7d...","id":1,...}}

# the same bytes again -> 200 {"duplicate":true,...}   not an image -> 400   not 64x64 -> 422

# load the whole eval set
for f in be-mlsys-assignment-dataset/eval_set/*.png; do
  curl -s -o /dev/null -X POST --data-binary @$f "localhost:8080/tiles?filename=$(basename $f)"; done

# query
curl "localhost:8080/predictions?status=needs_review"
curl "localhost:8080/predictions?label=Industrial&min_confidence=0.9&limit=20"
curl "localhost:8080/stats"        # per label and per day: count, avg confidence, review rate
curl "localhost:8080/health"
sqlite3 data/tiles.db "SELECT label, status, COUNT(*) FROM predictions GROUP BY 1, 2"
```

## Results (eval set, 210 tiles)

- **Accuracy:** 84.3%. River/Highway is the weak pair (60–73% recall); they account for 13 of the 33 errors.
- **Review policy** (confidence ≥ 0.6 and margin ≥ 0.2): 21% of tiles go to review, and the rest are 91% accurate.
- **Speed:** inference takes about 0.1 ms per tile; training takes under 1 s.

## Tests

```bash
cargo test                    # unit tests (features, model, db) + HTTP tests on an in-memory DB
cargo clippy --all-targets    # no warnings
```

## Layout

| Path | What it does |
|---|---|
| `src/features.rs` | image → 38 numbers (colour stats, edge/texture stats) |
| `src/model.rs` | `Classifier` trait; softmax regression train/predict/save/load; review policy |
| `src/db.rs` | SQLite table, dedup, search, stats |
| `src/api.rs` | axum routes, error type → HTTP status |
| `src/main.rs` | CLI: `train` / `eval` / `serve` / `classify` |
| `tests/api.rs` | end-to-end HTTP tests |
| `model.json` | trained model (its sha256 is the model version) |

The git history builds the project in phases 0–8, one commit each.

## Not built (see DESIGN.md §4)

- Auth
- Job queue
- Analyst corrections
- Spatial queries
- Re-classification job
- Calibration
- Metrics endpoint
- Multispectral input
