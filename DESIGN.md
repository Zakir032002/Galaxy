# Design note: offline tile classification service

## 1. The whole thing

One Rust binary (`tilesvc`), one SQLite file, one folder of tiles, one model file. Nothing else to install, no network needed. That's the main design driver for an isolated, unwatched box.

```
tilesvc train  candidate_tiles/ -> 38 features per tile -> softmax regression -> model.json
tilesvc eval   eval_set/ + labels -> accuracy, confusion matrix, coverage-vs-threshold table
tilesvc serve  HTTP on 127.0.0.1:8080

POST /tiles (PNG bytes)
  1. sha256(bytes); already stored for this model version? -> return that row (200)
  2. decode; must be 64x64 RGB                          -> else 400 / 422
  3. features -> Classifier -> probability for each of the 7 classes
  4. status = confident if top >= 0.6 AND top - second >= 0.2, else needs_review
  5. save PNG as data/tiles/<sha256>.png; insert row; reply 201 with the full result

GET /predictions?label=&status=&min_confidence=&limit=   GET /stats   GET /health
```

| Component | Job | Why separate |
|---|---|---|
| `features.rs` | image -> 38 numbers (colour mean/std/histogram, edge strength) | Can change without touching anything else |
| `model.rs` | `Classifier` trait + softmax regression (train, predict, save, load) | The API only sees the trait, so a CNN can be dropped in later |
| `db.rs` | SQLite: insert, dedup lookup, search, stats | All SQL in one place |
| `api.rs` | HTTP + one error type mapped to status codes | Thin; CPU/DB work runs on `spawn_blocking` |

**Model choice.** Hand-made colour and texture features with softmax regression:
- 84.3% on the 210 held-out tiles
- trains in under 1 s on CPU
- has no runtime dependencies, and every weight can be explained

I compared two simpler baselines: nearest class-centroid reached ~62% and k-NN ~73%. A pretrained EuroSAT CNN would reach the high 90s, but it would need a model download plus an ML runtime, and accuracy isn't the point here. The trait keeps that upgrade to one new struct.

## 2. Decisions and trade-offs

**Predictions the model isn't sure about.**

| Option | Problem |
|---|---|
| Drop them | Loses data; hides where the model struggles |
| Store only the top label | Can't tell guesses from certain answers; can't revisit later |
| Replace with "Unknown" | Hides what the model thought ("54% River, 40% Highway" is useful) |
| **Store everything + `status` + all 7 probabilities** | **Chosen**: nothing lost; analysts filter on `status`; thresholds can change later |

Two conditions, because they catch different failures. Low confidence means the model doesn't like any answer. A low margin means it's torn between two classes, which is exactly our main confusion, River vs Highway. The thresholds come from the eval coverage table, not a guess:

| Auto-accept if confidence >= | Accepted | Accuracy on accepted |
|---|---|---|
| 0.0 (everything) | 100% | 84.3% |
| 0.6 | 79% | 91.0% |
| 0.7 | 68% | 95.8% |
| 0.9 | 43% | 100% |

At 0.6 / 0.2, 21% of tiles go to review and the rest are 91% accurate. Because confidence, margin and all the probabilities are stored, a new policy is one `UPDATE` statement, with no re-inference needed.

**What to store.** Each row holds:
- the hash of the tile bytes
- the filename
- the **model version**, which is the sha256 of the model file, so every result points at the exact weights that made it
- the label, confidence, margin and status
- all the probabilities
- the time

The raw PNG is kept on disk under its hash, which makes it possible to debug ("what did the model actually see?") and to re-classify with a better model. `UNIQUE(sha256, model_version)` makes retries safe: the same tile sent twice gives the same row. A new model version adds rows instead of overwriting, so two models can be compared on the same tiles. I don't store the features: they're cheap to recompute and would go stale if the feature code changed.

**What "querying the results" means.** I assumed three needs:
1. Find tiles: filter by label, status and minimum confidence, newest first.
2. Aggregates for a dashboard: count, average confidence and review rate per label and per day (`/stats`). This doubles as the drift signal for monitoring.
3. Ad-hoc SQL: `data/tiles.db` is a normal SQLite file.

The real need is probably spatial ("Industrial tiles in this area"), but the tiles carry no coordinates, so that's out of scope. With coordinates I'd add bbox columns and SQLite's R-tree index.

**Storage engine.** SQLite, not Postgres: one box, one writer, no server to administer, compiled into the binary, and crash-safe with WAL. Postgres becomes worth it with several machines or many concurrent writers.

**Synchronous vs queued ingest.** Inference takes about 0.1 ms, so a synchronous request/response is simplest and reports bad input immediately. Bulk ingest, or a slower CNN, would need a `jobs` table and a bounded worker pool. Only `api.rs` changes.

**Fail fast.** The model file is validated on load: shape, feature count, no NaN. A wrong image size is rejected (422), because a 128x128 tile is a different ground resolution from anything the model was trained on. On an unwatched box, a loud failure at startup beats quietly wrong results for a month.

## 3. Assumptions, and questions I'd ask

**Assumptions:**
- Tiles are 64x64 RGB PNGs, like the sample.
- The 7 classes are fixed.
- Volume is modest (thousands per day).
- One machine, on a trusted network, so no auth; the service binds to localhost by default.

**Questions:**
1. **Input.** Do we really get 3 bands, or all 13 Sentinel-2 bands? Near-infrared alone would fix most water/vegetation confusion. Do tiles come with coordinates and a timestamp?
2. **Cost of errors.** What decision do these results feed? Is missing an Industrial tile worse than mislabelling a River? That sets per-class thresholds.
3. **Closed set?** Real imagery has clouds, snow and no-data edges. Today the model must pick one of 7 classes.
4. **Review loop.** Who reviews `needs_review` tiles, and can their corrections come back as labels?
5. **Scale and hardware.** How many tiles per day, what latency is needed, is there a GPU?
6. **Updates.** How do new model versions reach an isolated box, and who approves them?

## 4. Built vs not built

**Built:**
- The core path (`POST /tiles`: classify, store, dedup, validate)
- Search, stats and health endpoints
- `train`, `eval` and `classify` commands
- Unit and HTTP tests

**Not built, deliberately:**
- Auth
- Job queue for bulk ingest
- Analyst correction endpoint
- Spatial queries
- Re-classification job for new model versions
- Confidence calibration
- A metrics endpoint
- Multispectral input
