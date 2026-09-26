# Design note (draft, before code)

## What we're building
An offline service. It receives a 64x64 satellite tile, classifies its land use (7 classes) with a model that runs locally, stores the result, and lets an analyst query the stored results.

## Components
- **features**: image -> a few dozen numbers (colour stats, edge/texture stats)
- **model**: numbers -> probability per class. It sits behind a `Classifier` trait so it can be swapped later.
- **db**: SQLite, one table of predictions
- **api**: HTTP.
  - `POST /tiles`: classify and store
  - `GET /predictions`: search
  - `GET /stats`: aggregates
  - `GET /health`: liveness
- **cli**: `train`, `eval` and `serve` in one binary

## Tile flow
1. hash the bytes
2. if this model already saw these bytes, return the stored row
3. decode and validate the image
4. predict
5. decide confident / needs_review
6. save the PNG and insert the row

## Open decisions
- Unsure predictions: store them with a flag, or drop them? (leaning: store + flag + keep all probabilities)
- Threshold: pick it from eval numbers, not a guess
- What does "query" mean? Filter by label, status and confidence; aggregate counts

## Questions for GalaxEye
- 3 bands or 13? Coordinates?
- What does each kind of error cost?
- Is the set of classes closed?
- Who reviews flagged tiles?
- What volume, and what hardware?
