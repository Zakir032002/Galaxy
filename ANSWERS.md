# Part 3: Problem-solving

All numbers come from `tilesvc eval` on the provided 210-tile eval set (30 tiles per class).

## 1. The classifier is wrong ~30% of the time. What do I do, and is it good enough?

**First, find out where it's wrong.** An average hides everything. Our model is wrong 16% of the time overall, but the errors are not spread evenly:
- Industrial and Residential: ~97% recall.
- Highway: 60% recall.
- River: 73% recall.
- 13 of our 33 errors are River and Highway confused with each other: both are a thin line across fields, and our features see colour and edges, not shape.

A "30% wrong" model can be excellent on the classes that matter and useless on one.

**"Good enough" depends on the decision, not the model.** Compared to what: random guessing (14%), or an analyst labelling everything by hand? What does each error cost the user?

**The key lever is trading coverage for accuracy.** Auto-accept only confident predictions and send the rest to a person:
- confidence ≥ 0.7: 68% of tiles handled automatically at 96% accuracy
- confidence ≥ 0.9: 43% at 100%

So even a mediocre model can remove most of the manual work. If the confident slice still isn't accurate enough for the use case, the model isn't useful yet, whatever its average says.

**Then improve, cheapest first:**
1. Per-class thresholds.
2. Merge classes the user doesn't need separated.
3. Use more Sentinel-2 bands.
4. A pretrained CNN.
5. Feed the reviewers' corrections back in as training labels.

Also check the eval set itself: 30 tiles per class is small (roughly ±8% on each per-class number), and real traffic won't be balanced.

## 2. It runs offline with nobody watching. A month later, how do I know it still works?

The system has to check itself and leave evidence for whoever looks next.

1. **Golden set.** Ship a small labelled set with the model and re-run `tilesvc eval` on it at startup and daily. A drop here is proof, not a hint, because the right answers are known.
2. **Output drift against the eval baseline.** On the eval set the model's mean confidence is 0.80 and 21% of tiles go to review. `/stats` gives these per day, plus the label mix. A month later, mean confidence at 0.55 or "70% SeaLake" means the input changed: new sensor, season, clouds, or an upstream processing change.
3. **Input drift.** Log cheap image stats: brightness per band, share of blank tiles, rejected sizes.
4. **Operational health.**
   - error counts (400 / 422 / 500)
   - latency
   - free disk for the database and tiles
   - time of the last tile received (a service that silently stopped getting tiles looks perfectly "healthy")
5. **Real ground truth.** Human review of `needs_review` tiles plus a small random sample of `confident` ones. Without the random sample you only ever check the cases the model already doubts.
6. **Make it visible without a network:** a daily summary file, `/health` reporting "degraded" when a check fails, and logs collected on the next visit.

## 3. Tiles come in fine but stored results look wrong. My steps, in order

Each step splits the possible causes in half.

1. **Get concrete examples.** Which tile ids, what the analyst expected, and since when. "Looks wrong" becomes "these 12 rows since Tuesday".
2. **Look at the stored tile** (`data/tiles/<sha256>.png`). Is it what we think? Right area, not blank, not rotated, not BGR?
3. **Re-run the exact stored bytes offline** with the model version in the row (`tilesvc classify`).
   - Same answer: storage is fine; the problem is the model or the input. Go to 4.
   - Different answer: the serving path or storage is corrupting something. Go to 6.
4. **Check versions.** Did `model_version` change when the problem started? Does the model's class order match the code? A class-order shift mislabels everything and looks exactly like this. The loader already rejects a model with the wrong feature count.
5. **Run the golden set.**
   - Accuracy normal: the model is fine and **the input changed**. Compare the stats of recent tiles with the training tiles.
   - Accuracy bad: the model or feature code broke. Diff against the last good version.
6. **Check stored rows for consistency.** Does `label` equal the top of `probs`? Are timestamps and filenames plausible? Were rows edited by hand?
7. **Fix, then backfill.** Raw tiles are stored, so re-classify the affected range under a fixed model version (new rows; old rows kept for audit). Add a test or monitor that would have caught it.

## 4. Weakest part of the design, and what breaks first

**The weakest part is the model's confidence, and a lot rests on it.** The review policy, the drift signal and the analyst's trust all assume that "0.9 confident" means right about 90% of the time. That was only measured on 210 tiles from the same distribution as training.

**What breaks first is unusual input.** Softmax must share 100% across our 7 classes, so a cloud or snow tile can come out as "SeaLake 0.9". It lands in the *confident* bucket and never gets reviewed. Our eval already shows this failure mode: the most confident mistakes are River called AnnualCrop at 0.89 and SeaLake called River at 0.88.

Mitigations, cheapest first:
1. An input sanity gate (brightness and texture ranges seen in training).
2. Calibrate confidences on held-out data (temperature scaling).
3. Add an "other" class trained on clouds and edge cases.
4. The golden-set and drift monitoring above.
5. Randomly audit some confident predictions.

**Operationally**, the first thing to strain under load is the single SQLite connection behind a mutex, with inference inside the request. That's fine at ~0.1 ms per tile. A 200 ms CNN on bulk uploads would need a job queue and a bounded worker pool. That change stays inside `api.rs`: the `Classifier` trait and the table stay the same.
