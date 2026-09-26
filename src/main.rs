//! CLI: `tilesvc train | eval | serve | classify <file.png>...`

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tilesvc::api::{AppState, router};
use tilesvc::db::Db;
use tilesvc::features::extract;
use tilesvc::model::{Classifier, MIN_CONFIDENCE, MIN_MARGIN, SoftmaxModel, Status};

const DATA: &str = "be-mlsys-assignment-dataset";
const MODEL: &str = "model.json";

#[tokio::main] // starts the async runtime so `serve` can await; train/eval just run normally
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    // Slice patterns: match on the *shape* of the argument list.
    match args.as_slice() {
        ["train"] => train(),
        ["eval"] => eval(),
        ["serve"] => serve().await,
        ["classify", files @ ..] => {
            let model = SoftmaxModel::load(Path::new(MODEL))?;
            for f in files {
                let p = model.predict(&image::open(f)?.to_rgb8());
                println!("{f}\t{}\t{:.2}", p.label, p.confidence);
            }
            Ok(())
        }
        _ => {
            eprintln!("usage: tilesvc train | eval | serve | classify <file.png>...");
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

/// Fail fast: a bad model file or unwritable data dir stops startup, not the first request.
async fn serve() -> Result<()> {
    std::fs::create_dir_all("data/tiles")?;
    let state = AppState {
        model: Arc::new(SoftmaxModel::load(Path::new(MODEL))?),
        db: Arc::new(Db::open("data/tiles.db")?),
        tiles_dir: "data/tiles".into(),
    };
    let addr = std::env::var("ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into()); // localhost: no auth, so opt in to exposing it
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("model {} | listening on http://{addr}", state.model.version());
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        }) // finish in-flight requests on Ctrl-C
        .await?;
    Ok(())
}

/// Accuracy, confusion matrix, and the coverage table used to pick the review thresholds.
fn eval() -> Result<()> {
    let model = SoftmaxModel::load(Path::new(MODEL))?;
    let classes = &model.classes;
    let index = |label: &str| classes.iter().position(|c| c == label);
    let mut confusion = vec![vec![0usize; classes.len()]; classes.len()]; // [true][predicted]
    let mut rows: Vec<(f32, bool, Status)> = Vec::new(); // (confidence, correct?, status)
    let mut mistakes = Vec::new();

    for line in std::fs::read_to_string(format!("{DATA}/eval_labels.csv"))?.lines().skip(1) {
        let Some((file, truth)) = line.split_once(',') else {
            continue;
        }; // let-else: skip bad lines
        let p = model.predict(&image::open(format!("{DATA}/eval_set/{file}"))?.to_rgb8());
        let (Some(t), Some(pred)) = (index(truth.trim()), index(&p.label)) else {
            continue;
        };
        confusion[t][pred] += 1;
        if t != pred {
            mistakes.push((p.confidence, format!("{file}: true {} -> predicted {}", truth.trim(), p.label)));
        }
        rows.push((p.confidence, t == pred, p.status()));
    }

    // Closure: accuracy of any subset of rows.
    let accuracy = |subset: &[&(f32, bool, Status)]| {
        100.0 * subset.iter().filter(|r| r.1).count() as f32 / subset.len().max(1) as f32
    };
    let all: Vec<_> = rows.iter().collect();
    println!("model {}  |  {} tiles  |  accuracy {:.1}%\n", model.version(), rows.len(), accuracy(&all));

    println!("confusion (rows = true, cols = predicted)            recall");
    for (i, row) in confusion.iter().enumerate() {
        let cells: String = row.iter().map(|v| format!("{v:>4}")).collect();
        println!("{:<12}{cells}   {:>5.1}%", classes[i], 100.0 * row[i] as f32 / row.iter().sum::<usize>() as f32);
    }

    println!("\nauto-accept if confidence >= t:\n     t  accepted  accuracy");
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let kept: Vec<_> = rows.iter().filter(|r| r.0 >= t).collect();
        println!("  {t:.2}   {:>5.1}%    {:>5.1}%", 100.0 * kept.len() as f32 / rows.len() as f32, accuracy(&kept));
    }

    let confident: Vec<_> = rows.iter().filter(|r| r.2 == Status::Confident).collect();
    println!(
        "\npolicy (confidence >= {MIN_CONFIDENCE}, margin >= {MIN_MARGIN}): {:.1}% sent to review, {:.1}% accurate on the rest",
        100.0 - 100.0 * confident.len() as f32 / rows.len() as f32,
        accuracy(&confident)
    );

    mistakes.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("\nmost confident mistakes (the dangerous ones: they skip review):");
    for (conf, m) in mistakes.iter().take(5) {
        println!("  {conf:.2}  {m}");
    }
    Ok(())
}

/// (class names, feature rows, class index per row). A type alias names a long type once.
type TrainingSet = (Vec<String>, Vec<Vec<f32>>, Vec<usize>);

/// Folder name = label.
fn load_training_set() -> Result<TrainingSet> {
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
