//! Phase 7: end-to-end tests. The real router, an in-memory database, no network socket.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use image::{ImageFormat, Rgb, RgbImage};
use serde_json::Value;
use tower::ServiceExt; // .oneshot(): send one request straight into the router

use tilesvc::api::{AppState, router};
use tilesvc::db::Db;
use tilesvc::model::SoftmaxModel;

/// Returns the TempDir too: when it's dropped the folder is deleted (RAII), so it must live
/// as long as the test.
fn app() -> (axum::Router, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let model = SoftmaxModel::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("model.json")).unwrap();
    let state =
        AppState { model: Arc::new(model), db: Arc::new(Db::open(":memory:").unwrap()), tiles_dir: dir.path().into() };
    (router(state), dir)
}

fn png(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    RgbImage::from_pixel(w, h, Rgb(color)).write_to(&mut buf, ImageFormat::Png).unwrap();
    buf.into_inner()
}

async fn call(app: &axum::Router, method: &str, uri: &str, body: Vec<u8>) -> (StatusCode, Value) {
    let req = Request::builder().method(method).uri(uri).body(Body::from(body)).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn classify_store_and_dedup() {
    let (app, dir) = app();
    let (status, body) = call(&app, "POST", "/tiles?filename=lake.png", png(64, 64, [20, 40, 70])).await;
    assert_eq!(status, StatusCode::CREATED);
    let p = &body["prediction"];
    assert_eq!(p["filename"], "lake.png");
    assert_eq!(p["probs"].as_array().unwrap().len(), 7);
    assert!(dir.path().join(format!("{}.png", p["sha256"].as_str().unwrap())).exists()); // tile kept

    let (status, again) = call(&app, "POST", "/tiles", png(64, 64, [20, 40, 70])).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(again["duplicate"], true);
    assert_eq!(again["prediction"]["id"], p["id"]);
}

#[tokio::test]
async fn rejects_bad_input() {
    let (app, _dir) = app();
    assert_eq!(call(&app, "POST", "/tiles", vec![]).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(call(&app, "POST", "/tiles", b"not a png".to_vec()).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(call(&app, "POST", "/tiles", png(128, 128, [0, 0, 0])).await.0, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_stats_and_health() {
    let (app, _dir) = app();
    call(&app, "POST", "/tiles", png(64, 64, [20, 40, 70])).await;
    call(&app, "POST", "/tiles", png(64, 64, [200, 190, 120])).await;

    let (_, all) = call(&app, "GET", "/predictions", vec![]).await;
    assert_eq!(all.as_array().unwrap().len(), 2);
    let label = all[0]["label"].as_str().unwrap();
    let (_, filtered) = call(&app, "GET", &format!("/predictions?label={label}"), vec![]).await;
    assert!(filtered.as_array().unwrap().iter().all(|r| r["label"] == label));
    let (_, none) = call(&app, "GET", "/predictions?min_confidence=1.1", vec![]).await;
    assert!(none.as_array().unwrap().is_empty());

    let (_, stats) = call(&app, "GET", "/stats", vec![]).await;
    assert_eq!(stats["by_day"][0]["count"], 2);
    let (status, health) = call(&app, "GET", "/health", vec![]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(health["model_version"].as_str().unwrap().starts_with("softmax-"));
}
