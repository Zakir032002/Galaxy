//! Phase 6: the HTTP service.
//!   POST /tiles?filename=x.png   body = PNG bytes   -> classify + store  (the core path)
//!   GET  /predictions?label=&status=&min_confidence=&limit=  -> analyst search
//!   GET  /stats                                               -> per label / per day aggregates
//!   GET  /health                                              -> alive + model version

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::db::{Db, Search};
use crate::model::{Classifier, sha256_hex};

pub const TILE_SIZE: u32 = 64; // what the model was trained on (10 m/pixel Sentinel-2 tiles)

/// Shared by every request. Cloning is cheap: an `Arc` clone just bumps a reference count.
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<dyn Classifier>, // any model implementing the trait
    pub db: Arc<Db>,
    pub tiles_dir: PathBuf, // original PNGs, named <sha256>.png
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/tiles", post(upload))
        .route("/predictions", get(search))
        .route("/stats", get(stats))
        .route("/health", get(health))
        .with_state(state)
}

// Handler arguments are "extractors": axum fills each one from the request based on its type.
async fn search(State(s): State<AppState>, Query(q): Query<Search>) -> Result<Json<Value>, ApiError> {
    blocking(move || Ok(Json(json!(s.db.search(&q)?)))).await
}

async fn stats(State(s): State<AppState>) -> Result<Json<Value>, ApiError> {
    blocking(move || Ok(Json(s.db.stats()?))).await
}

async fn health(State(s): State<AppState>) -> Json<Value> {
    Json(json!({ "status": "ok", "model_version": s.model.version() }))
}

#[derive(Deserialize)]
struct UploadParams {
    filename: Option<String>,
}

async fn upload(
    State(s): State<AppState>,
    Query(q): Query<UploadParams>,
    body: Bytes,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    // Hashing, decoding, inference and SQLite all block the thread, so they run on tokio's
    // blocking pool instead of stalling the async workers that serve other requests.
    blocking(move || {
        if body.is_empty() {
            return Err(ApiError::BadRequest("empty body: send PNG bytes".into()));
        }
        let sha = sha256_hex(&body);
        if let Some(existing) = s.db.find(&sha, s.model.version())? {
            return Ok((StatusCode::OK, Json(json!({ "duplicate": true, "prediction": existing }))));
        }
        let img =
            image::load_from_memory(&body).map_err(|e| ApiError::BadRequest(format!("not an image: {e}")))?.to_rgb8();
        if img.dimensions() != (TILE_SIZE, TILE_SIZE) {
            let (w, h) = img.dimensions();
            return Err(ApiError::Unprocessable(format!("tile is {w}x{h}, model expects {TILE_SIZE}x{TILE_SIZE}")));
        }
        let pred = s.model.predict(&img);
        std::fs::write(s.tiles_dir.join(format!("{sha}.png")), &body)?; // keep what the model saw
        let record = s.db.insert(&sha, q.filename.as_deref(), s.model.version(), &pred)?;
        Ok((StatusCode::CREATED, Json(json!({ "duplicate": false, "prediction": record }))))
    })
    .await
}

/// Run blocking work off the async threads. The `?` turns a panicked task (JoinError) into a 500;
/// the closure's own `Result` is what we return.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f).await?
}

/// One error type -> one place that decides HTTP status codes.
pub enum ApiError {
    BadRequest(String),      // 400: can't read the input at all
    Unprocessable(String),   // 422: readable, but not valid for this model (wrong size)
    Internal(anyhow::Error), // 500: our fault. Logged in full, client sees a generic message
}

// Lets `?` convert any error (rusqlite, io, JoinError, anyhow...) into ApiError::Internal.
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError::Internal(e.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            ApiError::Unprocessable(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
            ApiError::Internal(e) => {
                eprintln!("internal error: {e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
            }
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}
