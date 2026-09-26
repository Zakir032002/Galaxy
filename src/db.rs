//! Phase 5: storage. One SQLite table; the PNG itself is saved to disk by the API (named by hash).
//!
//! Why SQLite: one file, no server, compiled into our binary (`bundled`), crash-safe, and an
//! analyst can open it with any SQL tool. Right for a single offline box.

use std::sync::Mutex;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::model::Prediction;

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;  -- crash-safe; readers don't block the writer
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY,
    sha256        TEXT NOT NULL,              -- hash of the tile bytes
    filename      TEXT,
    model_version TEXT NOT NULL,              -- exactly which model file produced this
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    margin        REAL NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('confident', 'needs_review')),
    probs_json    TEXT NOT NULL,              -- all 7 probabilities: thresholds can change later
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (sha256, model_version)            -- same tile + same model = one row
);
CREATE INDEX IF NOT EXISTS idx_label ON predictions(label, confidence);";

const COLUMNS: &str = "id, sha256, filename, model_version, label, confidence, margin, status, probs_json, created_at";

#[derive(Debug, Serialize)]
pub struct Record {
    pub id: i64,
    pub sha256: String,
    pub filename: Option<String>,
    pub model_version: String,
    pub label: String,
    pub confidence: f64,
    pub margin: f64,
    pub status: String,
    pub probs: Value,
    pub created_at: String,
}

/// Analyst filters, all optional. `Deserialize` lets axum fill it straight from `?label=...&limit=...`.
#[derive(Debug, Default, Deserialize)]
pub struct Search {
    pub label: Option<String>,
    pub status: Option<String>,
    pub min_confidence: Option<f64>,
    pub limit: Option<u32>,
}

/// `Connection` is `Send` but not `Sync` (one thread at a time). The Mutex makes `Db` shareable
/// between request threads; without it the server would not compile.
pub struct Db(Mutex<Connection>);

impl Db {
    /// Pass ":memory:" for tests.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Db(Mutex::new(conn)))
    }

    pub fn find(&self, sha256: &str, model_version: &str) -> Result<Option<Record>> {
        let conn = self.0.lock().unwrap();
        let sql = format!("SELECT {COLUMNS} FROM predictions WHERE sha256 = ?1 AND model_version = ?2");
        Ok(conn.query_row(&sql, params![sha256, model_version], to_record).optional()?)
    }

    pub fn insert(&self, sha256: &str, filename: Option<&str>, model_version: &str, p: &Prediction) -> Result<Record> {
        let conn = self.0.lock().unwrap();
        // DO NOTHING: if two identical uploads race, the second returns the first one's row.
        conn.execute(
            "INSERT INTO predictions (sha256, filename, model_version, label, confidence, margin, status, probs_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) ON CONFLICT DO NOTHING",
            params![
                sha256,
                filename,
                model_version,
                p.label,
                p.confidence,
                p.margin,
                p.status().as_str(),
                serde_json::to_string(&p.probs)?
            ],
        )?;
        let sql = format!("SELECT {COLUMNS} FROM predictions WHERE sha256 = ?1 AND model_version = ?2");
        Ok(conn.query_row(&sql, params![sha256, model_version], to_record)?)
    }

    /// `?1 IS NULL OR ...` makes each filter optional in ONE fixed query. Values are always bound
    /// parameters, never pasted into the SQL string, so SQL injection is impossible.
    pub fn search(&self, s: &Search) -> Result<Vec<Record>> {
        let conn = self.0.lock().unwrap();
        let sql = format!(
            "SELECT {COLUMNS} FROM predictions
             WHERE (?1 IS NULL OR label = ?1) AND (?2 IS NULL OR status = ?2) AND (?3 IS NULL OR confidence >= ?3)
             ORDER BY id DESC LIMIT ?4"
        );
        let limit = s.limit.unwrap_or(50).min(500);
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![s.label, s.status, s.min_confidence, limit], to_record)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Per label and per day. Also the raw material for monitoring: if average confidence or the
    /// label mix drifts away from the eval baseline, the input has probably changed.
    pub fn stats(&self) -> Result<Value> {
        let conn = self.0.lock().unwrap();
        let group = |sql: &str| -> Result<Vec<Value>> {
            let mut stmt = conn.prepare(sql)?;
            let rows = stmt.query_map([], |r| {
                Ok(json!({ "key": r.get::<_, String>(0)?, "count": r.get::<_, i64>(1)?,
                           "avg_confidence": r.get::<_, f64>(2)?, "review_rate": r.get::<_, f64>(3)? }))
            })?;
            Ok(rows.collect::<Result<_, _>>()?)
        };
        let agg = "COUNT(*), AVG(confidence), AVG(status = 'needs_review') FROM predictions";
        Ok(json!({
            "by_label": group(&format!("SELECT label, {agg} GROUP BY 1 ORDER BY 2 DESC"))?,
            "by_day": group(&format!("SELECT substr(created_at, 1, 10), {agg} GROUP BY 1 ORDER BY 1 DESC LIMIT 30"))?,
        }))
    }
}

fn to_record(r: &Row) -> rusqlite::Result<Record> {
    Ok(Record {
        id: r.get(0)?,
        sha256: r.get(1)?,
        filename: r.get(2)?,
        model_version: r.get(3)?,
        label: r.get(4)?,
        confidence: r.get(5)?,
        margin: r.get(6)?,
        status: r.get(7)?,
        probs: serde_json::from_str(&r.get::<_, String>(8)?).unwrap_or(Value::Null),
        created_at: r.get(9)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(label: &str, confidence: f32) -> Prediction {
        Prediction { label: label.into(), confidence, margin: confidence, probs: vec![(label.into(), confidence)] }
    }

    #[test]
    fn insert_dedup_and_search() {
        let db = Db::open(":memory:").unwrap();
        let a = db.insert("aaa", Some("a.png"), "v1", &pred("Forest", 0.9)).unwrap();
        let again = db.insert("aaa", Some("a.png"), "v1", &pred("Forest", 0.9)).unwrap();
        assert_eq!(a.id, again.id); // no duplicate row
        db.insert("bbb", None, "v1", &pred("River", 0.3)).unwrap();

        let forest = db.search(&Search { label: Some("Forest".into()), ..Default::default() }).unwrap();
        assert_eq!(forest.len(), 1);
        let unsure = db.search(&Search { status: Some("needs_review".into()), ..Default::default() }).unwrap();
        assert_eq!(unsure[0].label, "River");
        assert!(db.find("bbb", "v2").unwrap().is_none()); // a new model version = not seen yet
    }
}
