"""
SQLite storage for users, uploads, evaluations and the audit log.

Each call opens a short-lived connection, so the services work the same inside
a Flask request, in the CLI and in tests. The schema is created on first use.
"""
import json
import sqlite3
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('analyst', 'admin')),
    status        TEXT NOT NULL CHECK (status IN ('active', 'suspended')),
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id                 TEXT PRIMARY KEY,
    owner_id           TEXT NOT NULL REFERENCES users(id),
    original_filename  TEXT NOT NULL,
    stored_filename    TEXT NOT NULL,
    size_bytes         INTEGER NOT NULL,
    uploaded_at        TEXT NOT NULL,
    validation_status  TEXT NOT NULL,
    validation_message TEXT,
    required_inputs    TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id                 TEXT PRIMARY KEY,
    owner_id           TEXT NOT NULL REFERENCES users(id),
    original_filename  TEXT NOT NULL,
    stored_filename    TEXT NOT NULL,
    row_count          INTEGER NOT NULL,
    columns            TEXT NOT NULL,
    target_variable    TEXT NOT NULL,
    uploaded_at        TEXT NOT NULL,
    validation_status  TEXT NOT NULL,
    validation_message TEXT,
    schema_summary     TEXT
);

-- Evaluations keep copies of the file names so history and reports still
-- work after a model or dataset is deleted.
CREATE TABLE IF NOT EXISTS evaluations (
    id               TEXT PRIMARY KEY,
    owner_id         TEXT NOT NULL REFERENCES users(id),
    model_id         TEXT NOT NULL,
    model_filename   TEXT NOT NULL,
    model_type       TEXT,
    dataset_id       TEXT NOT NULL,
    dataset_filename TEXT NOT NULL,
    target_variable  TEXT NOT NULL,
    task_type        TEXT NOT NULL,
    row_count        INTEGER NOT NULL,
    metrics          TEXT NOT NULL,
    notes            TEXT NOT NULL,
    charts           TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    evaluated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id     TEXT,
    actor_email  TEXT,
    target_id    TEXT NOT NULL,
    target_email TEXT NOT NULL,
    action       TEXT NOT NULL,
    old_value    TEXT,
    new_value    TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_owner ON models(owner_id);
CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets(owner_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_owner ON evaluations(owner_id);
"""

# Columns stored as JSON text, per table.
JSON_COLUMNS = {
    "models": {"required_inputs"},
    "datasets": {"columns", "schema_summary"},
    "evaluations": {"metrics", "notes", "charts"},
}

_initialised = set()


def _connect():
    path = config.DATABASE_PATH
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path not in _initialised:
        conn.executescript(SCHEMA)
        _initialised.add(path)
    return conn


@contextmanager
def connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _encode(table, values):
    json_cols = JSON_COLUMNS.get(table, set())
    return {k: (json.dumps(v) if k in json_cols and v is not None else v) for k, v in values.items()}


def row_to_dict(table, row):
    if row is None:
        return None
    json_cols = JSON_COLUMNS.get(table, set())
    return {k: (json.loads(row[k]) if k in json_cols and row[k] is not None else row[k]) for k in row.keys()}


def insert(table, values):
    values = _encode(table, values)
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with connection() as conn:
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))


def update(table, record_id, values):
    values = _encode(table, values)
    sets = ", ".join(f"{k} = ?" for k in values)
    with connection() as conn:
        conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), record_id])


def fetch_one(table, where, params):
    with connection() as conn:
        row = conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchone()
    return row_to_dict(table, row)


def fetch_all(table, where="1 = 1", params=(), order="rowid"):
    with connection() as conn:
        rows = conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY {order}", params).fetchall()
    return [row_to_dict(table, r) for r in rows]


def delete(table, record_id):
    with connection() as conn:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))


def count(table, where="1 = 1", params=()):
    with connection() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]
