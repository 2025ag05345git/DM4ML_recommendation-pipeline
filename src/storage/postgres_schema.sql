-- Reserved PostgreSQL schema for a future database-backed pipeline.
-- The current workflow reads/writes CSV files and does not execute this DDL.
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    age INTEGER,
    occupation TEXT,
    preferred_genre TEXT
);

CREATE TABLE IF NOT EXISTS movies (
    item_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    genre TEXT NOT NULL,
    release_year INTEGER,
    language TEXT NOT NULL,
    language_code TEXT NOT NULL,
    overview TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    item_id TEXT NOT NULL REFERENCES movies(item_id),
    rating NUMERIC(2, 1) NOT NULL CHECK (rating BETWEEN 1 AND 5),
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_user_time
    ON interactions (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_item
    ON interactions (item_id);

CREATE TABLE IF NOT EXISTS user_features (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    feature_payload JSONB NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS movie_features (
    item_id TEXT PRIMARY KEY REFERENCES movies(item_id),
    feature_payload JSONB NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
