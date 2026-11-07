"""Explicit future-integration boundary for PostgreSQL storage."""

from __future__ import annotations

import os


def database_url() -> str:
    """Read the future database URL without logging or exposing its value."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "PostgreSQL is not configured. CSV is the current storage backend; "
            "set DATABASE_URL only when implementing the database adapter."
        )
    return url


class PostgresStore:
    """Placeholder only: PostgreSQL persistence is intentionally not enabled."""

    def connect(self) -> None:
        database_url()
        raise NotImplementedError(
            "The PostgreSQL adapter is a placeholder. The active pipeline uses CSV; "
            "implement and test database transactions before enabling PostgreSQL."
        )
