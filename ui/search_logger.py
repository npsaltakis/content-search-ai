"""
Lightweight search logger — writes to search_logs table in SQLite.
Uses try/except everywhere so it can NEVER crash the UI.
"""

import sqlite3
from pathlib import Path

# Same relative path used by the rest of the app
_DEFAULT_DB = Path(__file__).resolve().parents[1] / "content_search_ai.db"


def log_search(query: str, modality: str, db_path: str = str(_DEFAULT_DB)):
    """Insert one row into search_logs. Silently swallows all errors."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO search_logs (query, modality) VALUES (?, ?)",
            (query.strip(), modality),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # never break the UI
