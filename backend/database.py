from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

# A single SQLAlchemy instance shared across the app
# Models import this object to declare their tables.
db = SQLAlchemy()


def ensure_sqlite_schema(db):
    """Lightweight, in-app schema alignment for SQLite deployments.

    Older demo databases might miss new columns (e.g., rut/list_tag on persons).
    This helper issues ALTER TABLE statements when needed so the app can start
    without requiring an external migration tool.
    """

    engine = db.engine
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)

    def _ensure_columns(table: str, columns: dict[str, str]):
        existing = {col["name"] for col in inspector.get_columns(table)}
        for name, ddl in columns.items():
            if name in existing:
                continue
            # SQLAlchemy 2.x removed engine.execute; use an explicit connection.
            # Include the column name in the DDL to avoid invalid statements such as
            # "ADD COLUMN VARCHAR(50)" when upgrading legacy databases.
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    # Align the persons table with new optional fields
    _ensure_columns(
        "persons",
        {
            "rut": "VARCHAR(50)",
            "role": "VARCHAR(120)",
            "list_tag": "VARCHAR(50)",
        },
    )

