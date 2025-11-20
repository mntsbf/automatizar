from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy

# A single SQLAlchemy instance shared across the app
# Models import this object to declare their tables.
db = SQLAlchemy()
