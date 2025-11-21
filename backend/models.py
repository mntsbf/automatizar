from __future__ import annotations

from datetime import datetime
from typing import Optional

from .database import db


class Site(db.Model):
    __tablename__ = "sites"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(255), nullable=True)
    cameras = db.relationship("Camera", backref="site", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "location": self.location}


class Camera(db.Model):
    __tablename__ = "cameras"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    rtsp_url = db.Column(db.String(255), nullable=False)
    site_id = db.Column(db.Integer, db.ForeignKey("sites.id"), nullable=False)
    alerts = db.relationship("Alert", backref="camera", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rtsp_url": self.rtsp_url,
            "site_id": self.site_id,
            "site_name": self.site.name if self.site else None,
        }


class Person(db.Model):
    __tablename__ = "persons"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(120), nullable=True)
    embeddings = db.relationship("Embedding", backref="person", cascade="all, delete-orphan")

    @property
    def risk_level(self) -> str:
        role_lower = (self.role or "").lower()
        if "negra" in role_lower or "roja" in role_lower or "black" in role_lower:
            return "critical"
        if "gris" in role_lower or "watch" in role_lower or "observ" in role_lower:
            return "warning"
        return "info"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "role": self.role,
            "risk_level": self.risk_level,
            "embeddings": [e.to_dict() for e in self.embeddings],
        }


class Embedding(db.Model):
    __tablename__ = "embeddings"

    id = db.Column(db.Integer, primary_key=True)
    vector = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(80), default="ArcFace")
    person_id = db.Column(db.Integer, db.ForeignKey("persons.id"), nullable=False)

    def to_dict(self) -> dict:
        return {"id": self.id, "vector": self.vector, "model": self.model}


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    similarity = db.Column(db.Float, nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey("persons.id"), nullable=True)
    camera_id = db.Column(db.Integer, db.ForeignKey("cameras.id"), nullable=False)
    message = db.Column(db.String(255), nullable=False)

    person = db.relationship("Person")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "similarity": self.similarity,
            "person": self.person.to_dict() if self.person else None,
            "risk_level": self.person.risk_level if self.person else "info",
            "camera": self.camera.to_dict() if self.camera else None,
            "message": self.message,
        }
