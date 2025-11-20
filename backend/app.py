from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import List

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from .database import db
from .models import Alert, Camera, Embedding, Person, Site
from .recognition import best_match, build_bank, detect_and_encode, encode_image_array, parse_embedding


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    default_db = f"sqlite:///{os.path.join(base_dir, 'facehub.db')}"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", default_db)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JSON_SORT_KEYS"] = False

    CORS(app)
    db.init_app(app)

    with app.app_context():
        db.create_all()

    @app.route("/")
    def index():
        return render_template("index.html")

    # --- Sites ---
    @app.get("/api/sites")
    def list_sites():
        sites = Site.query.all()
        return jsonify([s.to_dict() for s in sites])

    @app.post("/api/sites")
    def create_site():
        payload = request.get_json() or {}
        site = Site(name=payload.get("name", "Sitio sin nombre"), location=payload.get("location"))
        db.session.add(site)
        db.session.commit()
        return jsonify(site.to_dict()), 201

    # --- Cameras ---
    @app.get("/api/cameras")
    def list_cameras():
        cameras = Camera.query.all()
        return jsonify([c.to_dict() for c in cameras])

    @app.post("/api/cameras")
    def create_camera():
        payload = request.get_json() or {}
        camera = Camera(
            name=payload.get("name", "Cámara"),
            rtsp_url=payload.get("rtsp_url", "rtsp://demo"),
            site_id=payload.get("site_id"),
        )
        db.session.add(camera)
        db.session.commit()
        return jsonify(camera.to_dict()), 201

    # --- Persons ---
    @app.get("/api/persons")
    def list_persons():
        persons = Person.query.all()
        return jsonify([p.to_dict() for p in persons])

    @app.post("/api/persons")
    def create_person():
        payload = request.get_json() or {}
        person = Person(full_name=payload.get("full_name", "Desconocido"), role=payload.get("role"))
        db.session.add(person)
        db.session.commit()
        embeddings: List[str] = payload.get("embeddings", [])
        for vector in embeddings:
            parsed = parse_embedding(vector)
            if parsed:
                emb = Embedding(vector=json.dumps(parsed), model=payload.get("model", "ArcFace"), person=person)
                db.session.add(emb)
        db.session.commit()
        return jsonify(person.to_dict()), 201

    @app.post("/api/persons/<int:person_id>/embedding")
    def add_person_embedding(person_id: int):
        person = Person.query.get_or_404(person_id)
        if "image" not in request.files:
            return {"error": "Debes enviar un archivo 'image'"}, 400

        image_file = request.files["image"]
        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400

        embedding = encode_image_array(bgr)
        if embedding is None:
            return {
                "error": "No se detectaron rostros en la imagen o falta un backend de reconocimiento (usa face_recognition o el modo liviano con OpenCV)"
            }, 400

        emb = Embedding(vector=json.dumps(embedding), model="face_recognition", person=person)
        db.session.add(emb)
        db.session.commit()
        return person.to_dict(), 201

    @app.post("/api/recognize")
    def recognize_from_image():
        if "image" not in request.files:
            return {"error": "Debes enviar un archivo 'image'"}, 400

        image_file = request.files["image"]
        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400
        detections = detect_and_encode(bgr)

        records = []
        persons = Person.query.all()
        for person in persons:
            for emb in person.embeddings:
                records.append((person.id, person.full_name, emb.vector))
        bank = build_bank(records)

        results = []
        for (top, right, bottom, left), encoding in detections:
            match, dist = best_match(encoding, bank)
            if match and dist is not None:
                similarity = max(0.0, 1.0 - float(dist))
                camera_id = request.form.get("camera_id")
                camera_ref = int(camera_id) if camera_id else None
                alert = Alert(
                    similarity=similarity,
                    person_id=match.person_id,
                    camera_id=camera_ref,
                    message=request.form.get("message", "Alerta desde imagen"),
                )
                db.session.add(alert)
                db.session.commit()
                results.append({"person": match.person_name, "similarity": similarity, "person_id": match.person_id})
            else:
                results.append({"person": None, "similarity": 0.0, "person_id": None})

        return {"count": len(results), "matches": results}

    # --- Alerts ---
    @app.get("/api/alerts")
    def list_alerts():
        alerts = Alert.query.order_by(Alert.created_at.desc()).limit(50).all()
        return jsonify([a.to_dict() for a in alerts])

    @app.post("/api/alerts")
    def create_alert():
        payload = request.get_json() or {}
        alert = Alert(
            similarity=float(payload.get("similarity", 0.0)),
            person_id=payload.get("person_id"),
            camera_id=payload.get("camera_id"),
            message=payload.get("message", "Alerta recibida"),
        )
        db.session.add(alert)
        db.session.commit()
        return jsonify(alert.to_dict()), 201

    # --- Healthcheck for remote agents ---
    @app.get("/api/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
