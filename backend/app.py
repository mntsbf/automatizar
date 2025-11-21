from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timedelta
from typing import Iterable

import cv2
import numpy as np

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from sqlalchemy import func
from .database import db
from .models import Alert, Camera, Embedding, Person, Site
from .recognition import best_match, build_bank, detect_and_encode, encode_image_array, parse_embedding


def _risk_level(role: str | None) -> str:
    if not role:
        return "info"
    lower = role.lower()
    if "negra" in lower or "roja" in lower or "black" in lower:
        return "critical"
    if "gris" in lower or "watch" in lower or "observ" in lower:
        return "warning"
    return "info"


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    new_db_path = os.path.join(base_dir, "facehub.db")
    legacy_db_path = os.path.join(project_root, "facehub.db")

    default_db_path = new_db_path
    if not os.environ.get("DATABASE_URL") and os.path.exists(legacy_db_path) and not os.path.exists(new_db_path):
        default_db_path = legacy_db_path

    default_db = f"sqlite:///{default_db_path}"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", default_db)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JSON_SORT_KEYS"] = False

    CORS(app)
    db.init_app(app)

    with app.app_context():
        db.create_all()

    @app.route("/")
    def index():
        return render_template("dashboard.html", user_name="Administrador", user_role="Admin")

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
        tolerance = float(request.form.get("tolerance", 0.7))
        margin = float(request.form.get("margin", 0.1))

        detections = detect_and_encode(bgr)

        records = []
        persons = Person.query.all()
        for person in persons:
            for emb in person.embeddings:
                records.append((person.id, person.full_name, emb.vector))
        bank = build_bank(records)

        results = []
        for (top, right, bottom, left), encoding in detections:
            match, similarity = best_match(encoding, bank, tolerance=tolerance, margin=margin)
            if match and similarity is not None:
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

    # --- Dashboard data ---
    @app.get("/api/dashboard/estadisticas")
    def dashboard_stats():
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        week_start = today_start - timedelta(days=6)

        alerts_today_q = Alert.query.filter(Alert.created_at >= today_start)
        alerts_today = alerts_today_q.count()

        role_expr = func.lower(func.coalesce(Person.role, ""))
        critical_alerts = (
            alerts_today_q.join(Person, Alert.person_id == Person.id, isouter=True)
            .filter(
                role_expr.contains("negra")
                | role_expr.contains("roja")
                | role_expr.contains("black")
            )
            .count()
        )

        cameras_total = Camera.query.count()
        active_since = now - timedelta(hours=24)
        cameras_active = (
            db.session.query(func.count(func.distinct(Alert.camera_id)))
            .filter(Alert.created_at >= active_since)
            .scalar()
            or 0
        )

        matches_by_site: list[dict] = []
        site_rows: Iterable[tuple[str, int]] = (
            db.session.query(Site.name, func.count(Alert.id))
            .select_from(Site)
            .outerjoin(Camera, Camera.site_id == Site.id)
            .outerjoin(Alert, Alert.camera_id == Camera.id)
            .group_by(Site.id)
            .order_by(func.count(Alert.id).desc())
            .limit(8)
            .all()
        )
        for name, count in site_rows:
            matches_by_site.append({"site": name, "count": count})

        alerts_series = []
        total_series = 0
        for i in range(7):
            day_start = week_start + timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            cnt = (
                Alert.query.filter(Alert.created_at >= day_start)
                .filter(Alert.created_at < day_end)
                .count()
            )
            alerts_series.append({"label": day_start.strftime("%d/%m"), "count": cnt})
            total_series += cnt

        return {
            "alerts_today": alerts_today,
            "critical_alerts": critical_alerts,
            "cameras_total": cameras_total,
            "cameras_active": cameras_active,
            "cameras_inactive": max(cameras_total - cameras_active, 0),
            "matches_by_site": matches_by_site,
            "alerts_series": alerts_series,
            "series_total": total_series,
            "sites_with_alerts": len([m for m in matches_by_site if m["count"] > 0]),
        }

    @app.get("/api/dashboard/ultimas-alertas")
    def dashboard_latest_alerts():
        alerts = Alert.query.order_by(Alert.created_at.desc()).limit(25).all()
        serialized = []
        for alert in alerts:
            person_name = alert.person.full_name if alert.person else None
            role = alert.person.role if alert.person else None
            level = _risk_level(role)
            serialized.append(
                {
                    "id": alert.id,
                    "created_at": alert.created_at.isoformat(),
                    "site": alert.camera.site.name if alert.camera and alert.camera.site else None,
                    "camera": alert.camera.name if alert.camera else None,
                    "person": person_name,
                    "similarity": alert.similarity,
                    "risk_level": level,
                    "risk_label": "Crítica" if level == "critical" else "Advertencia" if level == "warning" else "Normal",
                    "thumbnail_url": None,
                }
            )
        return {"alerts": serialized}

    @app.get("/api/dashboard/tiempo-real")
    def dashboard_live():
        @stream_with_context
        def stream():
            last_id = None
            while True:
                latest = Alert.query.order_by(Alert.created_at.desc()).first()
                payload = {"type": "heartbeat"}
                if latest and latest.id != last_id:
                    last_id = latest.id
                    payload = {"type": "alert", "id": latest.id, "created_at": latest.created_at.isoformat()}
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(5)

        return Response(stream(), mimetype="text/event-stream")

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
