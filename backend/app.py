from __future__ import annotations

import io
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Iterable, List

import cv2
import numpy as np

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from flask_cors import CORS
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename
from .database import db, ensure_sqlite_schema
from .models import Alert, Camera, Embedding, FacePhoto, Person, Setting, Site
from .recognition import (
    build_person_bank_from_persons,
    detect_and_encode,
    encode_image_array,
    parse_embedding,
    robust_detect_and_encode,
    recognize_faces,
    save_incremental_sample,
)


def _risk_level(role: str | None) -> str:
    if not role:
        return "info"
    lower = role.lower()
    if "negra" in lower or "roja" in lower or "black" in lower:
        return "critical"
    if "gris" in lower or "watch" in lower or "observ" in lower:
        return "warning"
    return "info"


def _ensure_default_settings():
    defaults = {
        "tolerance": "0.38",
        "margin": "0.08",
        "live_refresh": "true",
        "theme": "dark",
        "spoof_threshold": "0.5",
        "incremental_threshold": "0.87",
        "top_k": "5",
    }
    for key, value in defaults.items():
        setting = Setting.query.get(key)
        if not setting:
            db.session.add(Setting(key=key, value=value))
    db.session.commit()


def _get_setting_value(key: str, default: str) -> str:
    setting = Setting.query.get(key)
    return setting.value if setting else default


_EMBEDDING_BANK_CACHE: dict[str, object] = {"bank": [], "version": 0}


def _load_persons_with_embeddings() -> list[Person]:
    return (
        Person.query.options(
            joinedload(Person.embeddings),
            joinedload(Person.photos),
        )
        .all()
    )


def _rebuild_bank() -> list:
    persons = _load_persons_with_embeddings()
    bank = build_person_bank_from_persons(persons)
    _EMBEDDING_BANK_CACHE["bank"] = bank
    _EMBEDDING_BANK_CACHE["version"] = int(time.time())
    return bank


def _get_bank(force: bool = False):
    if force or not _EMBEDDING_BANK_CACHE["bank"]:
        return _rebuild_bank()
    return _EMBEDDING_BANK_CACHE["bank"]


def _faces_dir(app: Flask) -> str:
    faces_path = os.path.join(app.static_folder, "faces")
    os.makedirs(faces_path, exist_ok=True)
    return faces_path


def _save_face_photo(person: Person, storage, app: Flask, save_embedding_row: bool = False):
    """Procesa un FileStorage, genera embedding y persiste el archivo + DB."""

    filename = secure_filename(storage.filename or f"rostro_{uuid.uuid4().hex}.jpg")
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"

    data = np.frombuffer(storage.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, "No se pudo leer la imagen"

    spoof_threshold = float(_get_setting_value("spoof_threshold", "0.5"))
    detections = robust_detect_and_encode(bgr, live_check=True, spoof_threshold=spoof_threshold)
    if not detections:
        return None, None, "No se detectaron rostros válidos (anti-spoofing)"

    det0 = detections[0]
    (top, right, bottom, left) = det0["bbox"]
    embedding = det0["embedding"]
    h, w = bgr.shape[:2]
    face_area = (bottom - top) * (right - left)
    quality = min(1.0, (face_area / float(w * h)) * 4.0) * det0.get("metrics", {}).get("score", 1.0)
    metadata = {
        "box": [int(top), int(right), int(bottom), int(left)],
        "frame_size": [w, h],
        "source": storage.filename,
    }

    unique_name = f"{person.id}_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(_faces_dir(app), unique_name)
    cv2.imwrite(file_path, bgr)
    rel_path = os.path.relpath(file_path, app.static_folder)

    photo = FacePhoto(
        person=person,
        file_path=rel_path,
        embedding=json.dumps(embedding),
        quality=quality,
        metadata_json=json.dumps(metadata),
    )
    db.session.add(photo)

    emb_row = None
    if save_embedding_row:
        emb_row = Embedding(vector=json.dumps(embedding), model="face_recognition", person=person)
        db.session.add(emb_row)

    return photo, emb_row, None


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
        ensure_sqlite_schema(db)
        _ensure_default_settings()
        _rebuild_bank()

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html", user_name="Administrador", user_role="Admin")

    @app.route("/alertas")
    def alertas_view():
        return render_template("alerts.html", user_name="Operador", user_role="Seguridad")

    @app.route("/camaras")
    def camaras_view():
        return render_template("cameras.html", user_name="Administrador", user_role="Admin")

    @app.route("/personas")
    def personas_view():
        return render_template("persons.html", user_name="Administrador", user_role="Admin")

    @app.route("/config")
    def config_view():
        return render_template("config.html", user_name="Administrador", user_role="Admin")

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
        person = Person(
            full_name=payload.get("full_name", "Desconocido"),
            role=payload.get("role"),
            rut=payload.get("rut"),
            list_tag=payload.get("list_tag"),
        )
        db.session.add(person)
        db.session.commit()
        embeddings: List[str] = payload.get("embeddings", [])
        for vector in embeddings:
            parsed = parse_embedding(vector)
            if parsed:
                emb = Embedding(vector=json.dumps(parsed), model=payload.get("model", "ArcFace"), person=person)
                db.session.add(emb)
        db.session.commit()
        _rebuild_bank()
        return jsonify(person.to_dict()), 201

    @app.post("/api/persons/<int:person_id>/embedding")
    def add_person_embedding(person_id: int):
        person = Person.query.get_or_404(person_id)
        if "image" not in request.files:
            return {"error": "Debes enviar un archivo 'image'"}, 400

        image_file = request.files["image"]
        photo, emb_row, error = _save_face_photo(person, image_file, app, save_embedding_row=True)
        if error:
            return {"error": error}, 400

        db.session.commit()
        _rebuild_bank()
        return person.to_dict(), 201

    @app.post("/api/persons/<int:person_id>/photos")
    def upload_person_photos(person_id: int):
        person = Person.query.get_or_404(person_id)
        files = request.files.getlist("images") or request.files.getlist("photos")
        if not files:
            return {"error": "Incluye archivos en 'images' o 'photos'"}, 400

        created: list[dict] = []
        errors: list[dict] = []
        for storage in files:
            photo, _, error = _save_face_photo(person, storage, app, save_embedding_row=False)
            if error:
                errors.append({"filename": storage.filename, "error": error})
                continue
            created.append(photo.to_dict())

        if created:
            db.session.commit()
            _rebuild_bank()
        status = 201 if created else 400
        return {"created": len(created), "photos": created, "errors": errors}, status

    @app.post("/api/recognize")
    def recognize_from_image():
        if "image" not in request.files:
            return {"error": "Debes enviar un archivo 'image'"}, 400

        image_file = request.files["image"]
        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400
        threshold = float(
            request.form.get(
                "threshold",
                request.form.get(
                    "distance_threshold",
                    _get_setting_value("tolerance", "0.38"),
                ),
            )
        )
        margin = float(request.form.get("margin", _get_setting_value("margin", "0.08")))
        spoof_threshold = float(
            request.form.get("spoof_threshold", _get_setting_value("spoof_threshold", "0.5"))
        )
        incremental_threshold = float(
            request.form.get(
                "incremental_threshold",
                _get_setting_value("incremental_threshold", "0.87"),
            )
        )
        top_k = int(request.form.get("top_k", _get_setting_value("top_k", "5")))
        live_check = request.form.get("live_check", "true").lower() == "true"
        save_incremental = request.form.get("save_incremental", "false").lower() == "true"

        bank = _get_bank()
        matches = recognize_faces(
            bgr,
            bank,
            threshold=threshold,
            margin=margin,
            top_k=top_k,
            live_check=live_check,
            spoof_threshold=spoof_threshold,
        )

        results = []
        saved_any = False
        camera_id = request.form.get("camera_id")
        camera_ref = int(camera_id) if camera_id else None

        for det in matches:
            top, right, bottom, left = det["bbox"]
            match = det.get("match")
            similarity = det.get("similarity") or 0.0
            live_ok = det.get("live", True)

            if match and similarity is not None:
                alert = Alert(
                    similarity=similarity,
                    person_id=match.person_id,
                    camera_id=camera_ref,
                    message=request.form.get("message", "Alerta desde imagen"),
                )
                db.session.add(alert)

                if save_incremental and similarity >= incremental_threshold and live_ok:
                    face_crop = bgr[top:bottom, left:right]
                    photo = save_incremental_sample(
                        match.person_id,
                        face_crop,
                        det.get("embedding", []),
                        app.static_folder,
                        quality=det.get("metrics", {}).get("score", 1.0),
                        note="api-auto",
                    )
                    saved_any = saved_any or bool(photo)

                results.append(
                    {
                        "person": match.person_name,
                        "person_id": match.person_id,
                        "similarity": similarity,
                        "diagnostics": det.get("diagnostics"),
                        "live": live_ok,
                        "spoof_score": det.get("metrics", {}).get("score"),
                    }
                )
            else:
                results.append({"person": None, "similarity": similarity, "person_id": None, "live": live_ok})

        db.session.commit()
        if saved_any:
            _rebuild_bank()
        return {"count": len(results), "matches": results}

    @app.post("/api/embeddings/refresh")
    def refresh_embeddings():
        bank = _rebuild_bank()
        total = int(sum(entry.vectors.shape[0] for entry in bank)) if bank else 0
        return {"persons": len(bank), "embeddings": total, "version": _EMBEDDING_BANK_CACHE["version"]}

    # --- Dashboard data ---
    @app.get("/api/dashboard/estadisticas")
    def dashboard_stats():
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        week_start = today_start - timedelta(days=6)

        alerts_today_q = Alert.query.filter(Alert.created_at >= today_start)
        alerts_today = alerts_today_q.count()

        role_expr = func.lower(func.coalesce(Person.list_tag, Person.role, ""))
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
            role = alert.person.list_tag or alert.person.role if alert.person else None
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
        risk = request.args.get("risk")
        site_id = request.args.get("site_id", type=int)
        person_id = request.args.get("person_id", type=int)
        limit = request.args.get("limit", type=int, default=100)

        query = Alert.query.order_by(Alert.created_at.desc())
        if person_id:
            query = query.filter(Alert.person_id == person_id)
        if site_id:
            query = query.join(Camera, Alert.camera_id == Camera.id).filter(Camera.site_id == site_id)
        if risk:
            risk = risk.lower()
            query = query.join(Person, Alert.person_id == Person.id, isouter=True)
            tag_expr = func.lower(func.coalesce(Person.list_tag, Person.role, ""))
            if risk == "critical":
                query = query.filter(tag_expr.contains("negra") | tag_expr.contains("roja"))
            elif risk == "warning":
                query = query.filter(tag_expr.contains("gris") | tag_expr.contains("watch"))

        alerts = query.limit(limit).all()
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

    # --- Settings ---
    @app.get("/api/settings")
    def get_settings():
        settings = {s.key: s.value for s in Setting.query.all()}
        return settings

    @app.put("/api/settings")
    def update_settings():
        payload = request.get_json() or {}
        updated = {}
        for key, value in payload.items():
            setting = Setting.query.get(key)
            if not setting:
                setting = Setting(key=key, value=str(value))
                db.session.add(setting)
            else:
                setting.value = str(value)
            updated[key] = setting.value
        db.session.commit()
        return {"updated": updated}

    # --- Healthcheck for remote agents ---
    @app.get("/api/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
