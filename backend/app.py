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
    current_app,
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
    analyze_scan_frame,
    best_person_match,
    build_person_bank_from_persons,
    detect_and_encode,
    encode_image_array,
    has_heavy_embedding_backend,
    best_similarity_no_threshold,
    merge_scan_embeddings,
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
    # Si no hay backend pesado (dlib/ArcFace), subimos la tolerancia/margen
    # mínimos para que el modo liviano (OpenCV) no descarte coincidencias
    # válidas por embeddings menos discriminativos.
    if not has_heavy_embedding_backend():
        tol = Setting.query.get("tolerance")
        if tol is None:
            db.session.add(Setting(key="tolerance", value="0.6"))
        else:
            try:
                if float(tol.value) < 0.55:
                    tol.value = "0.6"
            except ValueError:
                tol.value = "0.6"

        margin = Setting.query.get("margin")
        if margin is None:
            db.session.add(Setting(key="margin", value="0.12"))
        else:
            try:
                if float(margin.value) < 0.1:
                    margin.value = "0.12"
            except ValueError:
                margin.value = "0.12"
    db.session.commit()


def _get_setting_value(key: str, default: str) -> str:
    setting = Setting.query.get(key)
    return setting.value if setting else default


_EMBEDDING_BANK_CACHE: dict[str, object] = {"bank": [], "version": 0}


def _default_match_threshold() -> float:
    """Devuelve un umbral adecuado según el backend de embeddings activo."""

    base = float(_get_setting_value("tolerance", "0.38"))
    if not has_heavy_embedding_backend():
        # Con embeddings livianos (OpenCV) necesitamos una tolerancia mayor.
        base = max(base, 0.6)
    return base


def _default_margin() -> float:
    base = float(_get_setting_value("margin", "0.08"))
    if not has_heavy_embedding_backend():
        base = max(base, 0.12)
    return base


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
    total = int(sum(entry.vectors.shape[0] for entry in bank)) if bank else 0
    try:
        current_app.logger.info(
            "Banco de embeddings reconstruido",
            extra={"persons": len(bank), "embeddings": total, "version": _EMBEDDING_BANK_CACHE["version"]},
        )
    except Exception:
        pass
    return bank


def _get_bank(force: bool = False):
    if force or not _EMBEDDING_BANK_CACHE["bank"]:
        return _rebuild_bank()
    return _EMBEDDING_BANK_CACHE["bank"]


def _safe_bbox(raw_bbox):
    if not raw_bbox:
        return None
    try:
        return [int(x) for x in raw_bbox]
    except Exception:
        try:
            return list(raw_bbox)
        except Exception:
            return None


def _faces_dir(app: Flask) -> str:
    faces_path = os.path.join(app.static_folder, "faces")
    os.makedirs(faces_path, exist_ok=True)
    return faces_path


def _lookup_embedding(
    embedding: list[float],
    *,
    threshold: float | None = None,
    margin: float | None = None,
    top_k: int | None = None,
    force_bank: bool = False,
    bank_override=None,
):
    """Busca una coincidencia contra el banco actual y agrega trazas de depuración."""

    bank = bank_override if bank_override is not None else _get_bank(force=force_bank)
    info = {
        "bank_version": _EMBEDDING_BANK_CACHE.get("version"),
        "bank_size": len(bank),
    }
    if not bank:
        current_app.logger.warning("Banco vacío al buscar persona", extra=info)
        return None, None, None, info

    thr = threshold if threshold is not None else _default_match_threshold()
    mar = margin if margin is not None else _default_margin()
    k = top_k if top_k is not None else int(_get_setting_value("top_k", "5"))

    entry, similarity, diagnostics = best_person_match(embedding, bank, threshold=thr, margin=mar, top_k=k)
    fallback_sim = None
    raw_best = best_similarity_no_threshold(embedding, bank)

    if entry and similarity is not None:
        current_app.logger.info(
            "Match encontrado",
            extra={
                "person_id": entry.person_id,
                "person_name": entry.person_name,
                "similarity": round(float(similarity), 4),
                "distance": round(float(1.0 - similarity), 4),
                "bank_size": len(bank),
                "threshold": thr,
                "margin": mar,
            },
        )
        return entry, similarity, diagnostics, info

    # No se aceptó match: buscamos la mejor similitud para reportar
    if raw_best:
        fallback_sim = raw_best.get("similarity")
        # En modo liviano podemos relajar el umbral: los embeddings son menos
        # discriminativos y pequeños cambios de iluminación dan distancias
        # mayores. Permitimos un colchón adicional de 0.25 sobre el umbral.
        if not has_heavy_embedding_backend():
            relaxed = thr + 0.25
            if raw_best.get("distance", 1.0) <= relaxed:
                current_app.logger.info(
                    "Match aceptado en modo relajado (light backend)",
                    extra={
                        "person_id": raw_best["entry"].person_id,
                        "similarity": round(float(raw_best["similarity"]), 4),
                        "distance": round(float(raw_best["distance"]), 4),
                        "threshold": thr,
                        "relaxed": relaxed,
                        "bank_size": len(bank),
                    },
                )
                return raw_best["entry"], raw_best["similarity"], diagnostics or {"mode": "relaxed"}, info
    else:
        _, fallback_sim, _ = best_person_match(embedding, bank, threshold=1.0, margin=0.0, top_k=k)
    current_app.logger.info(
        "Sin match bajo umbral",
        extra={
            "best_similarity": float(fallback_sim or 0.0),
            "threshold": thr,
            "margin": mar,
            "bank_size": len(bank),
        },
    )
    return None, fallback_sim, diagnostics, info


def _save_face_photo(
    person: Person,
    storage,
    app: Flask,
    save_embedding_row: bool = False,
    live_check: bool = False,
):
    """Procesa un FileStorage, genera embedding y persiste el archivo + DB.

    ``live_check`` se deja desactivado por defecto para no bloquear la carga de
    galerías históricas: algunos clasificadores de anti-spoofing pueden
    penalizar fotos fijas aunque sean legítimas.
    """

    filename = secure_filename(storage.filename or f"rostro_{uuid.uuid4().hex}.jpg")
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"

    data = np.frombuffer(storage.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, "No se pudo leer la imagen"

    spoof_threshold = float(_get_setting_value("spoof_threshold", "0.5"))
    detections = robust_detect_and_encode(
        bgr,
        live_check=live_check,
        spoof_threshold=spoof_threshold,
    )
    if not detections:
        app.logger.warning(
            "Carga de foto sin rostros válidos",
            extra={"person_id": person.id, "filename": storage.filename, "live_check": live_check},
        )
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
        registered_at=datetime.utcnow(),
    )
    db.session.add(photo)

    emb_row = None
    if save_embedding_row:
        emb_row = Embedding(vector=json.dumps(embedding), model="face_recognition", person=person)
        db.session.add(emb_row)

    app.logger.info(
        "Foto de rostro guardada",
        extra={
            "person_id": person.id,
            "filename": storage.filename,
            "path": rel_path,
            "quality": round(float(quality), 4),
            "bbox": [int(top), int(right), int(bottom), int(left)],
        },
    )
    return photo, emb_row, None


def _save_scan_photo(
    person: Person,
    frame_bgr: np.ndarray,
    embedding: list[float],
    app: Flask,
    *,
    note: str = "scan",
    bbox: list[int] | tuple[int, int, int, int] | None = None,
    quality: float | None = None,
):
    """Guarda un frame capturado durante el escaneo guiado."""

    unique_name = f"scan_{person.id}_{uuid.uuid4().hex}.jpg"
    faces_dir = _faces_dir(app)
    file_path = os.path.join(faces_dir, unique_name)
    cv2.imwrite(file_path, frame_bgr)
    rel_path = os.path.relpath(file_path, app.static_folder)

    metadata = {"box": _safe_bbox(bbox), "note": note}
    photo = FacePhoto(
        person=person,
        file_path=rel_path,
        embedding=json.dumps(embedding),
        quality=quality or 0.8,
        metadata_json=json.dumps(metadata),
        registered_at=datetime.utcnow(),
    )
    db.session.add(photo)
    app.logger.info(
        "Frame de escaneo guardado",
        extra={"person_id": person.id, "path": rel_path, "bbox": metadata["box"], "quality": quality},
    )
    return photo


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

    @app.route("/buscar-persona")
    def buscar_persona_view():
        return render_template(
            "lookup.html",
            user_name="Operador",
            user_role="Verificación",
            tolerance_default=_get_setting_value("tolerance", "0.38"),
            margin_default=_get_setting_value("margin", "0.08"),
            bank_version=_EMBEDDING_BANK_CACHE.get("version", 0),
        )

    @app.route("/scan")
    def scan_view():
        return render_template(
            "scanner.html",
            user_name="Operador",
            user_role="Liveness",
            tolerance_default=_get_setting_value("tolerance", "0.38"),
        )

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
        live_check = request.form.get("live_check", "false").lower() == "true"
        photo, emb_row, error = _save_face_photo(
            person,
            image_file,
            app,
            save_embedding_row=True,
            live_check=live_check,
        )
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

        live_check = request.form.get("live_check", "false").lower() == "true"

        created: list[dict] = []
        errors: list[dict] = []
        for storage in files:
            photo, _, error = _save_face_photo(
                person,
                storage,
                app,
                save_embedding_row=False,
                live_check=live_check,
            )
            if error:
                errors.append({"filename": storage.filename, "error": error})
                continue
            created.append(photo.to_dict())

        if created:
            db.session.commit()
            _rebuild_bank()
        status = 201 if created else 400
        return {"created": len(created), "photos": created, "errors": errors}, status

    @app.post("/api/scan/frame")
    def scan_frame():
        """Valida un paso guiado (giro/sonrisa) y devuelve embedding si es válido."""

        image_file = request.files.get("image") or request.files.get("frame")
        if not image_file:
            return {"error": "Incluye un archivo en 'image' o 'frame'"}, 400

        action = request.form.get("action", "frente")
        live_check = request.form.get("live_check", "true").lower() == "true"
        spoof_threshold = float(request.form.get("spoof_threshold", _get_setting_value("spoof_threshold", "0.5")))
        quality_threshold = float(request.form.get("quality_threshold", "0.45"))

        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400

        result = analyze_scan_frame(
            bgr,
            action,
            live_check=live_check,
            spoof_threshold=spoof_threshold,
            quality_threshold=quality_threshold,
        )
        result["bbox"] = _safe_bbox(result.get("bbox"))
        result["action"] = action

        if result.get("reason") == "accion_no_reconocida":
            return {"error": "Acción no soportada", "reason": result.get("reason")}, 400

        person_id = request.form.get("person_id", type=int)
        save_photo = request.form.get("save_photo", "false").lower() == "true"

        if result.get("ok") and person_id and save_photo:
            person = Person.query.get(person_id)
            if person:
                _save_scan_photo(
                    person,
                    bgr,
                    result.get("embedding", []),
                    app,
                    note=f"scan:{action}",
                    bbox=result.get("bbox"),
                    quality=result.get("quality"),
                )
                db.session.commit()
                _rebuild_bank()

        status = 200 if result.get("ok") else 202
        return result, status

    @app.post("/api/scan/finalize")
    def finalize_scan():
        payload = request.get_json() or {}
        embeddings = payload.get("embeddings") or []
        if not embeddings:
            return {"error": "Incluye una lista de embeddings"}, 400

        mode = payload.get("mode", "median")
        merged = merge_scan_embeddings(embeddings, mode=mode)
        if not merged:
            return {"error": "No se pudo combinar embeddings"}, 400

        person_id = payload.get("person_id")
        saved_id = None
        if person_id:
            person = Person.query.get_or_404(int(person_id))
            row = Embedding(vector=json.dumps(merged), model=f"scan-{mode}", person=person)
            db.session.add(row)
            db.session.commit()
            saved_id = row.id
            _rebuild_bank()

        return {
            "embedding": merged,
            "count": len(embeddings),
            "mode": mode,
            "saved_embedding_id": saved_id,
        }, 201

    @app.post("/buscar-persona")
    def buscar_persona():
        """Busca coincidencias usando la foto subida y el banco actual."""

        image_file = request.files.get("image") or request.files.get("photo")
        if not image_file:
            return {"error": "Incluye un archivo en 'image' o 'photo'"}, 400

        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400

        # Permite forzar la recarga del banco en cada consulta (útil si hubo
        # subidas recientes y el proceso corre con múltiples workers).
        force_bank = request.form.get("refresh_bank", "true").lower() == "true"
        bank = _get_bank(force=force_bank)
        if not bank:
            return {
                "resultado": "desconocido",
                "mensaje": "No hay embeddings registrados",
                "bank_version": _EMBEDDING_BANK_CACHE.get("version"),
                "bank_size": 0,
            }, 404

        threshold_raw = request.form.get("threshold") or request.form.get("distance_threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else _default_match_threshold()
        margin_raw = request.form.get("margin")
        margin = float(margin_raw) if margin_raw is not None else _default_margin()
        top_k = int(request.form.get("top_k", _get_setting_value("top_k", "5")))
        spoof_threshold = float(request.form.get("spoof_threshold", _get_setting_value("spoof_threshold", "0.5")))
        live_check = request.form.get("live_check", "false").lower() == "true"

        detections = robust_detect_and_encode(
            bgr,
            live_check=live_check,
            spoof_threshold=spoof_threshold,
        )

        current_app.logger.info(
            "Detecciones en buscar-persona",
            extra={"count": len(detections), "threshold": threshold, "margin": margin, "bank_size": len(bank)},
        )

        if not detections:
            return {"resultado": "sin_rostro", "mensaje": "No se detectaron rostros válidos"}, 400

        best_payload: dict | None = None
        fallback_similarity = None
        for det in detections:
            entry, sim_or_fallback, diagnostics, info = _lookup_embedding(
                det["embedding"],
                threshold=threshold,
                margin=margin,
                top_k=top_k,
                bank_override=bank,
            )
            if entry and sim_or_fallback is not None:
                if not best_payload or sim_or_fallback > best_payload["similarity"]:
                    best_payload = {
                        "entry": entry,
                        "similarity": sim_or_fallback,
                        "diagnostics": diagnostics,
                        "bbox": _safe_bbox(det.get("bbox")),
                        "live": det.get("live", True),
                        "bank_info": info,
                    }
                continue

            if sim_or_fallback is not None:
                fallback_similarity = max(fallback_similarity or 0.0, sim_or_fallback)

        if best_payload:
            entry = best_payload["entry"]
            similarity = best_payload["similarity"]
            bank_info = best_payload.get("bank_info", {})
            return {
                "resultado": "match",
                "persona": {
                    "id": entry.person_id,
                    "nombre": entry.person_name,
                    "list_tag": entry.list_tag,
                    "confianza": round(float(similarity), 4),
                    "distancia": round(float(1.0 - similarity), 4),
                },
                "diagnosticos": best_payload.get("diagnostics"),
                "bbox": best_payload.get("bbox"),
                "live": best_payload.get("live", True),
                "bank_version": bank_info.get("bank_version", _EMBEDDING_BANK_CACHE.get("version")),
                "bank_size": bank_info.get("bank_size", len(bank)),
            }

        return {
            "resultado": "desconocido",
            "mensaje": "No se encontró una coincidencia bajo el umbral",
            "similaridad_maxima": round(float(fallback_similarity or 0.0), 4),
            "bank_version": _EMBEDDING_BANK_CACHE.get("version"),
            "bank_size": len(bank),
        }

    @app.post("/api/recognize")
    def recognize_from_image():
        if "image" not in request.files:
            return {"error": "Debes enviar un archivo 'image'"}, 400

        image_file = request.files["image"]
        data = np.frombuffer(image_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "No se pudo leer la imagen"}, 400
        threshold_raw = request.form.get("threshold") or request.form.get("distance_threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else _default_match_threshold()
        margin_raw = request.form.get("margin")
        margin = float(margin_raw) if margin_raw is not None else _default_margin()
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
        force_bank = request.form.get("refresh_bank", "true").lower() == "true"

        bank = _get_bank(force=force_bank)
        if not bank:
            return {
                "count": 0,
                "matches": [],
                "resultado": "sin_embeddings",
                "mensaje": "No hay embeddings en el banco",
                "bank_size": 0,
            }, 404

        current_app.logger.info(
            "Reconocer desde imagen",
            extra={
                "bank_size": len(bank),
                "threshold": threshold,
                "margin": margin,
                "live_check": live_check,
            },
        )

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
                results.append(
                    {
                        "person": None,
                        "similarity": similarity,
                        "person_id": None,
                        "live": live_ok,
                        "message": "Sin coincidencia bajo umbral",
                    }
                )

        db.session.commit()
        if saved_any:
            _rebuild_bank()
        return {
            "count": len(results),
            "matches": results,
            "bank_size": len(bank),
            "bank_version": _EMBEDDING_BANK_CACHE.get("version"),
        }

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
