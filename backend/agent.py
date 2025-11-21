"""Agente local de reconocimiento para cámaras RTSP/Webcam.

Permite probar en local el pipeline de reconocimiento: carga embeddings
desde la base, abre la cámara indicada y genera alertas cuando encuentra
coincidencias bajo el umbral configurado.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import List

import cv2

from sqlalchemy.orm import joinedload
from .app import create_app
from .database import db
from .models import Alert, Camera, Person, Site
from .recognition import build_person_bank_from_persons, recognize_faces, save_incremental_sample


def load_bank_from_db() -> List:
    persons = (
        Person.query.options(
            joinedload(Person.photos),
            joinedload(Person.embeddings),
        )
        .all()
    )
    return build_person_bank_from_persons(persons)


def ensure_local_camera() -> int:
    """Garantiza una cámara y sitio por defecto para registrar alertas.

    Esto evita errores de restricción NOT NULL cuando el agente se ejecuta sin
    especificar un ``camera_id`` explícito.
    """

    site = Site.query.filter_by(name="Agente Local").first()
    if not site:
        site = Site(name="Agente Local", location="Local")
        db.session.add(site)
        db.session.commit()

    camera = (
        Camera.query.filter_by(name="Cámara Local", site_id=site.id).first()
    )
    if not camera:
        camera = Camera(name="Cámara Local", rtsp_url="local", site_id=site.id)
        db.session.add(camera)
        db.session.commit()

    return camera.id


def run_camera(
    camera_source: str | int,
    camera_id: int | None,
    threshold: float = 0.38,
    margin: float = 0.08,
    spoof_threshold: float = 0.5,
    incremental_threshold: float = 0.85,
    enable_incremental: bool = True,
    top_k: int = 5,
):
    app = create_app()
    with app.app_context():
        bank = load_bank_from_db()
        print(f"Embeddings cargados: {len(bank)}")

        resolved_camera_id = camera_id or ensure_local_camera()

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara/stream")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame, saliendo...")
                break

            detections = recognize_faces(
                frame,
                bank,
                threshold=threshold,
                margin=margin,
                top_k=top_k,
                live_check=True,
                spoof_threshold=spoof_threshold,
            )

            for det in detections:
                (top, right, bottom, left) = det["bbox"]
                match = det["match"]
                similarity = det.get("similarity") or 0.0
                label = "Spoof" if not det.get("live", True) else "Desconocido"
                color = (0, 0, 255)

                if match and similarity is not None:
                    label = f"{match.person_name} ({similarity*100:.1f}%)"
                    color = (0, 255, 0)

                    alert = Alert(
                        similarity=similarity,
                        person_id=match.person_id,
                        camera_id=resolved_camera_id,
                        message="Match detectado por agente local",
                        created_at=datetime.utcnow(),
                    )
                    db.session.add(alert)
                    db.session.commit()

                    if enable_incremental and similarity >= incremental_threshold:
                        face_crop = frame[top:bottom, left:right]
                        save_incremental_sample(
                            match.person_id,
                            face_crop,
                            det.get("embedding", []),
                            app.static_folder,
                            quality=det.get("metrics", {}).get("score", 1.0),
                            note="agent-auto",
                        )

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Agente FaceHub", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                bank = load_bank_from_db()
                print(f"Banco recargado. Embeddings: {len(bank)}")

        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Agente local de reconocimiento")
    parser.add_argument("--camera", default=0, help="Índice de webcam o URL RTSP")
    parser.add_argument("--camera-id", type=int, default=None, help="ID de cámara para registrar alertas")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        help="Distancia coseno máxima permitida (menor = más estricto)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Diferencia mínima frente al segundo mejor candidato para evitar falsos positivos",
    )
    parser.add_argument(
        "--spoof-threshold",
        type=float,
        default=0.5,
        help="Umbral mínimo de score anti-spoof (0-1).",
    )
    parser.add_argument(
        "--incremental-threshold",
        type=float,
        default=0.85,
        help="A partir de qué similitud guardar muestras incrementales",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Desactiva el guardado automático de nuevas fotos",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    run_camera(
        camera_source=source,
        camera_id=args.camera_id,
        threshold=args.threshold,
        margin=args.margin,
        spoof_threshold=args.spoof_threshold,
        incremental_threshold=args.incremental_threshold,
        enable_incremental=not args.no_incremental,
    )
