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

from .app import create_app
from .database import db
from .models import Alert, Camera, Person, Site
from .recognition import best_match, build_bank, detect_and_encode


def load_bank_from_db() -> List:
    records = []
    persons = Person.query.all()
    for person in persons:
        for emb in person.embeddings:
            records.append((person.id, person.full_name, emb.vector))
    return build_bank(records)


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


def run_camera(camera_source: str | int, camera_id: int | None, tolerance: float = 0.45):
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

            detections = detect_and_encode(frame)

            for (top, right, bottom, left), encoding in detections:
                match, dist = best_match(encoding, bank, tolerance=tolerance)
                label = "Desconocido"
                color = (0, 0, 255)
                similarity = 0.0

                if match and dist is not None:
                    similarity = max(0.0, 1.0 - float(dist))
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
    parser.add_argument("--tolerance", type=float, default=0.45, help="Umbral de distancia para match")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    run_camera(camera_source=source, camera_id=args.camera_id, tolerance=args.tolerance)
