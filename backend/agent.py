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
from .models import Alert, Person
from .recognition import best_match, build_bank, require_face_recognition


def load_bank_from_db() -> List:
    records = []
    persons = Person.query.all()
    for person in persons:
        for emb in person.embeddings:
            records.append((person.id, person.full_name, emb.vector))
    return build_bank(records)


def run_camera(camera_source: str | int, camera_id: int | None, tolerance: float = 0.45):
    try:
        require_face_recognition()
    except RuntimeError as exc:
        raise SystemExit(f"Dependencia faltante: {exc}")

    app = create_app()
    with app.app_context():
        bank = load_bank_from_db()
        print(f"Embeddings cargados: {len(bank)}")

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara/stream")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame, saliendo...")
                break

            rgb = frame[:, :, ::-1]
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), encoding in zip(locations, encodings):
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
                        camera_id=camera_id,
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
