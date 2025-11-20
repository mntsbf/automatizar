"""Utilidades de reconocimiento facial y manejo de embeddings.

El flujo principal intenta usar ``face_recognition`` (dlib) para obtener
embeddings precisos. Si esa dependencia no está instalada, cae en un modo
"liviano" basado solo en OpenCV + Haar cascades, suficiente para pruebas
rápidas sin compilar dlib (precisión limitada).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

try:  # Carga perezosa para no romper la app si falta la lib pesada
    import face_recognition  # type: ignore
except ImportError:  # pragma: no cover - rama defensiva
    face_recognition = None

_haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _opencv_embedding(gray_face: np.ndarray) -> Optional[List[float]]:
    """Genera un embedding simple con OpenCV (Haar + vector normalizado)."""

    resized = cv2.resize(gray_face, (32, 32))
    vec = resized.astype(np.float32).flatten()
    norm = np.linalg.norm(vec) or 1.0
    vec /= norm
    return vec.tolist()


def require_face_recognition():
    """Devuelve la librería dlib o lanza un error descriptivo si no está instalada."""

    if face_recognition is None:
        raise RuntimeError(
            "No hay backend 'face_recognition' disponible. "
            "Instala la dependencia pesada con 'pip install -r requirements-ml.txt' (CMake + compilador) "
            "o usa el modo liviano automático basado en OpenCV incluido en requirements.txt."
        )
    return face_recognition


def detect_and_encode(image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], List[float]]]:
    """Devuelve bounding boxes (top, right, bottom, left) y embeddings."""

    results: List[Tuple[Tuple[int, int, int, int], List[float]]] = []

    if face_recognition is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        for (top, right, bottom, left), enc in zip(locations, encodings):
            results.append(((top, right, bottom, left), enc.tolist()))
        return results

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        crop = gray[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        emb = _opencv_embedding(crop)
        if emb:
            top, right, bottom, left = y, x + w, y + h, x
            results.append(((top, right, bottom, left), emb))
    return results


def parse_embedding(vector: str) -> Optional[List[float]]:
    """Convierte una cadena almacenada en la base a una lista de floats.

    Acepta JSON ("[0.1, 0.2]") o cadenas separadas por comas/espacios.
    Devuelve ``None`` si no es posible parsear.
    """

    if not vector:
        return None

    try:
        data = json.loads(vector)
        if isinstance(data, Sequence):
            return [float(x) for x in data]
    except json.JSONDecodeError:
        pass

    try:
        parts = [x for x in vector.replace(",", " ").split() if x]
        return [float(x) for x in parts] if parts else None
    except ValueError:
        return None


def encode_image_array(image: np.ndarray) -> Optional[List[float]]:
    """Extrae el embedding de la primera cara encontrada en la imagen.

    Intenta primero con ``face_recognition`` (si está disponible) y, si no,
    recurre al modo liviano con OpenCV + Haar cascades.
    """

    detections = detect_and_encode(image)
    return detections[0][1] if detections else None


def encode_image_file(path: str) -> Optional[List[float]]:
    """Carga una imagen desde disco y devuelve el embedding de la primera cara."""

    if face_recognition is not None:
        image = cv2.cvtColor(face_recognition.load_image_file(path), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)
    return encode_image_array(image)


@dataclass
class KnownEmbedding:
    person_id: int
    person_name: str
    vector: np.ndarray


def build_bank(records: Iterable[Tuple[int, str, str]]) -> List[KnownEmbedding]:
    """Construye un banco de embeddings a partir de filas (id, nombre, vector)."""

    bank: List[KnownEmbedding] = []
    for person_id, name, raw_vector in records:
        parsed = parse_embedding(raw_vector)
        if parsed:
            bank.append(KnownEmbedding(person_id=person_id, person_name=name, vector=np.array(parsed)))
    return bank


def best_match(encoding: Sequence[float], bank: Sequence[KnownEmbedding], tolerance: float = 0.45):
    """Encuentra el match más cercano dentro del banco.

    Devuelve (match, distancia). Si no cumple el ``tolerance`` devuelve (None, None).
    """

    if not bank:
        return None, None

    sample = np.array(encoding)
    distances = np.linalg.norm([sample - item.vector for item in bank], axis=1)
    min_idx = int(np.argmin(distances))
    min_dist = float(distances[min_idx])
    if min_dist <= tolerance:
        return bank[min_idx], min_dist
    return None, None
