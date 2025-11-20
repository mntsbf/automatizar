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
    vec = _normalize(vec)
    return vec.tolist()


def _normalize(arr: Sequence[float]) -> np.ndarray:
    """Normaliza vectores para compararlos con coseno."""

    vec = np.array(arr, dtype=np.float32)
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


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
            bank.append(
                KnownEmbedding(
                    person_id=person_id, person_name=name, vector=_normalize(parsed)
                )
            )
    return bank


def best_match(
    encoding: Sequence[float],
    bank: Sequence[KnownEmbedding],
    tolerance: float = 0.7,
    margin: float = 0.1,
):
    """Encuentra el match más cercano usando similitud de coseno normalizada.

    Requiere simultáneamente que la similitud supere ``tolerance`` **y** que sea
    al menos ``margin`` mayor que la segunda mejor coincidencia, reduciendo falsos
    positivos cuando varios rostros son parecidos.

    Devuelve (match, similitud). Si no cumple las condiciones devuelve (None, None).
    """

    if not bank:
        return None, None

    sample = _normalize(encoding)
    bank_matrix = np.stack([item.vector for item in bank])
    similarities = bank_matrix @ sample  # producto punto con vectores normalizados

    max_idx = int(np.argmax(similarities))
    max_sim = float(similarities[max_idx])

    # Chequea ambigüedad: la mejor coincidencia debe destacar sobre la segunda
    second_best = float(np.partition(similarities, -2)[-2]) if len(similarities) > 1 else -1.0
    if max_sim < tolerance or (max_sim - second_best) < margin:
        return None, None

    return bank[max_idx], max_sim
