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
class PersonEmbeddings:
    """Agrupa todas las muestras de una persona para decidir con mayor precisión."""

    person_id: int
    person_name: str
    list_tag: Optional[str]
    vectors: np.ndarray
    qualities: List[float]


def build_person_bank(records: Iterable[Tuple[int, str, str, Optional[str], Optional[float]]]) -> List[PersonEmbeddings]:
    """Construye un banco por persona consolidando todas sus fotos/embeddings.

    Cada fila es (person_id, nombre, vector_raw, list_tag, quality).
    """

    grouped: dict[int, list[tuple[np.ndarray, float]]] = {}
    names: dict[int, tuple[str, Optional[str]]] = {}
    for person_id, name, raw_vector, tag, quality in records:
        parsed = parse_embedding(raw_vector)
        if not parsed:
            continue
        vec = _normalize(parsed)
        grouped.setdefault(person_id, []).append((vec, float(quality) if quality is not None else 1.0))
        names[person_id] = (name, tag)

    bank: list[PersonEmbeddings] = []
    for pid, items in grouped.items():
        vectors = np.stack([v for v, _ in items])
        qualities = [q for _, q in items]
        name, tag = names.get(pid, ("Desconocido", None))
        bank.append(PersonEmbeddings(person_id=pid, person_name=name, list_tag=tag, vectors=vectors, qualities=qualities))
    return bank


def build_person_bank_from_persons(persons) -> List[PersonEmbeddings]:  # type: ignore[override]
    """Conveniencia: recibe objetos ``Person`` cargados con relaciones.

    Se combinan embeddings heredados (tabla ``embeddings``) y fotos nuevas
    (tabla ``face_photos``), para mantener compatibilidad con bases existentes.
    """

    records: list[tuple[int, str, str, Optional[str], Optional[float]]] = []
    for person in persons:
        tag = getattr(person, "list_tag", None) or getattr(person, "role", None)
        for photo in getattr(person, "photos", []) or []:
            records.append((person.id, person.full_name, photo.embedding, tag, getattr(photo, "quality", None)))
        for emb in getattr(person, "embeddings", []) or []:
            records.append((person.id, person.full_name, emb.vector, tag, None))
    return build_person_bank(records)


def best_person_match(
    encoding: Sequence[float],
    bank: Sequence[PersonEmbeddings],
    threshold: float = 0.45,
    margin: float = 0.05,
    top_k: int = 3,
):
    """Busca el mejor candidato considerando **todas** las fotos por persona.

    Estrategia:
    - Normaliza el embedding entrante.
    - Para cada persona, calcula la distancia coseno (1 - similitud) contra todas sus muestras.
    - Usa el mínimo de distancia como puntaje principal y el promedio de las ``top_k``
      mejores como respaldo para diagnósticos.
    - Acepta la coincidencia si ``min_dist <= threshold`` y mejora al segundo
      candidato al menos por ``margin``.

    Devuelve (entry, similarity, diagnostics) o (None, None, None).
    """

    if not bank:
        return None, None, None

    sample = _normalize(encoding)
    best: Optional[tuple[PersonEmbeddings, float, float]] = None
    best_dist = float("inf")
    second_dist = float("inf")

    for entry in bank:
        sims = entry.vectors @ sample
        distances = 1.0 - sims
        min_dist = float(distances.min())

        top = np.sort(distances)[: top_k if top_k else len(distances)]
        mean_top = float(top.mean()) if len(top) else min_dist

        if min_dist < best_dist:
            second_dist = best_dist
            best_dist = min_dist
            best = (entry, min_dist, mean_top)
        elif min_dist < second_dist:
            second_dist = min_dist

    if best is None:
        return None, None, None

    entry, min_dist, mean_top = best
    # Decide: distancia baja y diferencia clara contra el resto
    if min_dist > threshold or (second_dist - min_dist) < margin:
        return None, None, None

    similarity = 1.0 - min_dist
    diagnostics = {"min_distance": min_dist, "mean_top": mean_top, "second_best_distance": second_dist}
    return entry, similarity, diagnostics


# Mantener compatibilidad con el código previo: convierte el banco plano en uno agrupado
def build_bank(records: Iterable[Tuple[int, str, str]]) -> List[PersonEmbeddings]:
    return build_person_bank([(pid, name, vec, None, None) for pid, name, vec in records])


def best_match(encoding: Sequence[float], bank: Sequence[PersonEmbeddings], tolerance: float = 0.7, margin: float = 0.1):
    entry, similarity, _ = best_person_match(encoding, bank, threshold=1 - tolerance, margin=margin)
    return entry, similarity
