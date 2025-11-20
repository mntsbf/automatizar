"""Utilidades de reconocimiento facial y manejo de embeddings.

Este módulo encapsula la lógica de extracción y comparación de embeddings
usando la librería ``face_recognition`` para que pueda ser reutilizada por
la API y el agente local.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Carga perezosa para no romper la app si falta la lib
    import face_recognition  # type: ignore
except ImportError:  # pragma: no cover - rama defensiva
    face_recognition = None


def require_face_recognition():
    """Devuelve la librería o lanza un error descriptivo si no está instalada."""

    if face_recognition is None:
        raise RuntimeError(
            "La dependencia opcional 'face_recognition' no está instalada. "
            "Ejecuta 'pip install -r requirements-ml.txt' tras instalar CMake y un compilador. "
            "En Windows instala Build Tools con C++ + CMake y abre una terminal nueva. "
            "Consulta README (sección de problemas con dlib/CMake) para más detalles."
        )
    return face_recognition


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
    """Extrae el embedding de la primera cara encontrada en la imagen."""

    fr = require_face_recognition()
    rgb = image[:, :, ::-1]
    locations = fr.face_locations(rgb)
    if not locations:
        return None
    encodings = fr.face_encodings(rgb, known_face_locations=locations)
    return encodings[0].tolist() if encodings else None


def encode_image_file(path: str) -> Optional[List[float]]:
    """Carga una imagen desde disco y devuelve el embedding de la primera cara."""

    fr = require_face_recognition()
    image = fr.load_image_file(path)
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
