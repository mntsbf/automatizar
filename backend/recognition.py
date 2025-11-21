"""Utilidades de reconocimiento facial y manejo de embeddings.

El flujo principal intenta usar backends modernos para embeddings (ArcFace/
InsightFace) cuando están disponibles y mantiene compatibilidad con
``face_recognition`` o el modo liviano (OpenCV). También incluye:

- Preprocesamiento/normalización (alineado de ojos + mejora de iluminación).
- Anti-spoofing liviano en CPU con modelo ONNX opcional y heurísticas.
- Banco incremental de embeddings por persona con comparación coseno.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

try:  # Carga perezosa para no romper la app si falta la lib pesada
    import face_recognition  # type: ignore
except ImportError:  # pragma: no cover - rama defensiva
    face_recognition = None

# InsightFace / ArcFace es opcional. Si no está instalado, el pipeline seguirá
# funcionando con face_recognition u OpenCV.
try:  # pragma: no cover - opcional
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - opcional
    FaceAnalysis = None

_haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_anti_spoof_net = None  # Cargado bajo demanda
_arcface_app = None  # Cache de modelo InsightFace

# Parámetros por defecto para el pipeline perfeccionado
DEFAULT_FACE_SIZE = (112, 112)  # ArcFace/InsightFace suelen usar 112x112
DEFAULT_SPOOF_THRESHOLD = 0.5
DEFAULT_QUALITY_THRESHOLD = 0.6
DEFAULT_MATCH_THRESHOLD = 0.45  # menor = más estricto (distancia coseno)


def has_heavy_embedding_backend() -> bool:
    """Indica si hay un backend de embeddings robusto disponible.

    Se considera "robusto" cuando existe ``face_recognition`` (dlib) o el
    módulo de InsightFace/ArcFace. Esto nos permite ajustar umbrales de
    coincidencia: en modo liviano (sólo OpenCV), los embeddings son menos
    discriminativos y conviene usar tolerancias más permisivas.
    """

    return face_recognition is not None or FaceAnalysis is not None


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


def _load_anti_spoof_model(model_path: Optional[str] = None):
    """Carga un modelo ONNX ligero de anti-spoofing si existe.

    Por defecto busca ``models/silentface.onnx`` relativo al proyecto. Si no
    se encuentra, devuelve ``None`` para usar heurísticas suaves.
    """

    global _anti_spoof_net
    if _anti_spoof_net is not None:
        return _anti_spoof_net

    if model_path:
        candidate = Path(model_path)
    else:
        candidate = Path(__file__).resolve().parent / "models" / "silentface.onnx"

    if candidate.exists():
        try:
            _anti_spoof_net = cv2.dnn.readNetFromONNX(str(candidate))
        except Exception:
            _anti_spoof_net = None
    return _anti_spoof_net


def is_live_face(face_bgr: np.ndarray, threshold: float = 0.5) -> tuple[bool, dict]:
    """Evalúa si el rostro parece real (anti-spoofing).

    1) Si existe un modelo ONNX (SilentFace) se usa su probabilidad de "live".
    2) Si no, aplica heurísticas de textura y saturación para descartar fotos
       impresas o pantallas con poca variación.

    Devuelve ``(es_real, metricas)`` donde ``metricas`` incluye ``score``.
    """

    net = _load_anti_spoof_model()
    metrics: dict[str, float] = {}

    if net is not None:
        resized = cv2.resize(face_bgr, (128, 128))
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (128, 128))
        net.setInput(blob)
        preds = net.forward()
        live_score = float(preds[0][1])  # índice 1 = prob. vivo en SilentFace
        metrics["score"] = live_score
        return live_score >= threshold, metrics

    # Heurística: variación de textura + saturación
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    sat_mean = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    # Normalizar heurística a [0,1]
    live_score = min(1.0, (variance / 100.0) * 0.6 + (sat_mean / 128.0) * 0.4)
    metrics["score"] = float(live_score)
    return live_score >= threshold, metrics


def es_rostro_real(face_bgr: np.ndarray, threshold: float = DEFAULT_SPOOF_THRESHOLD) -> bool:
    """Alias simple orientado al usuario para el chequeo de anti-spoofing."""

    ok, _metrics = is_live_face(face_bgr, threshold=threshold)
    return ok


def normalizar_luz(face_bgr: np.ndarray) -> np.ndarray:
    """Mejora iluminación con CLAHE y corrección de contraste local."""

    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    # Suavizar sombras duras manteniendo detalle
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return enhanced


def alinear_rostro(
    img_bgr: np.ndarray,
    landmarks: Optional[dict] = None,
    output_size: tuple[int, int] = DEFAULT_FACE_SIZE,
    padding: float = 0.25,
) -> np.ndarray:
    """Alinea y recorta un rostro usando landmarks (ojos/boca) si existen."""

    rotated = align_face(img_bgr, landmarks)
    h, w = rotated.shape[:2]

    if landmarks:
        points = []
        for key in ("left_eye", "right_eye", "nose_tip", "top_lip", "bottom_lip"):
            pts = landmarks.get(key)
            if pts is None:
                continue
            points.extend(pts)
        if points:
            pts_arr = np.array(points)
            x1, y1 = pts_arr.min(axis=0)
            x2, y2 = pts_arr.max(axis=0)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)
            size = max(bw, bh)
            size *= 1 + padding * 2
            x1 = int(max(0, cx - size / 2))
            y1 = int(max(0, cy - size / 2))
            x2 = int(min(w, cx + size / 2))
            y2 = int(min(h, cy + size / 2))
            cropped = rotated[y1:y2, x1:x2]
        else:
            cropped = rotated
    else:
        cropped = rotated

    if cropped.size == 0:
        cropped = rotated

    return cv2.resize(cropped, output_size)


def _get_arcface_app():
    """Carga (lazy) el modelo InsightFace para embeddings ArcFace."""

    global _arcface_app
    if _arcface_app is not None:
        return _arcface_app

    if FaceAnalysis is None:  # pragma: no cover - dependencia opcional
        return None

    try:
        app = FaceAnalysis(name="buffalo_l", providers=("CPUExecutionProvider",))
        app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
        _arcface_app = app
    except Exception:
        _arcface_app = None
    return _arcface_app


def align_face(face_bgr: np.ndarray, landmarks: Optional[dict] = None) -> np.ndarray:
    """Alinea la imagen de rostro usando los ojos si están disponibles."""

    if not landmarks:
        return face_bgr
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not left_eye or not right_eye:
        return face_bgr

    left = np.mean(left_eye, axis=0)
    right = np.mean(right_eye, axis=0)
    dy, dx = right[1] - left[1], right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(np.mean([left, right], axis=0).astype(int))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face_bgr, rot_mat, (face_bgr.shape[1], face_bgr.shape[0]))
    return aligned


def preprocess_face(face_bgr: np.ndarray, landmarks: Optional[dict] = None, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Normaliza un rostro: alinea, mejora iluminación y redimensiona.

    - Alinea los ojos para que la orientación sea consistente.
    - Aplica CLAHE para mitigar sombras/iluminación desigual.
    - Redimensiona al tamaño esperado por ArcFace (112x112) o similar.
    """

    aligned = align_face(face_bgr, landmarks)
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    resized = cv2.resize(bgr_eq, size)
    return resized


def evaluar_calidad(
    face_bgr: np.ndarray,
    landmarks: Optional[dict] = None,
    min_size: int = 80,
    blur_threshold: float = 90.0,
    brightness_range: tuple[int, int] = (60, 190),
    angle_limit: float = 25.0,
) -> float:
    """Evalúa la calidad del recorte de rostro y retorna score 0..1."""

    h, w = face_bgr.shape[:2]
    if min(h, w) < min_size:
        return 0.0

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, blur_var / blur_threshold)

    mean_light = gray.mean()
    low, high = brightness_range
    mid = (low + high) / 2
    light_score = 1.0 - min(1.0, abs(mean_light - mid) / mid)

    contrast_score = min(1.0, gray.std() / 64.0)

    angle_penalty = 0.0
    if landmarks and landmarks.get("left_eye") and landmarks.get("right_eye"):
        left = np.mean(landmarks["left_eye"], axis=0)
        right = np.mean(landmarks["right_eye"], axis=0)
        dy, dx = right[1] - left[1], right[0] - left[0]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > angle_limit:
            angle_penalty = min(0.5, (angle - angle_limit) / 90.0)

    score = (
        0.35 * blur_score
        + 0.25 * contrast_score
        + 0.2 * light_score
        + 0.2 * min(1.0, min(h, w) / (min_size * 2))
    )
    score = max(0.0, min(1.0, score - angle_penalty))
    return float(score)


def generate_embedding(face_bgr: np.ndarray, landmarks: Optional[dict] = None) -> Optional[List[float]]:
    """Genera un embedding con el mejor backend disponible.

    Prioridad: ArcFace/InsightFace -> face_recognition -> OpenCV ligero.
    """

    # 1) ArcFace / InsightFace
    app = _get_arcface_app()
    if app is not None:  # pragma: no cover - opcional
        try:
            # La app espera RGB uint8, normalización interna
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            # Usa inferencia directa del módulo de reconocimiento
            # (evita detección porque ya tenemos el rostro recortado)
            feat = app.models["recognition"].get_feature(rgb)
            if feat is not None:
                return _normalize(feat).tolist()
        except Exception:
            pass

    # 2) face_recognition clásico
    if face_recognition is not None:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Si hay landmarks los pasamos como dlib points, de lo contrario None
        encs = face_recognition.face_encodings(rgb)
        if encs:
            return _normalize(encs[0]).tolist()

    # 3) Fallback ligero OpenCV
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return _opencv_embedding(gray)


def generar_embedding_perfeccionado(
    frame_bgr: np.ndarray,
    live_check: bool = True,
    spoof_threshold: float = DEFAULT_SPOOF_THRESHOLD,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    preprocess_size: tuple[int, int] = DEFAULT_FACE_SIZE,
    top_k: int = 1,
) -> tuple[Optional[List[float]], dict]:
    """Pipeline de extremo a extremo con anti-spoofing, alineado y filtros."""

    detections = robust_detect_and_encode(
        frame_bgr,
        live_check=live_check,
        spoof_threshold=spoof_threshold,
        preprocess_size=preprocess_size,
        quality_threshold=quality_threshold,
    )

    if not detections:
        return None, {"reason": "no_face_or_low_quality"}

    ordered = sorted(detections, key=lambda d: d.get("quality", 0.0), reverse=True)
    embeddings = [d["embedding"] for d in ordered[:top_k]]
    merged = combinar_embeddings(embeddings)
    info = {
        "live": ordered[0].get("live", True),
        "quality": ordered[0].get("quality", None),
        "bbox": ordered[0].get("bbox"),
        "count": len(ordered),
    }
    if not merged:
        info["reason"] = "embedding_failed"
    return merged, info


def require_face_recognition():
    """Devuelve la librería dlib o lanza un error descriptivo si no está instalada."""

    if face_recognition is None:
        raise RuntimeError(
            "No hay backend 'face_recognition' disponible. "
            "Instala la dependencia pesada con 'pip install -r requirements-ml.txt' (CMake + compilador) "
            "o usa el modo liviano automático basado en OpenCV incluido en requirements.txt."
        )
    return face_recognition


def _collect_face_landmarks(rgb_image, locations):
    """Devuelve landmarks de ojos para cada bbox (cuando dlib está disponible)."""

    if face_recognition is None:
        return [None for _ in locations]
    try:
        landmarks = face_recognition.face_landmarks(rgb_image, locations)
    except Exception:
        return [None for _ in locations]
    parsed = []
    for lm in landmarks:
        parsed.append(
            {
                "left_eye": lm.get("left_eye"),
                "right_eye": lm.get("right_eye"),
                "nose_tip": lm.get("nose_tip"),
                "top_lip": lm.get("top_lip"),
                "bottom_lip": lm.get("bottom_lip"),
            }
        )
    return parsed


def _shift_landmarks(landmarks: dict, offset: tuple[int, int]) -> dict:
    """Ajusta landmarks absolutos a coordenadas relativas del recorte."""

    ox, oy = offset
    shifted: dict[str, list[tuple[float, float]]] = {}
    for key, pts in landmarks.items():
        shifted[key] = [(float(x - ox), float(y - oy)) for x, y in pts]
    return shifted


def robust_detect_and_encode(
    image: np.ndarray,
    live_check: bool = False,
    spoof_threshold: float = 0.5,
    preprocess_size: tuple[int, int] = (112, 112),
    quality_threshold: float = 0.0,
) -> List[dict]:
    """Detecta caras, aplica anti-spoofing y devuelve embeddings enriquecidos.

    Cada elemento contiene ``bbox``, ``embedding``, ``live`` y ``metrics``.
    """

    results: List[dict] = []

    if face_recognition is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        landmarks = _collect_face_landmarks(rgb, locations)
        for (top, right, bottom, left), lm in zip(locations, landmarks):
            face_crop = image[top:bottom, left:right]
            rel_lm = _shift_landmarks(lm, (left, top)) if lm else None
            aligned = alinear_rostro(face_crop, rel_lm, output_size=preprocess_size)
            pre = normalizar_luz(aligned)
            live_ok = True
            metrics: Dict[str, float] = {}
            if live_check:
                live_ok, metrics = is_live_face(pre, threshold=spoof_threshold)
            if not live_ok:
                continue
            quality = evaluar_calidad(pre, rel_lm)
            if quality_threshold and quality < quality_threshold:
                continue
            emb = generate_embedding(pre, lm)
            if emb:
                results.append({
                    "bbox": (top, right, bottom, left),
                    "embedding": emb,
                    "live": live_ok,
                    "metrics": metrics,
                    "quality": quality,
                })
        return results

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        aligned = alinear_rostro(crop, None, output_size=preprocess_size)
        pre = normalizar_luz(aligned)
        live_ok = True
        metrics: Dict[str, float] = {}
        if live_check:
            live_ok, metrics = is_live_face(pre, threshold=spoof_threshold)
        if not live_ok:
            continue
        quality = evaluar_calidad(pre, None)
        if quality_threshold and quality < quality_threshold:
            continue
        emb = generate_embedding(pre, None)
        if emb:
            top, right, bottom, left = y, x + w, y + h, x
            results.append({
                "bbox": (top, right, bottom, left),
                "embedding": emb,
                "live": live_ok,
                "metrics": metrics,
                "quality": quality,
            })
    return results


def detect_and_encode(image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], List[float]]]:
    """Compatibilidad: devuelve sólo bbox + embedding."""

    enriched = robust_detect_and_encode(image, live_check=False)
    return [(item["bbox"], item["embedding"]) for item in enriched]


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


def combinar_embeddings(
    embeddings: Sequence[Sequence[float]],
    mode: str = "mean",
    top_k: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
) -> Optional[List[float]]:
    """Combina múltiples embeddings en uno solo normalizado."""

    if not embeddings:
        return None

    arr = np.stack([_normalize(v) for v in embeddings])
    if weights is not None and len(weights) == len(arr):
        w = np.array(weights, dtype=np.float32)
        w = w / (w.sum() or 1.0)
    else:
        w = None

    if top_k and top_k > 0 and len(arr) > top_k:
        if w is not None:
            order = np.argsort(w)[::-1][:top_k]
        else:
            # Seleccionar las muestras más cercanas al centroide preliminar
            centroid = arr.mean(axis=0)
            dists = np.linalg.norm(arr - centroid, axis=1)
            order = np.argsort(dists)[:top_k]
        arr = arr[order]
        if w is not None:
            w = w[order]

    if mode == "median":
        merged = np.median(arr, axis=0)
    elif mode == "top":
        merged = arr[0]
    else:  # mean o weighted
        if w is not None:
            merged = (arr * w[:, None]).sum(axis=0)
        else:
            merged = arr.mean(axis=0)

    return _normalize(merged).tolist()


def recognize_faces(
    frame_bgr: np.ndarray,
    bank: Sequence["PersonEmbeddings"],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = 0.08,
    top_k: int = 5,
    live_check: bool = True,
    spoof_threshold: float = 0.5,
):
    """Pipeline completo: detección + anti-spoof + embeddings + matching.

    Devuelve una lista de dicts con ``bbox``, ``match`` (PersonEmbeddings o None)
    y métricas auxiliares.
    """

    detections = robust_detect_and_encode(
        frame_bgr,
        live_check=live_check,
        spoof_threshold=spoof_threshold,
    )

    results = []
    for det in detections:
        entry, similarity, diagnostics = best_person_match(
            det["embedding"],
            bank,
            threshold=threshold,
            margin=margin,
            top_k=top_k,
        )
        results.append(
            {
                "bbox": det["bbox"],
                "match": entry,
                "similarity": similarity,
                "diagnostics": diagnostics,
                "live": det.get("live", True),
                "metrics": det.get("metrics", {}),
                "embedding": det.get("embedding"),
            }
        )
    return results


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
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = 0.08,
    top_k: int = 5,
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


def best_similarity_no_threshold(
    encoding: Sequence[float], bank: Sequence[PersonEmbeddings]
) -> Optional[dict]:
    """Devuelve el mejor candidato sin aplicar umbrales ni margen.

    Se usa para diagnósticos o para relajar la decisión en modo liviano
    (cuando los embeddings OpenCV son menos discriminativos).  Retorna un
    diccionario con ``entry``, ``similarity``, ``distance`` y ``second_best``.
    """

    if not bank:
        return None

    sample = _normalize(encoding)
    best: Optional[tuple[PersonEmbeddings, float]] = None
    second: float = float("inf")

    for entry in bank:
        sims = entry.vectors @ sample
        if sims.size == 0:
            continue
        sim = float(sims.max())
        dist = 1.0 - sim
        if best is None or dist < best[1]:
            second = best[1] if best else second
            best = (entry, dist)
        elif dist < second:
            second = dist

    if best is None:
        return None

    entry, dist = best
    return {
        "entry": entry,
        "similarity": 1.0 - dist,
        "distance": dist,
        "second_best_distance": second,
    }


# Mantener compatibilidad con el código previo: convierte el banco plano en uno agrupado
def build_bank(records: Iterable[Tuple[int, str, str]]) -> List[PersonEmbeddings]:
    return build_person_bank([(pid, name, vec, None, None) for pid, name, vec in records])


def best_match(encoding: Sequence[float], bank: Sequence[PersonEmbeddings], tolerance: float = 0.7, margin: float = 0.1):
    entry, similarity, _ = best_person_match(encoding, bank, threshold=1 - tolerance, margin=margin)
    return entry, similarity


def save_incremental_sample(
    person_id: int,
    frame_bgr: np.ndarray,
    embedding: Sequence[float],
    app_static_dir: str,
    quality: float = 1.0,
    note: str | None = None,
):
    """Guarda una nueva muestra (foto + embedding) para refinar el banco.

    Se debe invocar dentro de un contexto de app para que SQLAlchemy funcione.
    """

    from .models import FacePhoto, Person  # import local para evitar ciclos
    from .database import db

    person = Person.query.get(person_id)
    if not person:
        return None

    faces_dir = Path(app_static_dir) / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    filename = f"inc_{person_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    file_path = faces_dir / filename
    cv2.imwrite(str(file_path), frame_bgr)
    rel_path = os.path.relpath(file_path, app_static_dir)

    metadata = {"source": "incremental", "note": note or "auto", "ts": time.time()}
    photo = FacePhoto(
        person_id=person_id,
        file_path=rel_path,
        embedding=json.dumps(list(embedding)),
        quality=quality,
        metadata_json=json.dumps(metadata),
    )
    db.session.add(photo)
    db.session.commit()
    return photo
