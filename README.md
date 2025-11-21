# FaceHub demo

Aplicación Flask + SQLite lista para probar localmente un flujo simplificado de vigilancia distribuida con:
- API REST para sitios, cámaras, personas, embeddings y alertas.
- Dashboard moderno (sidebar, topbar y tarjetas) para visualizar alertas, métricas y gráficas en tiempo real.
- Base de datos SQLite autocontenida.
  - La base se crea por defecto en `backend/facehub.db`. Si ya tenías un `facehub.db` en la raíz del proyecto, el servidor lo detectará y lo reutilizará automáticamente para no perder embeddings previos.

## Requisitos
- Python 3.10+
- `pip`
- Hay dos modos de reconocimiento:
  - **Modo completo (dlib/face_recognition)**: mayor precisión, requiere compilar `dlib` (Build Tools + CMake).
  - **Modo liviano (OpenCV Haar)**: ya viene con `requirements.txt`, no necesita compilar nada. Usa cascadas Haar + vectores normalizados (precisión básica para pruebas locales).
- Modo avanzado opcional: **ArcFace/InsightFace** y anti-spoof con ONNX (CPU). Instala `insightface` + `onnxruntime` con `pip install -r requirements-ml.txt`. Si incluyes un modelo tipo SilentFace en `backend/models/silentface.onnx` se usará automáticamente para liveness.
- Para habilitar el modo completo (`face_recognition`/`dlib`), instala herramientas de compilación y CMake:
  - Windows (guía rápida):
    1. Instala [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) eligiendo **Desktop development with C++** (incluye MSVC y el SDK de Windows).
    2. Instala [CMake](https://cmake.org/download/) marcando la opción de agregarlo al **PATH**.
    3. Abre **una terminal nueva** y comprueba `cmake --version` y `cl`.
    4. Actualiza las utilidades de compilación y CMake en tu venv: `pip install --upgrade pip setuptools wheel cmake`.
    5. Instala las dependencias opcionales: `pip install -r requirements-ml.txt`.
  - Linux/macOS: ten disponibles compiladores (`build-essential`/`xcode-select --install`) y `cmake` (`sudo apt install cmake` en Debian/Ubuntu), luego `pip install -r requirements-ml.txt`.

## Instalación y ejecución
### 1) Descarga del proyecto
Si usas Git, clona el repositorio:
```bash
git clone <url-de-tu-repo>/automatizar.git
cd automatizar
```

Si prefieres descargar un ZIP, desde la interfaz web del repositorio pulsa **Code > Download ZIP**, descomprime el archivo y entra en la carpeta `automatizar`.

### 2) Preparar entorno y dependencias
Instala dependencias base (sin reconocimiento facial) para que el servidor y el panel funcionen:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.app
```
La aplicación queda disponible en `http://localhost:5000`. En este punto funcionará el modo liviano (detección con OpenCV). Si quieres embeddings más precisos, instala el modo completo opcional.

> Nota: si no instalas el paquete opcional `face_recognition`, el sistema usa automáticamente el modo liviano de OpenCV (cascadas Haar + vectores normalizados). Funciona para pruebas, pero la precisión es menor.

Para habilitar reconocimiento facial e ingesta de fotos, instala la dependencia opcional (tras cumplir los requisitos del sistema):
```bash
pip install -r requirements-ml.txt
```

Si al instalar ves el error "CMake is not installed on your system" o fallos al compilar **dlib**:

- Verifica que `cmake --version` y `cl` (Windows) funcionan desde la misma terminal donde ejecutas `pip`.
- Reinstala CMake desde el enlace oficial y abre una terminal nueva para que el PATH se actualice.
- Asegúrate de haber instalado los **Build Tools con C++** (Windows) o `build-essential` (Linux).

### 3) Probar reconocimiento en local
1. Registra personas desde el panel y sube una foto en la sección **Embeddings desde foto** para generar el embedding automáticamente (o usa el formulario de **galería** para múltiples fotos por persona).
2. Ejecuta el agente local para abrir la webcam o un RTSP y generar alertas sobre tu base:

   ```bash
   python -m backend.agent --camera 0 --camera-id 1 --threshold 0.38 --margin 0.08 --spoof-threshold 0.5
   ```

   Pulsa **q** para cerrar la ventana o **r** para recargar el banco de embeddings sin reiniciar.
3. Ajusta `--threshold` (distancia coseno máxima aceptada; menor = más estricto), `--margin` (diferencia mínima con el segundo mejor) y `--spoof-threshold` (score mínimo de anti-spoof) para balancear precisión/recall.

### Cómo mejorar la precisión
- **Usa el modo completo (`face_recognition`/dlib)**: instala `pip install -r requirements-ml.txt` tras preparar CMake + compilador para embeddings más sólidos.
- **Ajusta tolerancia**: en modo completo suelen funcionar 0.7-0.8; en modo liviano prueba 0.6-0.7. Sube `--margin` (p.ej. 0.12-0.15) si sigues viendo falsos positivos cuando hay personas parecidas.
- **Normaliza datos**: sube fotos frontales, bien iluminadas y nítidas; evita recortes con gafas oscuras o baja resolución.
- **Recarga el banco**: botón "Actualizar" en el panel o `r` en la ventana del agente para usar nuevos embeddings al instante.
- **Calidad de cámara**: mayor resolución (720p+) y buena iluminación reducen ruido en detección y comparación.

### Embeddings múltiples por persona
- **Modelo de datos**: una persona puede tener múltiples fotos (`face_photos`) con su embedding y metadata (calidad, bounding box). Se almacena en `static/faces/` con nombre único.
- **Carga masiva**: usa el formulario "Subir varias fotos por persona" o el endpoint `POST /api/persons/<id>/photos` (campo `images`). Cada imagen valida que haya rostro, genera embedding y guarda calidad para filtrar ruido. El anti-spoofing está desactivado por defecto en esta ruta para no bloquear galerías históricas; envía `live_check=true` si quieres forzarlo.
- **Decisión de match**: se calcula la distancia coseno del rostro entrante contra **todas** las fotos de cada persona y se toma el mínimo. Se acepta si `distancia <= threshold` y mejora al segundo candidato al menos por `margin`.
- **Refresco del banco**: tras agregar fotos puedes llamar `POST /api/embeddings/refresh` o pulsar **Actualizar**/`r` para recalcular el banco en memoria.
- **ArcFace + normalización**: si instalas `insightface`, el pipeline usa ArcFace (buffalo_l) con alineamiento de ojos, resize a 112x112 y CLAHE para mejorar contraste; métricas con distancia coseno + margen reducen falsos positivos.
- **Anti-spoofing**: si detectas fotos/pantallas, sube `backend/models/silentface.onnx` o usa el score heurístico; define `spoof_threshold` (0.5-0.7 recomendado) en el agente o en `POST /api/recognize`.

### Pipeline robusto: anti-spoof + normalización + dataset incremental
- **Anti-spoofing**: `is_live_face` usa modelo ONNX si está disponible y, si no, heurísticas de textura/saturación. Solo genera embedding si `live` supera el umbral.
- **Preprocesamiento**: `preprocess_face` alinea los ojos, corrige iluminación con CLAHE y normaliza a 112x112 antes de ArcFace/face_recognition.
- **Embeddings modernos**: `generate_embedding` elige ArcFace (InsightFace) si está instalado; si no, recurre a face_recognition o OpenCV ligero.
- **Matching**: `recognize_faces` compara contra todas las fotos de cada persona con distancia coseno (umbral por defecto 0.45, margen 0.08, top_k=5) y descarta empates.
- **Dataset incremental**: si una coincidencia es confiable (`similarity >= incremental_threshold`, p.ej. 0.85) se puede llamar `save_incremental_sample` para guardar automáticamente el recorte y su embedding. Se activa con `--no-incremental` para deshabilitar en el agente o `save_incremental=true` en `/api/recognize`.

### Pipeline perfeccionado (copiar/pegar)
- **Anti-spoofing**: `es_rostro_real` / `is_live_face` usan SilentFace ONNX si está disponible; `spoof_threshold` recomendado 0.5-0.7.
- **Alineación + luz**: `alinear_rostro` rota y centra con landmarks (ojos/nariz/boca) y `normalizar_luz` aplica CLAHE + bilateral para recuperar detalle en cámaras baratas.
- **Filtro de calidad**: `evaluar_calidad` devuelve score 0-1 (blur, contraste, iluminación y ángulo); por defecto se descartan muestras <0.6.
- **Embeddings modernos**: `generar_embedding_perfeccionado` usa ArcFace/InsightFace si está instalado, combina top_k embeddings y devuelve metadata (bbox, live, quality). Umbral sugerido de match con coseno: 0.45–0.55.

## Endpoints principales
- `GET /api/health` — estado del servidor.
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas (nombre, rut, lista) y embeddings en texto.
- `POST /api/persons/<id>/embedding` — subir foto y crear embedding (guarda archivo y vector).
- `POST /api/persons/<id>/photos` — subir varias fotos (campo `images`) y generar embeddings por foto.
- `POST /api/recognize` — enviar imagen y devolver coincidencias (genera alerta si coincide).
- `GET/POST /api/alerts` — recibir y consultar alertas.
- `GET /api/dashboard/estadisticas` — métricas para tarjetas y gráfico (alertas hoy, críticas, cámaras activas, serie de tiempo y top sedes).
- `GET /api/dashboard/ultimas-alertas` — tabla con últimos eventos (sede, cámara, persona y nivel de riesgo).
- `GET /api/dashboard/tiempo-real` — SSE sencillo para refrescar el dashboard cuando llegan nuevas alertas.
- `POST /api/embeddings/refresh` — recalcula el banco en memoria tras agregar fotos/embeddings.

Puedes usar el panel incluido para poblar la base y ver las alertas en vivo.

## Estructura de templates y estáticos
- `backend/frontend/templates/base.html`: layout con sidebar, topbar y bloques Jinja2.
- `backend/frontend/templates/dashboard.html`: tablero principal con tarjetas, gráfico de Chart.js y tabla de alertas.
- `backend/frontend/templates/alerts.html`: módulo de alertas con filtros por riesgo/sede/persona y live toggle.
- `backend/frontend/templates/cameras.html`: alta y listado de cámaras por sede.
- `backend/frontend/templates/persons.html`: gestión de personas, roles/listas y subida de embedding por foto.
- `backend/frontend/templates/scanner.html`: escaneo guiado con movimientos y liveness para capturar varias fotos.
- `backend/frontend/templates/config.html`: ajustes de tolerancia, margen, tema y refresco en vivo.
- `backend/frontend/static/dashboard.js`: lógica de Fetch + SSE para actualizar tarjetas, gráfica y tabla.
- `backend/frontend/static/alerts.js`: filtros y tabla dinámica de alertas.
- `backend/frontend/static/cameras.js`: creación de cámaras y contadores de actividad 24h.
- `backend/frontend/static/persons.js`: alta de personas y embeddings.
- `backend/frontend/static/lookup.js`: formulario para buscar una persona por foto y mostrar coincidencias.
- `backend/frontend/static/config.js`: sincroniza y guarda parámetros globales.
- `backend/frontend/static/styles.css`: estilos de dashboard (tiles, sidebar, badges de riesgo).

### API rápida
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas y embeddings en texto (nombre, rut, lista).
- `POST /api/persons/<id>/embedding` — subir foto y crear embedding con `face_recognition` o modo liviano (se guarda la foto).
- `POST /api/persons/<id>/photos` — subir varias fotos (campo `images`) y generar embeddings por foto.
- `POST /api/scan/frame` — capturar un frame guiado (mirar al frente, girar, sonreír) con anti-spoofing y validación de movimiento.
- `POST /api/scan/finalize` — combinar embeddings capturados y (opcional) guardarlos en la persona.
- `POST /api/recognize` — enviar imagen y devolver coincidencias (genera alerta si coincide).
- `POST /buscar-persona` — subir foto puntual y devolver el mejor match (JSON simple con confianza/distancia).
- `GET/POST /api/alerts` — recibir y consultar alertas (filtros por persona, sitio y riesgo con query params).
- `GET /api/dashboard/estadisticas` — métricas para tarjetas y gráfico (alertas hoy, críticas, cámaras activas, serie de tiempo y top sedes).
- `GET /api/dashboard/ultimas-alertas` — tabla con últimos eventos (sede, cámara, persona y nivel de riesgo).
- `GET /api/dashboard/tiempo-real` — SSE sencillo para refrescar el dashboard cuando llegan nuevas alertas.
- `POST /api/embeddings/refresh` — recalcula el banco en memoria tras agregar fotos/embeddings.
- `GET/PUT /api/settings` — tolerancia, margen, tema y live refresh para el dashboard.

#### Búsqueda puntual por foto (`POST /buscar-persona`)

- `multipart/form-data` con campo `image` o `photo` (JPG/PNG).
- Parámetros opcionales: `threshold` (distancia coseno), `margin` (separación vs. segundo mejor), `live_check=true|false` (anti-spoofing) y `spoof_threshold`.
- Respuesta de match:

```json
{
  "resultado": "match",
  "persona": {"id": 12, "nombre": "Juan Pérez", "confianza": 0.92, "distancia": 0.08},
  "diagnosticos": {"min_distance": 0.08, "second_best_distance": 0.21},
  "bank_version": 1700000000
}
```

- Si no hay coincidencias: `{ "resultado": "desconocido", "mensaje": "No se encontró una coincidencia bajo el umbral" }`
- Si no se detectan rostros: `{ "resultado": "sin_rostro" }`

### Escáner vivo con movimientos

- Vista `/scan`: abre la cámara, guía al usuario (frente, derecha, izquierda, arriba, abajo, sonreír) y captura frames de buena calidad.
  Ahora soporta modo automático en vivo: evalúa el movimiento en tiempo real y avanza al siguiente paso sin que el operador presione
  "Capturar" en cada giro.
- `POST /api/scan/frame` espera `image` + `action` (frente|derecha|izquierda|arriba|abajo|sonreir), aplica anti-spoofing, verifica el movimiento con pose/landmarks y devuelve `embedding`, `pose`, `quality` y `bbox` si es válido.
- `POST /api/scan/finalize` combina los embeddings (`mode=median` por defecto) y puede persistir el vector en la persona (`person_id`).
- Usa los mismos umbrales de tolerancia/calidad que el resto del pipeline y permite guardar cada frame en la galería (`save_photo=true`).
