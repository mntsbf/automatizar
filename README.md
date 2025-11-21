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
1. Registra personas desde el panel y sube una foto en la sección **Embeddings desde foto** para generar el embedding automáticamente.
2. Ejecuta el agente local para abrir la webcam o un RTSP y generar alertas sobre tu base:

   ```bash
   python -m backend.agent --camera 0 --camera-id 1 --tolerance 0.7 --margin 0.1
   ```

   Pulsa **q** para cerrar la ventana o **r** para recargar el banco de embeddings sin reiniciar.
3. Ajusta `--tolerance` (similitud mínima de coseno ya normalizada): valores altos → más precisión y menos falsos positivos; valores bajos → más recall. Usa también `--margin` (diferencia mínima sobre el segundo mejor candidato) para filtrar casos ambiguos.

### Cómo mejorar la precisión
- **Usa el modo completo (`face_recognition`/dlib)**: instala `pip install -r requirements-ml.txt` tras preparar CMake + compilador para embeddings más sólidos.
- **Ajusta tolerancia**: en modo completo suelen funcionar 0.7-0.8; en modo liviano prueba 0.6-0.7. Sube `--margin` (p.ej. 0.12-0.15) si sigues viendo falsos positivos cuando hay personas parecidas.
- **Normaliza datos**: sube fotos frontales, bien iluminadas y nítidas; evita recortes con gafas oscuras o baja resolución.
- **Recarga el banco**: botón "Actualizar" en el panel o `r` en la ventana del agente para usar nuevos embeddings al instante.
- **Calidad de cámara**: mayor resolución (720p+) y buena iluminación reducen ruido en detección y comparación.

## Endpoints principales
- `GET /api/health` — estado del servidor.
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas y embeddings en texto.
- `POST /api/persons/<id>/embedding` — subir foto y crear embedding con `face_recognition`.
- `POST /api/recognize` — enviar imagen y devolver coincidencias (genera alerta si coincide).
- `GET/POST /api/alerts` — recibir y consultar alertas.
- `GET /api/dashboard/estadisticas` — métricas para tarjetas y gráfico (alertas hoy, críticas, cámaras activas, serie de tiempo y top sedes).
- `GET /api/dashboard/ultimas-alertas` — tabla con últimos eventos (sede, cámara, persona y nivel de riesgo).
- `GET /api/dashboard/tiempo-real` — SSE sencillo para refrescar el dashboard cuando llegan nuevas alertas.

Puedes usar el panel incluido para poblar la base y ver las alertas en vivo.

## Estructura de templates y estáticos
- `backend/frontend/templates/base.html`: layout con sidebar, topbar y bloques Jinja2.
- `backend/frontend/templates/dashboard.html`: tablero principal con tarjetas, gráfico de Chart.js y tabla de alertas.
- `backend/frontend/templates/alerts.html`: módulo de alertas con filtros por riesgo/sede/persona y live toggle.
- `backend/frontend/templates/cameras.html`: alta y listado de cámaras por sede.
- `backend/frontend/templates/persons.html`: gestión de personas, roles/listas y subida de embedding por foto.
- `backend/frontend/templates/config.html`: ajustes de tolerancia, margen, tema y refresco en vivo.
- `backend/frontend/static/dashboard.js`: lógica de Fetch + SSE para actualizar tarjetas, gráfica y tabla.
- `backend/frontend/static/alerts.js`: filtros y tabla dinámica de alertas.
- `backend/frontend/static/cameras.js`: creación de cámaras y contadores de actividad 24h.
- `backend/frontend/static/persons.js`: alta de personas y embeddings.
- `backend/frontend/static/config.js`: sincroniza y guarda parámetros globales.
- `backend/frontend/static/styles.css`: estilos de dashboard (tiles, sidebar, badges de riesgo).

### API rápida
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas y embeddings en texto.
- `POST /api/persons/<id>/embedding` — subir foto y crear embedding con `face_recognition` o modo liviano.
- `POST /api/recognize` — enviar imagen y devolver coincidencias (genera alerta si coincide).
- `GET/POST /api/alerts` — recibir y consultar alertas (filtros por persona, sitio y riesgo con query params).
- `GET /api/dashboard/estadisticas` — métricas para tarjetas y gráfico (alertas hoy, críticas, cámaras activas, serie de tiempo y top sedes).
- `GET /api/dashboard/ultimas-alertas` — tabla con últimos eventos (sede, cámara, persona y nivel de riesgo).
- `GET /api/dashboard/tiempo-real` — SSE sencillo para refrescar el dashboard cuando llegan nuevas alertas.
- `GET/PUT /api/settings` — tolerancia, margen, tema y live refresh para el dashboard.
