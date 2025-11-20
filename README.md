# FaceHub demo

Aplicación Flask + SQLite lista para probar localmente un flujo simplificado de vigilancia distribuida con:
- API REST para sitios, cámaras, personas, embeddings y alertas.
- Panel web minimalista para ingresar datos, cargar rostros y visualizar alertas recientes.
- Base de datos SQLite autocontenida.

## Requisitos
- Python 3.10+
- `pip`
- Para habilitar reconocimiento facial (`face_recognition`/`dlib`), instala herramientas de compilación y CMake:
  - Windows: [Build Tools para Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) + [CMake](https://cmake.org/download/). Tras instalarlos abre una terminal nueva y ejecuta `pip install --upgrade pip setuptools wheel`.
  - Linux/macOS: ten disponibles compiladores (`build-essential`/`xcode-select --install`) y `cmake` (`sudo apt install cmake` en Debian/Ubuntu).

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
La aplicación queda disponible en `http://localhost:5000`.

> Nota: si no instalas el paquete opcional `face_recognition`, las rutas que procesan imágenes y el agente local mostrarán un mensaje de dependencia faltante, pero el servidor puede iniciar y el resto de la API funciona.

Para habilitar reconocimiento facial e ingesta de fotos, instala la dependencia opcional (tras cumplir los requisitos del sistema):
```bash
pip install -r requirements-ml.txt
```

### 3) Probar reconocimiento en local
1. Registra personas desde el panel y sube una foto en la sección **Embeddings desde foto** para generar el embedding automáticamente.
2. Ejecuta el agente local para abrir la webcam o un RTSP y generar alertas sobre tu base:

   ```bash
   python -m backend.agent --camera 0 --camera-id 1 --tolerance 0.45
   ```

   Pulsa **q** para cerrar la ventana o **r** para recargar el banco de embeddings sin reiniciar.

## Endpoints principales
- `GET /api/health` — estado del servidor.
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas y embeddings en texto.
- `POST /api/persons/<id>/embedding` — subir foto y crear embedding con `face_recognition`.
- `POST /api/recognize` — enviar imagen y devolver coincidencias (genera alerta si coincide).
- `GET/POST /api/alerts` — recibir y consultar alertas.

Puedes usar el panel incluido para poblar la base y ver las alertas en vivo.
