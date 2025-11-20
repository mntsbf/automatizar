# FaceHub demo

Aplicación Flask + SQLite lista para probar localmente un flujo simplificado de vigilancia distribuida con:
- API REST para sitios, cámaras, personas, embeddings y alertas.
- Panel web minimalista para ingresar datos y visualizar alertas recientes.
- Base de datos SQLite autocontenida.

## Requisitos
- Python 3.10+
- `pip`

## Instalación y ejecución
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.app
```
La aplicación queda disponible en `http://localhost:5000`.

## Endpoints principales
- `GET /api/health` — estado del servidor.
- `GET/POST /api/sites` — registrar y listar ubicaciones.
- `GET/POST /api/cameras` — agregar cámaras asociadas a un sitio.
- `GET/POST /api/persons` — registrar personas y embeddings en texto.
- `GET/POST /api/alerts` — recibir y consultar alertas.

Puedes usar el panel incluido para poblar la base y ver las alertas en vivo.
