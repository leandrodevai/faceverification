---
title: Face Verification
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: "Face verification demo with FaceNet and VectorDB"
tags: ["face-verification", "computer-vision", "facenet", "vector-database"]
pinned: false
license: mit
---
![CI](https://github.com/leandrodevai/faceverification/actions/workflows/ci.yml/badge.svg)
![Container Security](https://github.com/leandrodevai/faceverification/actions/workflows/container-build-scan-publish.yml/badge.svg)
![Deploy](https://github.com/leandrodevai/faceverification/actions/workflows/deploy-to-HSspaces.yml/badge.svg)

# Face Verification

AI Engineering Demo project demonstrating an end-to-end face verification workflow with FaceNet embeddings, ChromaDB vector search, and an interactive Gradio interface.

Try it on Hugging Face Spaces: https://huggingface.co/spaces/leandrodevai/faceverification

The app lets users add known people to a local embeddings database and verify whether a new face image matches one of the stored identities.

## Features

- Face detection and preprocessing from uploaded images
- FaceNet embedding extraction with PyTorch
- Similarity search with ChromaDB
- Interactive Gradio UI with add-person and verify-identity flows
- FastAPI interface for containerized API deployments
- Docker-based Hugging Face Space deployment

## Tech Stack

- Python
- PyTorch
- FaceNet / facenet-pytorch
- ChromaDB
- FastAPI
- Gradio
- Docker / GHCR
- Hugging Face Spaces Docker SDK

## Run Locally

Using uv:

```bash
uv sync
uv run python app.py
```

Using pip:

```bash
pip install -r requirements.txt
python app.py
```

## FastAPI Interface

The project includes an HTTP API for the same enroll-and-verify workflow.
Run it locally with:

```bash
uv run uvicorn faceverification.interfaces.fastapi_app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Authentication

Protected endpoints require a bearer token. The default demo credentials are
`demo` / `demo123`; override them with `FACEVERIFICATION_DEMO_USERNAME` and
`FACEVERIFICATION_DEMO_PASSWORD` in `.env`.

```bash
curl -X POST http://localhost:8000/auth/login \
  -F "username=demo" \
  -F "password=demo123"
```

Use the returned token in the `Authorization` header:

```bash
Authorization: Bearer <access_token>
```

### Endpoints

- `GET /health`: returns API status.
- `POST /auth/login`: returns a JWT access token for the demo user.
- `POST /persons`: enrolls a known person from an uploaded image and form `name`.
- `POST /verify`: verifies whether an uploaded face matches a known person.

## Deployment Notes

For API deployments, the recommended baseline is the FastAPI container running
Uvicorn:

```bash
uvicorn faceverification.interfaces.fastapi_app:app --host 0.0.0.0 --port 8000
```

This keeps the demo lightweight and avoids loading the FaceNet/MTCNN models in
multiple worker processes unnecessarily. Because each worker can hold its own
model instance in memory, increasing worker count should be done only after
checking available RAM and expected traffic.

Gunicorn with Uvicorn workers and an Nginx reverse proxy are valid production
options, but they are intentionally not required for the baseline deployment:

- Use Gunicorn/Uvicorn workers when the service needs a traditional process
  manager or multiple worker processes.
- Use Nginx when deploying on a self-managed VM that needs TLS termination,
  upload-size limits, reverse proxy routing, compression, or centralized access
  logs.
- On managed platforms such as Render, Railway, Fly.io, Cloud Run, or similar,
  the platform usually provides the external reverse proxy and TLS layer, so
  running Uvicorn directly inside the application container is sufficient.

### Docker

Build and run the FastAPI service:

```bash
docker compose up --build api
```

The API will be available at:

- http://localhost:8000/health
- http://localhost:8000/docs

Run the optional Gradio interface:

```bash
docker compose --profile gradio up --build
```

The Gradio UI will be available at http://localhost:7860.

Both services are built from the same `Dockerfile` with different variants:

- `fastapi`: published to GHCR by the container workflow.
- `gradio`: used by the Hugging Face Docker Space.

The published FastAPI image is tagged as:

```text
ghcr.io/leandrodevai/faceverification:fastapi
```

Each successful container workflow also publishes an immutable SHA tag with the
`fastapi-sha-*` prefix.

By default, ChromaDB runs in memory, so
enrolled faces are ephemeral and disappear when the container restarts. This is
intentional for the demo baseline.

To persist embeddings to disk, provide a database name:

```bash
FACEVERIFICATION_VECTOR_DB_NAME=local-demo \
  docker compose -f docker-compose.yml -f docker-compose.persist.yml up --build api
```

The base `docker-compose.yml` does not mount any volume, so the default
deployment stays ephemeral. The optional `docker-compose.persist.yml` override
mounts the `faceverification-data` volume at `/data` and translates
`FACEVERIFICATION_VECTOR_DB_NAME` into
`FACEVERIFICATION_VECTOR_DB_PERSIST_DIRECTORY=/data/chroma/<name>` before the
application starts. The application itself still defaults to in-memory ChromaDB
unless `FACEVERIFICATION_VECTOR_DB_PERSIST_DIRECTORY` is explicitly provided.

The default container configuration sets `FACEVERIFICATION_DEVICE=cpu` to keep
deployment portable.

The shared local ChromaDB volume is intended for a small demo deployment when a
persist name is enabled. For a multi-container production setup with concurrent
writers or multiple replicas, use an external database/vector-store service or
make one service the clear owner of writes.

For production deployments, override at least:

```bash
FACEVERIFICATION_DEMO_USERNAME
FACEVERIFICATION_DEMO_PASSWORD
FACEVERIFICATION_JWT_SECRET_KEY
```

Example enrollment request:

```bash
curl -X POST http://localhost:8000/persons \
  -H "Authorization: Bearer <access_token>" \
  -F "name=Ada Lovelace" \
  -F "image=@test/images/person_anchor.jpg"
```

Example verification request:

```bash
curl -X POST http://localhost:8000/verify \
  -H "Authorization: Bearer <access_token>" \
  -F "image=@test/images/person_positive.jpg"
```

## Project Structure

```text
src/faceverification/
  config.py
  core/
    image_processor.py
    vectordb.py
  interfaces/
    fastapi_app.py
    gradio_app.py
  services/
    face_verification.py
```

## Demo Focus

This project highlights practical AI engineering skills: model-based feature extraction, vector database integration, application packaging, automated deployment, and a simple user-facing ML interface.
