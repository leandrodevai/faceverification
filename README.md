---
title: Face Verification
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0
python_version: '3.11'
app_file: app.py
short_description: "Face verification demo with FaceNet and VectorDB"
tags: ["face-verification", "computer-vision", "facenet", "vector-database"]
pinned: false
license: mit
---
![CI](https://github.com/leandrodevai/faceverification/actions/workflows/ci.yml/badge.svg)
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
- Hugging Face Spaces deployment-ready structure

## Tech Stack

- Python
- PyTorch
- FaceNet / facenet-pytorch
- ChromaDB
- Gradio
- Hugging Face Spaces

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

## Project Structure

```text
src/faceverification/
  app.py
  config.py
  core/
    image_processor.py
    vectordb.py
  services/
    face_verification.py
```

## Demo Focus

This project highlights practical AI engineering skills: model-based feature extraction, vector database integration, application packaging, automated deployment, and a simple user-facing ML interface.
