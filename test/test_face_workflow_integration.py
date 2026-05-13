import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from faceverification.core.image_processor import ImageProcessor
from faceverification.core.vectordb import VectorDB

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

IMAGES_DIR = Path(__file__).parent / "images"


def test_real_image_embedding_can_be_enrolled_and_matched_in_vectordb():
    processor = ImageProcessor(device="cpu")
    vector_db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_real_image_faces_{uuid.uuid4().hex}",
    )
    anchor_image = Image.open(IMAGES_DIR / "person_anchor.jpg").convert("RGB")
    positive_image = Image.open(IMAGES_DIR / "person_positive.jpg").convert("RGB")

    anchor_embedding = processor.get_embedding(anchor_image).cpu().numpy()
    positive_embedding = processor.get_embedding(positive_image).cpu().numpy()
    vector_db.add_embedding(anchor_embedding, {"name": "Ada"})

    metadata, distance = vector_db.query_embedding(
        positive_embedding,
        threshold=1.08,
        n_results=1,
    )

    assert metadata == {"name": "Ada"}
    assert distance == pytest.approx(np.linalg.norm(positive_embedding - anchor_embedding))
