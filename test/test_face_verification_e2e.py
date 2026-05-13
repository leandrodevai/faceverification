import importlib
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

from faceverification.core import vectordb as vectordb_module
from faceverification.core.vectordb import VectorDB

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

IMAGES_DIR = Path(__file__).parent / "images"


@pytest.fixture
def service_with_test_vectordb(monkeypatch):
    vector_db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_service_e2e_faces_{uuid.uuid4().hex}",
    )

    monkeypatch.setattr(vectordb_module, "VectorDB", lambda: vector_db)
    sys.modules.pop("faceverification.services.face_verification", None)
    service = importlib.import_module("faceverification.services.face_verification")

    yield service

    sys.modules.pop("faceverification.services.face_verification", None)


def test_service_enrolls_and_verifies_person_with_real_models(service_with_test_vectordb):
    service = service_with_test_vectordb
    anchor_image = Image.open(IMAGES_DIR / "person_anchor.jpg").convert("RGB")
    positive_image = Image.open(IMAGES_DIR / "person_positive.jpg").convert("RGB")

    service.add_person(anchor_image, "Ada")
    name, annotated_image = service.verify_person(positive_image)

    assert name == "Ada"
    assert annotated_image is not positive_image


def test_service_raises_before_enrollment(monkeypatch):
    vector_db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_empty_service_e2e_faces_{uuid.uuid4().hex}",
    )

    monkeypatch.setattr(vectordb_module, "VectorDB", lambda: vector_db)
    sys.modules.pop("faceverification.services.face_verification", None)
    service = importlib.import_module("faceverification.services.face_verification")
    image = Image.open(IMAGES_DIR / "person_anchor.jpg").convert("RGB")

    with pytest.raises(ValueError, match="Add a person before verifying faces"):
        service.verify_person(image)

    sys.modules.pop("faceverification.services.face_verification", None)
