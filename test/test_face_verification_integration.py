import importlib
import sys
import uuid

import numpy as np
import pytest
from PIL import Image

from faceverification.core import image_processor as image_processor_module
from faceverification.core import vectordb as vectordb_module
from faceverification.core.vectordb import VectorDB

pytestmark = pytest.mark.integration


class FakeEmbedding:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeImageProcessor:
    def __init__(self):
        self.embeddings = []

    def detect_faces(self, image):
        return image, True

    def get_embedding(self, _image):
        return FakeEmbedding(self.embeddings.pop(0))


@pytest.fixture
def service_with_real_vectordb(monkeypatch):
    fake_image_processor = FakeImageProcessor()
    vector_db = VectorDB(
        distance_metric="l2",
        name_collection=f"test_service_faces_{uuid.uuid4().hex}",
    )

    monkeypatch.setattr(
        image_processor_module,
        "ImageProcessor",
        lambda: fake_image_processor,
    )
    monkeypatch.setattr(vectordb_module, "VectorDB", lambda: vector_db)

    sys.modules.pop("faceverification.services.face_verification", None)
    service = importlib.import_module("faceverification.services.face_verification")

    return service, fake_image_processor


def test_add_then_verify_person_with_real_vectordb(service_with_real_vectordb):
    service, image_processor = service_with_real_vectordb
    image_processor.embeddings = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.11, 0.19, 0.31]),
    ]
    image = Image.new("RGB", (10, 10), "white")

    service.add_person(image, "Ada")
    name, annotated_image = service.verify_person(image)

    assert name == "Ada"
    assert annotated_image is not None


def test_verify_person_returns_unregistered_when_real_vectordb_match_is_too_far(
    service_with_real_vectordb,
):
    service, image_processor = service_with_real_vectordb
    image_processor.embeddings = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 10.0]),
    ]
    image = Image.new("RGB", (10, 10), "white")

    service.add_person(image, "Ada")
    name, annotated_image = service.verify_person(image)

    assert name == "Unregistered Person"
    assert annotated_image is not None
