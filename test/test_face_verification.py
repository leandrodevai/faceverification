import importlib
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from faceverification.core import image_processor as image_processor_module
from faceverification.core import vectordb as vectordb_module
from faceverification.core.image_processor import FaceNotDetectedError


class FakeEmbedding:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def service_module(monkeypatch):
    fake_image_processor = Mock()
    fake_vector_db = Mock()

    monkeypatch.setattr(
        image_processor_module,
        "ImageProcessor",
        lambda: fake_image_processor,
    )
    monkeypatch.setattr(vectordb_module, "VectorDB", lambda: fake_vector_db)

    sys.modules.pop("faceverification.services.face_verification", None)
    module = importlib.import_module("faceverification.services.face_verification")

    return module, fake_image_processor, fake_vector_db


def test_add_person_stores_embedding_and_returns_annotated_image(service_module):
    service, image_processor, vector_db = service_module
    original_image = Image.new("RGB", (10, 10), "white")
    annotated_image = Image.new("RGB", (10, 10), "black")
    embedding = np.array([0.1, 0.2, 0.3])
    image_processor.detect_faces.return_value = (annotated_image, True)
    image_processor.get_embedding.return_value = FakeEmbedding(embedding)

    result = service.add_person(original_image, "Ada")

    assert result is annotated_image
    image_processor.detect_faces.assert_called_once_with(original_image)
    image_processor.get_embedding.assert_called_once_with(annotated_image)
    vector_db.add_embedding.assert_called_once()
    args = vector_db.add_embedding.call_args.args
    np.testing.assert_array_equal(args[0], embedding)
    assert args[1] == {"name": "Ada"}


def test_add_person_raises_when_no_face_is_detected(service_module):
    service, image_processor, vector_db = service_module
    image_processor.detect_faces.return_value = (Image.new("RGB", (10, 10)), False)

    with pytest.raises(FaceNotDetectedError, match="No faces were detected"):
        service.add_person(Image.new("RGB", (10, 10)), "Ada")

    image_processor.get_embedding.assert_not_called()
    vector_db.add_embedding.assert_not_called()


def test_add_person_raises_type_error_when_embedding_is_none(service_module):
    service, image_processor, vector_db = service_module
    image = Image.new("RGB", (10, 10))
    image_processor.detect_faces.return_value = (image, True)
    image_processor.get_embedding.return_value = None

    with pytest.raises(TypeError, match="extracted face embedding"):
        service.add_person(image, "Ada")

    vector_db.add_embedding.assert_not_called()


def test_verify_person_returns_matched_name_and_annotated_image(service_module):
    service, image_processor, vector_db = service_module
    original_image = Image.new("RGB", (10, 10), "white")
    detected_faces = Image.new("RGB", (10, 10), "black")
    embedding = np.array([0.4, 0.5, 0.6])
    image_processor.detect_faces.return_value = (detected_faces, True)
    image_processor.get_embedding.return_value = FakeEmbedding(embedding)
    vector_db.query_embedding.return_value = ({"name": "Grace"}, 0.2)

    name, image = service.verify_person(original_image)

    assert name == "Grace"
    assert image is detected_faces
    detected_input = image_processor.detect_faces.call_args.args[0]
    assert detected_input is not original_image
    image_processor.get_embedding.assert_called_once_with(original_image)
    vector_db.query_embedding.assert_called_once()
    np.testing.assert_array_equal(vector_db.query_embedding.call_args.args[0], embedding)


def test_verify_person_returns_unregistered_when_no_metadata_matches(service_module):
    service, image_processor, vector_db = service_module
    detected_faces = Image.new("RGB", (10, 10))
    image_processor.detect_faces.return_value = (detected_faces, True)
    image_processor.get_embedding.return_value = FakeEmbedding(np.array([0.1, 0.2]))
    vector_db.query_embedding.return_value = (None, 1.5)

    name, image = service.verify_person(Image.new("RGB", (10, 10)))

    assert name == "Unregistered Person"
    assert image is detected_faces


def test_verify_person_raises_when_vector_db_has_no_records(service_module):
    service, image_processor, vector_db = service_module
    detected_faces = Image.new("RGB", (10, 10))
    image_processor.detect_faces.return_value = (detected_faces, True)
    image_processor.get_embedding.return_value = FakeEmbedding(np.array([0.1, 0.2]))
    vector_db.query_embedding.side_effect = ValueError(
        "No record found in the vector database. Add a person before verifying faces."
    )

    with pytest.raises(ValueError, match="Add a person before verifying faces"):
        service.verify_person(Image.new("RGB", (10, 10)))


def test_verify_person_raises_when_no_face_is_detected(service_module):
    service, image_processor, vector_db = service_module
    image_processor.detect_faces.return_value = (Image.new("RGB", (10, 10)), False)

    with pytest.raises(FaceNotDetectedError, match="No faces were detected"):
        service.verify_person(Image.new("RGB", (10, 10)))

    image_processor.get_embedding.assert_not_called()
    vector_db.query_embedding.assert_not_called()


def test_verify_person_raises_when_embedding_is_none(service_module):
    service, image_processor, vector_db = service_module
    image_processor.detect_faces.return_value = (Image.new("RGB", (10, 10)), True)
    image_processor.get_embedding.return_value = None

    with pytest.raises(FaceNotDetectedError, match="No faces were detected"):
        service.verify_person(Image.new("RGB", (10, 10)))

    vector_db.query_embedding.assert_not_called()
