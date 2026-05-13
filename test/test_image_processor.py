from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from faceverification.core import image_processor
from faceverification.core.image_processor import FaceNotDetectedError, ImageProcessor


class FakeMTCNN:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeMTCNN.instances.append(self)


class FakeFacenet:
    instances = []

    def __init__(self, pretrained):
        self.pretrained = pretrained
        self.eval_called = False
        self.device = None
        FakeFacenet.instances.append(self)

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


def test_init_uses_cpu_when_auto_and_cuda_is_unavailable(monkeypatch):
    FakeMTCNN.instances = []
    FakeFacenet.instances = []
    monkeypatch.setattr(image_processor.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(image_processor, "MTCNN", FakeMTCNN)
    monkeypatch.setattr(image_processor, "InceptionResnetV1", FakeFacenet)

    processor = ImageProcessor(
        device="auto",
        mtcnn_thresholds=(0.1, 0.2, 0.3),
        facenet_pretrained="test-weights",
    )

    assert processor.device == "cpu"
    assert FakeMTCNN.instances[0].kwargs == {
        "select_largest": False,
        "device": "cpu",
        "thresholds": [0.1, 0.2, 0.3],
    }
    assert FakeFacenet.instances[0].pretrained == "test-weights"
    assert FakeFacenet.instances[0].eval_called is True
    assert FakeFacenet.instances[0].device == "cpu"


def test_init_rejects_invalid_device():
    with pytest.raises(ValueError, match="Device must be"):
        ImageProcessor(device="gpu")


def test_get_embedding_returns_normalized_embedding():
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.device = "cpu"
    processor.mtcnn = Mock(return_value=torch.ones((3, 2, 2)))
    processor.facenet = Mock(return_value=torch.tensor([[3.0, 4.0]]))
    image = Image.new("RGB", (10, 10))

    embedding = processor.get_embedding(image)

    torch.testing.assert_close(embedding, torch.tensor([0.6, 0.8]))
    processor.mtcnn.assert_called_once_with(image)
    assert processor.facenet.call_args.args[0].shape == torch.Size([1, 3, 2, 2])


def test_get_embedding_keeps_batched_face_tensor_shape():
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.device = "cpu"
    processor.mtcnn = Mock(return_value=torch.ones((1, 3, 2, 2)))
    processor.facenet = Mock(return_value=torch.tensor([[3.0, 4.0]]))
    image = Image.new("RGB", (10, 10))

    embedding = processor.get_embedding(image)

    torch.testing.assert_close(embedding, torch.tensor([0.6, 0.8]))
    assert processor.facenet.call_args.args[0].shape == torch.Size([1, 3, 2, 2])


def test_get_embedding_raises_when_no_face_is_detected():
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.mtcnn = Mock(return_value=None)

    with pytest.raises(FaceNotDetectedError, match="No face detected"):
        processor.get_embedding(Image.new("RGB", (10, 10)))


def test_detect_faces_draws_boxes_and_returns_true():
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.mtcnn = Mock()
    processor.mtcnn.detect.return_value = (
        [[1.0, 1.0, 8.0, 8.0]],
        [0.98765],
    )
    image = Image.new("RGB", (10, 10), "white")

    annotated_image, presence = processor.detect_faces(image)

    assert annotated_image is image
    assert presence is True
    assert image.getpixel((1, 1)) == (255, 0, 0)


def test_detect_faces_returns_false_when_no_boxes_are_detected():
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.mtcnn = Mock()
    processor.mtcnn.detect.return_value = (None, None)
    image = Image.new("RGB", (10, 10), "white")

    annotated_image, presence = processor.detect_faces(image)

    assert annotated_image is image
    assert presence is False
