from pathlib import Path

import pytest
import torch
from PIL import Image

from faceverification.core.image_processor import ImageProcessor

pytestmark = pytest.mark.integration

IMAGES_DIR = Path(__file__).parent / "images"


@pytest.fixture(scope="module")
def processor():
    return ImageProcessor(device="cpu")


@pytest.fixture(scope="module")
def anchor_image():
    return Image.open(IMAGES_DIR / "person_anchor.jpg").convert("RGB")


@pytest.fixture(scope="module")
def positive_image():
    return Image.open(IMAGES_DIR / "person_positive.jpg").convert("RGB")


@pytest.fixture(scope="module")
def negative_image():
    return Image.open(IMAGES_DIR / "other_people_negative.jpg").convert("RGB")


@pytest.mark.parametrize(
    "image_name",
    [
        "person_anchor.jpg",
        "person_positive.jpg",
        "other_people_negative.jpg",
    ],
)
def test_detect_faces_finds_faces_in_fixture_images(processor, image_name):
    image = Image.open(IMAGES_DIR / image_name).convert("RGB")

    annotated_image, presence = processor.detect_faces(image)

    assert annotated_image is image
    assert presence is True


def test_get_embedding_returns_normalized_facenet_vector(processor, anchor_image):
    embedding = processor.get_embedding(anchor_image)

    assert embedding.ndim == 1
    assert embedding.shape[0] == 512
    torch.testing.assert_close(torch.linalg.norm(embedding), torch.tensor(1.0))


def test_positive_image_embedding_is_closer_than_negative_image(
    processor,
    anchor_image,
    positive_image,
    negative_image,
):
    anchor_embedding = processor.get_embedding(anchor_image)
    positive_embedding = processor.get_embedding(positive_image)
    negative_embedding = processor.get_embedding(negative_image)

    positive_distance = torch.linalg.norm(anchor_embedding - positive_embedding)
    negative_distance = torch.linalg.norm(anchor_embedding - negative_embedding)

    assert positive_distance < negative_distance
