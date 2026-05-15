"""Application service functions for face enrollment and verification."""

from PIL import Image

from faceverification.core.image_processor import FaceNotDetectedError, ImageProcessor
from faceverification.core.vectordb import VectorDB

UNREGISTERED_PERSON = "Unregistered Person"

image_processor = ImageProcessor()

vector_db = VectorDB()


def add_person(image: Image.Image, name: str) -> Image.Image:
    """Store a person's face embedding and return the annotated image.

    Args:
        image: PIL image containing the person's face.
        name: Person name to store with the embedding.

    Returns:
        Image annotated with detected face boxes.

    Raises:
        FaceNotDetectedError: If no face is detected.
        TypeError: If embedding extraction returns an unexpected value.
    """

    img, presence = image_processor.detect_faces(image)

    if not presence:
        raise FaceNotDetectedError("No faces were detected in the image.")

    faces_pt = image_processor.get_embedding(img)
    if faces_pt is None:
        raise TypeError("The extracted face embedding is not a torch.Tensor.")

    vector_db.add_embedding(faces_pt.cpu().numpy(), {"name": name})

    return img


def verify_person(image: Image.Image) -> tuple[str, Image.Image]:
    """Return the closest known person name and the annotated image.

    Args:
        image: PIL image containing the face to verify.

    Returns:
        Matched person name, or `UNREGISTERED_PERSON`, plus the annotated image.

    Raises:
        FaceNotDetectedError: If no face is detected.
        ValueError: If the vector database has no stored embeddings.
    """
    detected_faces, presence = image_processor.detect_faces(image.copy())

    if not presence:
        raise FaceNotDetectedError("No faces were detected in the image.")

    faces_pt = image_processor.get_embedding(image)
    if faces_pt is None:
        raise FaceNotDetectedError("No faces were detected in the image.")

    metadata, _ = vector_db.query_embedding(faces_pt.cpu().numpy())

    if metadata:
        return metadata["name"], detected_faces

    return UNREGISTERED_PERSON, detected_faces
