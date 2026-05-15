"""Application service functions for face enrollment and verification.

This module coordinates image preprocessing, embedding extraction, and vector
database operations for the Gradio interface.
"""

from PIL import Image

from faceverification.core.image_processor import FaceNotDetectedError, ImageProcessor
from faceverification.core.vectordb import VectorDB

UNREGISTERED_PERSON = "Unregistered Person"

image_processor = ImageProcessor()

vector_db = VectorDB()


def add_person(image: Image.Image, name: str) -> Image.Image:
    """Enroll a person by extracting and storing their face embedding.

    Args:
        image: Input image provided by the UI as a PIL image.
        name: Person name to store as embedding metadata.

    Returns:
        The input image annotated with detected face bounding boxes.

    Raises:
        FaceNotDetectedError: If no face is detected in the input image.
        TypeError: If embedding extraction does not return the expected tensor.
    """

    img, presence = image_processor.detect_faces(image)

    if not presence:
        raise FaceNotDetectedError("No faces were detected in the image.")

    faces_pt = image_processor.get_embedding(img)
    if faces_pt is not None:
        vector_db.add_embedding(faces_pt.cpu().numpy(), {"name": name})
    else:
        raise TypeError("The extracted face embedding is not a torch.Tensor.")

    return img


def verify_person(image: Image.Image) -> tuple[str, Image.Image]:
    """Verify whether the input face matches a stored person.

    Args:
        image: Input image provided by the UI as a PIL image.

    Returns:
        A tuple with the matched person name, or `"Unregistered Person"` when no match is
        found, and the image annotated with detected face bounding boxes.

    Raises:
        FaceNotDetectedError: If no face is detected in the input image.
    """
    detected_faces, presence = image_processor.detect_faces(image.copy())

    if not presence:
        raise FaceNotDetectedError("No faces were detected in the image.")

    faces_pt = image_processor.get_embedding(image)
    if faces_pt is not None:
        metadata, _ = vector_db.query_embedding(faces_pt.cpu().numpy())

        if metadata:
            return metadata["name"], detected_faces

        return UNREGISTERED_PERSON, detected_faces

    raise FaceNotDetectedError("No faces were detected in the image.")
