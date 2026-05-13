"""Face detection and embedding extraction utilities.

This module centralizes the computer vision models used by the application:
MTCNN detects faces and draws bounding boxes, while FaceNet converts a detected
face into a normalized embedding suitable for vector search.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw

from faceverification.config import settings


class FaceNotDetectedError(ValueError):
    """Raised when no face can be detected in an image."""


class ImageProcessor:
    """Detect faces and generate FaceNet embeddings.

    The processor owns the model lifecycle for MTCNN and InceptionResnetV1. It
    accepts explicit configuration for tests or experiments, and falls back to
    application settings when arguments are omitted.
    """

    def __init__(
        self,
        device: str | None = None,
        mtcnn_thresholds: Sequence[float] | None = None,
        facenet_pretrained: str | None = None,
    ):
        """Initialize face detection and embedding models.

        Args:
            device: Device used for model inference. Use `"auto"` to select
                CUDA when available, otherwise CPU.
            mtcnn_thresholds: Detection thresholds for the three MTCNN stages.
            facenet_pretrained: Pretrained FaceNet weights identifier.

        Raises:
            ValueError: If `device` is not `"auto"`, `"cpu"`, or `"cuda"`.
        """
        if device is None:
            device = settings.device
        if mtcnn_thresholds is None:
            mtcnn_thresholds = settings.mtcnn_thresholds
        if facenet_pretrained is None:
            facenet_pretrained = settings.facenet_pretrained

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()
            if self.device not in ["cpu", "cuda"]:
                raise ValueError("Device must be 'auto', 'cpu' or 'cuda'.")
        self.mtcnn = MTCNN(
            select_largest=False,
            device=self.device,
            thresholds=list(mtcnn_thresholds),
        )
        self.facenet = (
            InceptionResnetV1(pretrained=facenet_pretrained).eval().to(self.device)
        )

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Return a normalized embedding for the detected face in an image.

        Args:
            image: PIL image containing a face.

        Returns:
            A one-dimensional normalized FaceNet embedding tensor.

        Raises:
            FaceNotDetectedError: If no face can be detected in the image.
        """
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            raise FaceNotDetectedError("No face detected in the image.")

        face_tensor = (
            face_tensor.unsqueeze(0) if face_tensor.ndim == 3 else face_tensor
        ).to(self.device)

        with torch.no_grad():
            features = self.facenet(face_tensor)
            features = F.normalize(features, p=2, dim=1)

        return features.squeeze(0)

    def detect_faces(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Draw detected face bounding boxes on an image.

        Args:
            image: PIL image to inspect and annotate.

        Returns:
            A tuple with the annotated image and a boolean indicating whether at
            least one face was detected.
        """
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None:
            return image, False

        draw = ImageDraw.Draw(image)
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"{prob:.4f}", fill="red")

        return image, True
