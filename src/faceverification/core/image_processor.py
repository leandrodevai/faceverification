"""Face detection and FaceNet embedding utilities."""

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw

from faceverification.config import settings

logger = logging.getLogger(__name__)


class FaceNotDetectedError(ValueError):
    """Raised when no face can be detected in an image."""


class ImageProcessor:
    """Wrap MTCNN detection and FaceNet embedding extraction."""

    def __init__(
        self,
        device: str | None = None,
        mtcnn_thresholds: Sequence[float] | None = None,
        facenet_pretrained: str | None = None,
    ):
        """Load models using explicit values or application settings.

        Args:
            device: Inference device. Use `"auto"` to prefer CUDA when available.
            mtcnn_thresholds: Detection thresholds for the three MTCNN stages.
            facenet_pretrained: Pretrained FaceNet weights name.
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
        self.facenet = InceptionResnetV1(pretrained=facenet_pretrained).eval().to(self.device)
        logger.info(
            "image_processor_initialized",
            extra={
                "extra_fields": {
                    "event": "image_processor_initialized",
                    "device": self.device,
                    "mtcnn_thresholds": list(mtcnn_thresholds),
                    "facenet_pretrained": facenet_pretrained,
                }
            },
        )

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Return a normalized FaceNet embedding for the detected face.

        Args:
            image: PIL image containing a detectable face.

        Returns:
            One normalized FaceNet embedding tensor.

        Raises:
            FaceNotDetectedError: If MTCNN cannot extract a face.
        """
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            logger.debug(
                "face_embedding_no_face",
                extra={"extra_fields": {"event": "face_embedding_no_face"}},
            )
            raise FaceNotDetectedError("No face detected in the image.")

        face_tensor = (face_tensor.unsqueeze(0) if face_tensor.ndim == 3 else face_tensor).to(
            self.device
        )

        with torch.no_grad():
            features = self.facenet(face_tensor)
            features = F.normalize(features, p=2, dim=1)

        logger.debug(
            "face_embedding_created",
            extra={
                "extra_fields": {
                    "event": "face_embedding_created",
                    "shape": list(features.shape),
                    "device": self.device,
                }
            },
        )
        return features.squeeze(0)

    def detect_faces(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Draw face boxes and return whether any face was found.

        Args:
            image: PIL image to inspect and annotate.

        Returns:
            The annotated image and a flag indicating if at least one face was found.
        """
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None:
            logger.debug(
                "face_detection_completed",
                extra={"extra_fields": {"event": "face_detection_completed", "face_count": 0}},
            )
            return image, False

        probabilities = [float(prob) for prob in probs]
        logger.debug(
            "face_detection_completed",
            extra={
                "extra_fields": {
                    "event": "face_detection_completed",
                    "face_count": len(boxes),
                    "min_probability": min(probabilities),
                    "max_probability": max(probabilities),
                }
            },
        )

        draw = ImageDraw.Draw(image)
        for box, prob in zip(boxes, probs, strict=True):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), f"{prob:.4f}", fill="red")

        return image, True
