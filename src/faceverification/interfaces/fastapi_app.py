import logging
from base64 import b64encode
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import BytesIO
from secrets import compare_digest
from typing import Annotated, Protocol

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from faceverification.config import settings
from faceverification.core.image_processor import FaceNotDetectedError
from faceverification.services.face_verification import UNREGISTERED_PERSON

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)

AUTH_RESPONSES = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": "The request is missing a bearer token or the token is invalid.",
        "content": {
            "application/json": {
                "example": {"detail": "Not authenticated"},
            },
        },
    },
}

IMAGE_ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {
        "description": "The uploaded file is empty, invalid, or rejected by the service.",
        "content": {
            "application/json": {
                "example": {"detail": "Uploaded file is not a valid image."},
            },
        },
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "description": "The uploaded file content type is not an image.",
        "content": {
            "application/json": {
                "example": {"detail": "Uploaded file must be an image."},
            },
        },
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "description": "The uploaded image is too large.",
        "content": {
            "application/json": {
                "example": {"detail": "Uploaded image is too large."},
            },
        },
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": "The request is valid, but no usable face or name was found.",
        "content": {
            "application/json": {
                "example": {"detail": "No faces were detected in the image."},
            },
        },
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "description": "The face verification pipeline failed unexpectedly.",
        "content": {
            "application/json": {
                "example": {"detail": "Face verification failed."},
            },
        },
    },
}


class FaceService(Protocol):
    """Service contract required by the HTTP layer.

    Implementations must accept normalized PIL images and return annotated PIL
    images for optional API responses.
    """

    def add_person(self, image: Image.Image, name: str) -> Image.Image: ...

    def verify_person(self, image: Image.Image) -> tuple[str, Image.Image]: ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    from faceverification.services import face_verification

    app.state.face_service = face_verification
    yield


app = FastAPI(
    title="Face Verification API",
    description="Enroll known people and verify uploaded face images.",
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "Leandro",
        "url": "https://github.com/leandrodevai/faceverification",
    },
)


class HealthResponse(BaseModel):
    status: str = "ok"


class EnrollResponse(BaseModel):
    message: str
    name: str
    annotated_image: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Person added to the embeddings database.",
                "name": "Ada Lovelace",
                "annotated_image": "data:image/png;base64,iVBORw0KGgo...",
            },
        },
    }


class VerifyResponse(BaseModel):
    name: str
    matched: bool
    annotated_image: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Ada Lovelace",
                "matched": True,
                "annotated_image": "data:image/png;base64,iVBORw0KGgo...",
            },
        },
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            },
        },
    }


def get_face_service(request: Request) -> FaceService:
    return request.app.state.face_service


def _unauthorized_error(detail: str = "Could not validate credentials.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _create_access_token(username: str) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {"sub": username, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _authenticate_demo_user(username: str, password: str) -> bool:
    valid_username = compare_digest(username, settings.demo_username)
    valid_password = compare_digest(password, settings.demo_password)
    return valid_username and valid_password


def get_current_username(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    if credentials is None:
        raise _unauthorized_error("Not authenticated")

    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError as exc:
        raise _unauthorized_error("Token has expired.") from exc
    except InvalidTokenError as exc:
        raise _unauthorized_error() from exc

    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise _unauthorized_error()

    return username


async def _read_image(upload: UploadFile) -> Image.Image:
    """Validate an upload and return it as an RGB image.

    Args:
        upload: Multipart file received by FastAPI.

    Returns:
        RGB PIL image with EXIF orientation applied.

    Raises:
        HTTPException: If the file is not an image, is empty, too large, or invalid.
    """

    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image.",
        )

    contents = await upload.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded image is too large.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from exc


def _image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _face_not_detected_error(exc: FaceNotDetectedError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=str(exc),
    )


def _bad_service_request(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc),
    )


def _unexpected_service_error(exc: Exception) -> HTTPException:
    logger.exception("Face verification service failed")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Face verification failed.",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    tags=["system"],
)
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Log in",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "The username or password is incorrect.",
            "content": {
                "application/json": {
                    "example": {"detail": "Incorrect username or password."},
                },
            },
        },
    },
    tags=["auth"],
)
def login(
    username: Annotated[
        str,
        Form(),
    ],
    password: Annotated[
        str,
        Form(),
    ],
) -> TokenResponse:
    if not _authenticate_demo_user(username, password):
        raise _unauthorized_error("Incorrect username or password.")

    return TokenResponse(access_token=_create_access_token(username))


@app.post(
    "/persons",
    response_model=EnrollResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a known person",
    responses={**AUTH_RESPONSES, **IMAGE_ERROR_RESPONSES},
    tags=["face verification"],
)
async def enroll_person(
    image: Annotated[
        UploadFile,
        File(description="Image containing one clear face to enroll."),
    ],
    name: Annotated[
        str,
        Form(description="Person name to associate with the generated face embedding."),
    ],
    current_username: Annotated[str, Depends(get_current_username)],
    service: Annotated[FaceService, Depends(get_face_service)],
    include_image: Annotated[
        bool,
        Query(description="Include the annotated image as a base64 data URL."),
    ] = True,
) -> EnrollResponse:
    _ = current_username
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Person name is required.",
        )

    pil_image = await _read_image(image)
    try:
        annotated_image = service.add_person(pil_image, cleaned_name)
    except FaceNotDetectedError as exc:
        raise _face_not_detected_error(exc) from exc
    except ValueError as exc:
        raise _bad_service_request(exc) from exc
    except Exception as exc:
        raise _unexpected_service_error(exc) from exc

    return EnrollResponse(
        message="Person added to the embeddings database.",
        name=cleaned_name,
        annotated_image=_image_to_data_url(annotated_image) if include_image else None,
    )


@app.post(
    "/verify",
    response_model=VerifyResponse,
    response_model_exclude_none=True,
    summary="Verify an uploaded face",
    responses={**AUTH_RESPONSES, **IMAGE_ERROR_RESPONSES},
    tags=["face verification"],
)
async def verify_identity(
    image: Annotated[
        UploadFile,
        File(description="Image containing one clear face to compare with known people."),
    ],
    current_username: Annotated[str, Depends(get_current_username)],
    service: Annotated[FaceService, Depends(get_face_service)],
    include_image: Annotated[
        bool,
        Query(description="Include the annotated image as a base64 data URL."),
    ] = True,
) -> VerifyResponse:
    _ = current_username
    pil_image = await _read_image(image)
    try:
        name, annotated_image = service.verify_person(pil_image)
    except FaceNotDetectedError as exc:
        raise _face_not_detected_error(exc) from exc
    except ValueError as exc:
        raise _bad_service_request(exc) from exc
    except Exception as exc:
        raise _unexpected_service_error(exc) from exc

    return VerifyResponse(
        name=name,
        matched=name != UNREGISTERED_PERSON,
        annotated_image=_image_to_data_url(annotated_image) if include_image else None,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("faceverification.interfaces.fastapi_app:app", port=8000)


if __name__ == "__main__":
    main()
