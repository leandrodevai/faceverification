"""HTTP API for enrolling and verifying faces.

The module exposes a small FastAPI application around the service layer:

- ``POST /auth/login`` issues a short-lived JWT for the demo user.
- ``POST /persons`` stores a known person embedding from an uploaded image.
- ``POST /verify`` checks whether an uploaded face matches the local database.

FastAPI uses the route metadata, Pydantic field descriptions, and endpoint
docstrings below to build the interactive documentation at ``/docs`` and
``/redoc``.
"""

from base64 import b64encode
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import BytesIO
from secrets import compare_digest
from types import ModuleType
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from faceverification.config import settings
from faceverification.core.image_processor import FaceNotDetectedError

bearer_scheme = HTTPBearer(auto_error=False)

DATA_URL_DESCRIPTION = (
    "PNG image encoded as a data URL. The image contains the service annotations "
    "for the detected face."
)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the face verification service once when the API starts."""
    from faceverification.services import face_verification

    app.state.face_service = face_verification
    yield


app = FastAPI(
    title="Face Verification API",
    description=(
        "Demo API for enrolling known people and verifying uploaded face images. "
        "Authenticate with `/auth/login`, then send the returned bearer token to "
        "the protected face-verification endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "Leandro",
        "url": "https://github.com/leandrodevai/faceverification",
    },
)


class HealthResponse(BaseModel):
    """Health-check payload returned by the system endpoint."""

    status: str = Field(default="ok", description="Current API status.")


class EnrollResponse(BaseModel):
    """Response returned after a person is stored in the embeddings database."""

    message: str = Field(description="Human-readable result message.")
    name: str = Field(description="Normalized person name stored with the embedding.")
    annotated_image: str = Field(description=DATA_URL_DESCRIPTION)

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
    """Response returned after comparing an uploaded face against known people."""

    name: str = Field(
        description=(
            "Matched person name. Returns `Unregistered Person` when the closest "
            "embedding is outside the configured match threshold."
        ),
    )
    matched: bool = Field(description="Whether the uploaded face matched a known person.")
    annotated_image: str = Field(description=DATA_URL_DESCRIPTION)

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
    """Bearer token returned by the demo authentication endpoint."""

    access_token: str = Field(description="JWT access token used in the Authorization header.")
    token_type: str = Field(default="bearer", description="OAuth2-compatible token type.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            },
        },
    }


def get_face_service(request: Request) -> ModuleType:
    """Return the service module stored during application startup."""
    return request.app.state.face_service


def _unauthorized_error(detail: str = "Could not validate credentials.") -> HTTPException:
    """Build a consistent 401 response with the bearer authentication challenge."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _create_access_token(username: str) -> str:
    """Create a signed JWT for the authenticated demo user."""
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {"sub": username, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _authenticate_demo_user(username: str, password: str) -> bool:
    """Validate demo credentials using constant-time comparisons."""
    valid_username = compare_digest(username, settings.demo_username)
    valid_password = compare_digest(password, settings.demo_password)
    return valid_username and valid_password


def get_current_username(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    """Decode the bearer token and return the authenticated username."""
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
    """Read an uploaded image, apply EXIF orientation, and return it as RGB."""
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
    """Serialize a PIL image as a PNG data URL for JSON responses."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _service_error(exc: Exception) -> HTTPException:
    """Map service-layer exceptions to API-friendly HTTP errors."""
    if isinstance(exc, FaceNotDetectedError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Face verification failed.",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    response_description="The API is running.",
    tags=["system"],
)
def health() -> HealthResponse:
    """Return a lightweight status response for uptime checks."""
    return HealthResponse()


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Issue a demo access token",
    response_description="JWT bearer token for protected endpoints.",
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
        Form(description="Demo username configured with `FACEVERIFICATION_DEMO_USERNAME`."),
    ],
    password: Annotated[
        str,
        Form(description="Demo password configured with `FACEVERIFICATION_DEMO_PASSWORD`."),
    ],
) -> TokenResponse:
    """Authenticate the demo user and return a signed JWT access token."""
    if not _authenticate_demo_user(username, password):
        raise _unauthorized_error("Incorrect username or password.")

    return TokenResponse(access_token=_create_access_token(username))


@app.post(
    "/persons",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a known person",
    response_description="The person was stored and the annotated upload is returned.",
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
    service: Annotated[ModuleType, Depends(get_face_service)],
) -> EnrollResponse:
    """Store a new known person in the embeddings database.

    The endpoint extracts a face embedding from the uploaded image and stores it
    under the submitted name. It returns the normalized name and a PNG data URL
    with the annotated detection result.
    """
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Person name is required.",
        )

    pil_image = await _read_image(image)
    try:
        annotated_image = service.add_person(pil_image, cleaned_name)
    except Exception as exc:
        raise _service_error(exc) from exc

    return EnrollResponse(
        message="Person added to the embeddings database.",
        name=cleaned_name,
        annotated_image=_image_to_data_url(annotated_image),
    )


@app.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify an uploaded face",
    response_description="Best match result and the annotated upload.",
    responses={**AUTH_RESPONSES, **IMAGE_ERROR_RESPONSES},
    tags=["face verification"],
)
async def verify_identity(
    image: Annotated[
        UploadFile,
        File(description="Image containing one clear face to compare with known people."),
    ],
    current_username: Annotated[str, Depends(get_current_username)],
    service: Annotated[ModuleType, Depends(get_face_service)],
) -> VerifyResponse:
    """Compare an uploaded face against the local embeddings database.

    A successful response always includes the closest label and whether it is
    considered a match according to the configured distance threshold.
    """
    pil_image = await _read_image(image)
    try:
        name, annotated_image = service.verify_person(pil_image)
    except Exception as exc:
        raise _service_error(exc) from exc

    return VerifyResponse(
        name=name,
        matched=name != "Unregistered Person",
        annotated_image=_image_to_data_url(annotated_image),
    )


def main() -> None:
    import uvicorn

    uvicorn.run("faceverification.interfaces.fastapi_app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
