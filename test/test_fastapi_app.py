from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from faceverification.core.image_processor import FaceNotDetectedError
from faceverification.interfaces.fastapi_app import app, get_face_service


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 12), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeService:
    def __init__(self):
        self.add_person_calls = []
        self.verify_person_calls = []

    def add_person(self, image, name):
        self.add_person_calls.append((image, name))
        return Image.new("RGB", image.size, "black")

    def verify_person(self, image):
        self.verify_person_calls.append(image)
        return "Ada", Image.new("RGB", image.size, "black")


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        data={"username": "demo", "password": "demo123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_login_returns_access_token():
    client = TestClient(app)

    response = client.post(
        "/auth/login",
        data={"username": "demo", "password": "demo123"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_rejects_invalid_credentials():
    client = TestClient(app)

    response = client.post(
        "/auth/login",
        data={"username": "demo", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password."}


def test_verify_identity_requires_token():
    client = TestClient(app)

    response = client.post(
        "/verify",
        files={"image": ("face.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_enroll_person_calls_service_and_returns_annotated_image():
    fake_service = FakeService()
    app.dependency_overrides[get_face_service] = lambda: fake_service
    try:
        client = TestClient(app)
        response = client.post(
            "/persons",
            headers=_auth_headers(client),
            data={"name": " Ada "},
            files={"image": ("face.png", _image_bytes(), "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 201
    assert body["name"] == "Ada"
    assert body["message"] == "Person added to the embeddings database."
    assert body["annotated_image"].startswith("data:image/png;base64,")
    assert fake_service.add_person_calls[0][1] == "Ada"


def test_verify_identity_returns_match_result():
    fake_service = FakeService()
    app.dependency_overrides[get_face_service] = lambda: fake_service
    try:
        client = TestClient(app)
        response = client.post(
            "/verify",
            headers=_auth_headers(client),
            files={"image": ("face.png", _image_bytes(), "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["name"] == "Ada"
    assert body["matched"] is True
    assert body["annotated_image"].startswith("data:image/png;base64,")


def test_verify_identity_returns_unprocessable_when_no_face_is_detected():
    class NoFaceService(FakeService):
        def verify_person(self, image):
            raise FaceNotDetectedError("No faces were detected in the image.")

    app.dependency_overrides[get_face_service] = lambda: NoFaceService()
    try:
        client = TestClient(app)
        response = client.post(
            "/verify",
            headers=_auth_headers(client),
            files={"image": ("face.png", _image_bytes(), "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {"detail": "No faces were detected in the image."}


def test_enroll_person_rejects_blank_name():
    app.dependency_overrides[get_face_service] = lambda: FakeService()
    try:
        client = TestClient(app)
        response = client.post(
            "/persons",
            headers=_auth_headers(client),
            data={"name": "   "},
            files={"image": ("face.png", _image_bytes(), "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {"detail": "Person name is required."}


def test_upload_rejects_non_image_content_type():
    app.dependency_overrides[get_face_service] = lambda: FakeService()
    try:
        client = TestClient(app)
        response = client.post(
            "/verify",
            headers=_auth_headers(client),
            files={"image": ("face.txt", b"hello", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 415
    assert response.json() == {"detail": "Uploaded file must be an image."}
