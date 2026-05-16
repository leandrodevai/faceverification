from functools import cache
from os import getenv

import gradio as gr
from PIL import Image

from faceverification.core.image_processor import FaceNotDetectedError
from faceverification.logging_config import configure_logging

configure_logging()


@cache
def _face_service():
    from faceverification.services.face_verification import add_person, verify_person

    return add_person, verify_person


def add_person_ui(image: Image.Image | None, name: str) -> Image.Image:
    try:
        if image is None:
            raise gr.Error("Upload an image before adding a person.")
        if not name or not name.strip():
            raise gr.Error("Enter a name before adding the person.")

        add_person, _ = _face_service()
        return add_person(image, name.strip())
    except FaceNotDetectedError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def verify_person_ui(image: Image.Image | None) -> tuple[str, Image.Image]:
    try:
        if image is None:
            raise gr.Error("Upload an image before verifying an identity.")

        _, verify_person = _face_service()
        name, annotated_image = verify_person(image)
        return name, annotated_image
    except FaceNotDetectedError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


APP_CSS = """
.app-shell {
    max-width: 1040px;
    margin: 0 auto;
}

.hero {
    padding: 18px 2px 8px;
}

.hero h1 {
    margin-bottom: 8px;
}

.hero p {
    max-width: 760px;
    color: var(--body-text-color-subdued);
    font-size: 1rem;
}

.hint {
    color: var(--body-text-color-subdued);
    font-size: 0.94rem;
    line-height: 1.5;
}

.result-box textarea {
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

.primary-action {
    margin-top: 8px;
}
"""

APP_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="green")


with (
    gr.Blocks(
        title="Face Verification Demo",
    ) as FV_gr,
    gr.Column(elem_classes="app-shell"),
):
    gr.HTML(
        """
<div class="hero">
    <h1>Face Verification Demo</h1>
    <p>
        Register a known face, then upload another image to check whether it
        matches an identity stored in the database.
    </p>
</div>
            """
    )

    gr.HTML(
        """
<span class="hint">
Use clear, front-facing photos with one visible face. The demo stores data
only for this running session, so the database may reset when the Space restarts.
</span>
            """
    )

    with gr.Tabs():
        with gr.TabItem("1. Add Person"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Reference image",
                        type="pil",
                        height=360,
                    )
                    input_name = gr.Textbox(
                        label="Person name",
                        placeholder="Example: John Doe",
                    )
                    submit_btn = gr.Button(
                        "Add person to database",
                        variant="primary",
                        elem_classes="primary-action",
                    )
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Detected face",
                        height=420,
                    )
                    gr.HTML(
                        """
<span class="hint">
If a face is detected, the annotated image confirms what face was stored.
</span>
                            """
                    )

            submit_btn.click(
                fn=add_person_ui,
                inputs=[input_image, input_name],
                outputs=output_image,
            )

        with gr.TabItem("2. Verify Identity"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    verify_image = gr.Image(
                        label="Image to verify",
                        type="pil",
                        height=360,
                    )
                    verify_btn = gr.Button(
                        "Verify identity",
                        variant="primary",
                        elem_classes="primary-action",
                    )
                with gr.Column(scale=1):
                    verify_output_name = gr.Textbox(
                        label="Verification result",
                        interactive=False,
                        elem_classes="result-box",
                    )
                    verify_output_image = gr.Image(
                        label="Detected face",
                        height=360,
                    )

            verify_btn.click(
                fn=verify_person_ui,
                inputs=verify_image,
                outputs=[verify_output_name, verify_output_image],
            )


def main():
    FV_gr.launch(
        server_name=getenv("GRADIO_SERVER_NAME") or None,
        server_port=int(getenv("GRADIO_SERVER_PORT", "7860")),
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
