import gradio as gr
from PIL import Image

from faceverification.core.image_processor import FaceNotDetectedError
from faceverification.services.face_verification import add_person, verify_person


def add_person_ui(image: Image.Image, name: str) -> Image.Image:
    try:
        return add_person(image, name)
    except FaceNotDetectedError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def verify_person_ui(image: Image.Image) -> tuple[str, Image.Image]:
    try:
        return verify_person(image)
    except FaceNotDetectedError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks() as FV_gr:
    gr.Markdown(
        "# Face Verification Tool\n"
        "Upload an image and assign a name to add it to the embeddings database."
    )
    with gr.Tabs():
        with gr.TabItem("Add Person"):
            with gr.Row():
                input_image = gr.Image(label="Upload an image with a clear face", type="pil")
                input_name = gr.Textbox(label="Person name", placeholder="Example: John Doe")
            output_image = gr.Image(label="Detection result")
            submit_btn = gr.Button("Add to database")
            submit_btn.click(
                fn=add_person_ui,
                inputs=[input_image, input_name],
                outputs=output_image,
            )
        with gr.TabItem("Verify Identity"):
            with gr.Row():
                verify_image = gr.Image(label="Upload an image to verify", type="pil")
            verify_btn = gr.Button("Verify identity")
            verify_output_name = gr.Textbox(label="Verification result")
            verify_output_image = gr.Image(label="Detected face")
            verify_btn.click(
                fn=verify_person_ui,
                inputs=verify_image,
                outputs=[verify_output_name, verify_output_image],
            )


def main():
    FV_gr.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
