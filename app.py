import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from faceverification.interfaces.gradio_app import FV_gr

if __name__ == "__main__":
    FV_gr.launch()
