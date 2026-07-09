import os
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Set ROBOFLOW_API_KEY env var")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("project-p5nyc").project("car-damages-v3gyz")
dataset = project.version(5).download("coco-segmentation")
