import os
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Set ROBOFLOW_API_KEY env var")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("college-gxdrt").project("car-damage-detection-ha5mm")
dataset = project.version(1).download("yolov8")
