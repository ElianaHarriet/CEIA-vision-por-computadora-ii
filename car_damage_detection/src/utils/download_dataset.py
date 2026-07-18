import os
import subprocess
import sys
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "eliharriet"
PROJECT = "car-damages-v3gyz-r4aou"
VERSION = 1
TARGET = Path("car_damage_detection/data/car-damages-forked")


def main():
    API_KEY = os.environ.get("ROBOFLOW_API_KEY")
    if not API_KEY:
        raise ValueError("Set ROBOFLOW_API_KEY env var, e.g. in .env file")

    if not TARGET.exists():
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        project.version(VERSION).download("coco-segmentation", location=str(TARGET))
        print("✅ Downloaded.")
    else:
        print("✅ Already downloaded.")

    subprocess.run([sys.executable, "car_damage_detection/src/utils/prepare_datasets.py"], check=True)


if __name__ == "__main__":
    main()
