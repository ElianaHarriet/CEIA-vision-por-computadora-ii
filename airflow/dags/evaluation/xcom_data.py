"""Helpers to pass large numpy-array payloads between tasks via disk.

Airflow's XCom backend serializes values as JSON, which cannot handle
numpy arrays (masks, predictions, images). Instead of pushing that data
directly, tasks pickle it to a shared path and push the path through XCom.
"""
import pickle
from pathlib import Path


def save_pickle(data, path: str) -> str:
    """Pickle data to path, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def load_pickle(path: str):
    """Load pickled data from path."""
    with open(path, 'rb') as f:
        return pickle.load(f)
