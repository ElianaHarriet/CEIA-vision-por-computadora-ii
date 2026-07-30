"""
Roboflow API client.
Single Responsibility: Handle Roboflow API interactions.
"""
from pathlib import Path
import sys

dag_path = Path(__file__).parent.parent
if str(dag_path) not in sys.path:
    sys.path.insert(0, str(dag_path))

from roboflow import Roboflow
from core.config import RoboflowConfig


class RoboflowClient:
    """Client for Roboflow API operations."""

    def __init__(self, config: RoboflowConfig):
        """Initialize client with configuration."""
        self._config = config
        self._client = Roboflow(api_key=config.api_key)

    def download_dataset(self, format_type: str, location: Path) -> None:
        """Download dataset in specified format."""
        project = self._get_project()
        version = project.version(self._config.version)
        version.download(format_type, location=str(location))

    def _get_project(self):
        """Get Roboflow project instance."""
        workspace = self._client.workspace(self._config.workspace)
        return workspace.project(self._config.project)
