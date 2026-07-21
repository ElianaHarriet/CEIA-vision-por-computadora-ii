"""
Configuration management for data preparation.
Single Responsibility: Handle environment configuration.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoboflowConfig:
    """Roboflow API configuration."""
    api_key: str
    workspace: str
    project: str
    version: int

    @classmethod
    def from_env(cls, project_key: str, version_key: str) -> 'RoboflowConfig':
        """Create config from environment variables."""
        return cls(
            api_key=cls._get_required_env('ROBOFLOW_API_KEY'),
            workspace=cls._get_required_env('ROBOFLOW_WORKSPACE'),
            project=cls._get_required_env(project_key),
            version=int(os.getenv(version_key, '1'))
        )

    @staticmethod
    def _get_required_env(key: str) -> str:
        """Get required environment variable or raise."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} not configured")
        return value


@dataclass(frozen=True)
class DatasetPaths:
    """Dataset directory paths."""
    base: Path
    raw: Path
    ready: Path

    @classmethod
    def create(cls, base_dir: str, raw_name: str, ready_name: str) -> 'DatasetPaths':
        """Create dataset paths structure."""
        base = Path(base_dir)
        return cls(
            base=base,
            raw=base / raw_name,
            ready=base / ready_name
        )
