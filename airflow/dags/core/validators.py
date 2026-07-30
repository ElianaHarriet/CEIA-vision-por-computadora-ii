"""
Data validation utilities.
Single Responsibility: Validate dataset integrity.
"""
from pathlib import Path
from typing import List, Dict


class DatasetValidator:
    """Validate dataset structure and content."""

    def __init__(self, base_path: Path):
        """Initialize validator with base path."""
        self._base_path = base_path

    def validate_directories(self, required_dirs: List[str]) -> None:
        """Validate required directories exist."""
        missing = self._find_missing_directories(required_dirs)
        if missing:
            raise Exception(f"Missing directories: {', '.join(missing)}")

    def validate_file_exists(self, relative_path: str) -> None:
        """Validate specific file exists."""
        full_path = self._base_path / relative_path
        if not full_path.exists():
            raise Exception(f"File not found: {relative_path}")

    def _find_missing_directories(self, required: List[str]) -> List[str]:
        """Find missing required directories."""
        missing = []
        for dir_path in required:
            full_path = self._base_path / dir_path
            if not full_path.exists():
                missing.append(str(full_path))
        return missing

    def count_split_files(self, split: str, subdirs: List[str]) -> Dict[str, int]:
        """Count files in split subdirectories."""
        counts = {}
        for subdir in subdirs:
            path = self._base_path / split / subdir
            if path.exists():
                counts[subdir] = len(list(path.glob("*")))
        return counts
