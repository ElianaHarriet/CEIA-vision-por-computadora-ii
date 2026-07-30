"""
File system operations.
Single Responsibility: Handle file/directory operations.
"""
import shutil
from pathlib import Path
from typing import List


class FileSystemOperations:
    """Encapsulate file system operations."""

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def remove_directory(path: Path) -> None:
        """Remove directory and all contents."""
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def copy_file(source: Path, destination: Path) -> None:
        """Copy file from source to destination."""
        shutil.copy2(source, destination)

    @staticmethod
    def count_files(directory: Path, pattern: str = "*") -> int:
        """Count files matching pattern in directory."""
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))

    @staticmethod
    def list_images(directory: Path) -> List[Path]:
        """List all image files in directory."""
        if not directory.exists():
            return []
        jpg_files = list(directory.glob("*.jpg"))
        png_files = list(directory.glob("*.png"))
        return jpg_files + png_files

    @staticmethod
    def copy_directory(source: Path, destination: Path) -> None:
        """Copy entire directory tree."""
        shutil.copytree(source, destination, dirs_exist_ok=True)
