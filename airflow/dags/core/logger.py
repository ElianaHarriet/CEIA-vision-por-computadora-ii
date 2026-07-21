"""
Logging utilities.
Single Responsibility: Centralized logging.
"""
from typing import Dict, Any


class DatasetLogger:
    """Logger for dataset operations."""

    @staticmethod
    def log_download_start(workspace: str, project: str, version: int) -> None:
        """Log dataset download initiation."""
        print("⬇️  Descargando dataset desde Roboflow:")
        print(f"    workspace={workspace}")
        print(f"    project={project}")
        print(f"    version={version}")

    @staticmethod
    def log_download_complete(location: str) -> None:
        """Log successful download."""
        print(f"✓ Dataset descargado exitosamente en {location}")

    @staticmethod
    def log_split_stats(split: str, count: int) -> None:
        """Log split statistics."""
        print(f"  {split}: {count} imágenes")

    @staticmethod
    def log_processing_start(split: str) -> None:
        """Log split processing start."""
        print(f"\n🔄 Procesando split: {split}")

    @staticmethod
    def log_validation_success() -> None:
        """Log validation success."""
        print("\n✓ Validación completada exitosamente")

    @staticmethod
    def log_stats(stats: Dict[str, Any]) -> None:
        """Log dataset statistics."""
        for split, data in stats.items():
            print(f"{split.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
