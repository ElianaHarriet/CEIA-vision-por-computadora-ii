"""RunPod profile configuration."""
import os
from pathlib import Path


class RunPodProfileConfig:
    """Read RunPod configuration based on active profile."""

    @staticmethod
    def get_active_profile() -> str:
        """Get active profile name from config/.runpod-profile or env var."""
        # Primero intenta leer de airflow/config/.runpod-profile (versionado en git)
        profile_file = Path(__file__).parent.parent.parent / "config" / ".runpod-profile"
        if profile_file.exists():
            for line in profile_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "RUNPOD_PROFILE":
                        return value.strip()
        # Fallback a variable de entorno (para compatibilidad)
        return os.getenv("RUNPOD_PROFILE", "santiago")

    @staticmethod
    def get_endpoint_id() -> str:
        """Get RunPod endpoint ID for active profile."""
        profile = RunPodProfileConfig.get_active_profile()
        var_name = f"RUNPOD_ENDPOINT_ID_{profile.upper()}"
        endpoint_id = os.getenv(var_name)
        if not endpoint_id:
            fallback = os.getenv("RUNPOD_ENDPOINT_ID")
            if fallback:
                return fallback
            raise ValueError(
                f"No endpoint ID found for profile '{profile}' "
                f"(expected {var_name} in .env)"
            )
        return endpoint_id

    @staticmethod
    def print_active_config():
        """Print active configuration."""
        profile = RunPodProfileConfig.get_active_profile()
        endpoint = RunPodProfileConfig.get_endpoint_id()
        print(f"🎯 RunPod Profile: {profile}")
        print(f"   Endpoint ID: {endpoint}")
