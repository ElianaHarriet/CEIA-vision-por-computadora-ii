"""RunPod profile configuration."""
import os


class RunPodProfileConfig:
    """Read RunPod configuration based on active profile."""

    @staticmethod
    def get_active_profile() -> str:
        """Get active profile name."""
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
