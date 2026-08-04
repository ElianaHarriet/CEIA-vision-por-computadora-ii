"""Client for RunPod Serverless Endpoints."""
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from airflow.exceptions import AirflowException
from airflow.models import Variable

RUNPOD_API_BASE = "https://api.runpod.ai/v2"

TERMINAL_FAILURE_STATUSES = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodClient:
    """Client to submit and poll jobs on a RunPod Serverless Endpoint."""

    def __init__(self, endpoint_id: str):
        """Initialize client for a given endpoint."""
        self.endpoint_id = endpoint_id
        self._api_key = None
        self._session = self._create_session_with_retries()

    def _create_session_with_retries(self):
        """Create requests session with automatic retry logic."""
        retry_strategy = Retry(
            total=3,                             # Máximo 3 reintentos
            backoff_factor=2,                    # Espera 2, 4, 8 segundos
            status_forcelist=[429, 500, 502, 503, 504],  # Errores HTTP a reintentar
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_api_key(self):
        """Read RUNPOD_API_KEY from Airflow's secrets backend."""
        if self._api_key is None:
            self._api_key = Variable.get("RUNPOD_API_KEY")
        return self._api_key

    def _headers(self):
        """Build auth headers for RunPod API requests."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    def submit_job(self, payload: dict) -> str:
        """Submit a job to the endpoint and return its job_id."""
        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/run"
        response = self._session.post(
            url, json={"input": payload}, headers=self._headers(), timeout=30
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"✓ RunPod job submitted: {job_id}")
        return job_id

    def get_status(self, job_id: str) -> dict:
        """Get the current status of a job."""
        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/status/{job_id}"
        response = self._session.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    def poll_job(self, job_id: str, timeout_s: int = 10800, interval_s: int = 45) -> dict:
        """Poll a job until it reaches a terminal state.

        Raises AirflowException on FAILED/CANCELLED/TIMED_OUT or if the
        polling loop itself exceeds timeout_s.
        """
        elapsed = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while elapsed < timeout_s:
            try:
                status_data = self.get_status(job_id)
                consecutive_errors = 0  # Reset error counter on success
                
                status = status_data.get("status")
                print(f"RunPod job {job_id} status: {status} (elapsed: {elapsed}s)")
                
                if status == "COMPLETED":
                    return status_data.get("output", {})
                if status in TERMINAL_FAILURE_STATUSES:
                    raise AirflowException(
                        f"RunPod job {job_id} ended with status {status}: "
                        f"{status_data.get('error')}"
                    )
            except requests.exceptions.RequestException as e:
                consecutive_errors += 1
                print(f"⚠️  Network error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    raise AirflowException(
                        f"RunPod job {job_id} polling failed after {max_consecutive_errors} "
                        f"consecutive network errors: {e}"
                    )
                # Wait a bit before retrying
                time.sleep(10)
                continue
            
            time.sleep(interval_s)
            elapsed += interval_s
            
        raise AirflowException(
            f"RunPod job {job_id} did not complete within {timeout_s}s"
        )
