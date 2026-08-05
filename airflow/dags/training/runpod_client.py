"""Client for RunPod Serverless Endpoints."""
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from airflow.exceptions import AirflowException
from airflow.models import Variable

RUNPOD_API_BASE = "https://api.runpod.ai/v2"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

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
            from training.profile_config import RunPodProfileConfig
            profile = RunPodProfileConfig.get_active_profile()
            var_name = f"RUNPOD_API_KEY_{profile.upper()}"
            try:
                self._api_key = Variable.get(var_name)
            except KeyError:
                # Fallback to generic RUNPOD_API_KEY
                self._api_key = Variable.get("RUNPOD_API_KEY")
        return self._api_key

    def _headers(self):
        """Build auth headers for RunPod API requests."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    def get_balance(self) -> dict:
        """Query the account credit balance via the GraphQL ``myself`` query.

        ``clientBalance`` is the current prepaid credit balance in USD;
        ``currentSpendPerHr`` is the instantaneous spend rate. This is the
        same query the ``runpodctl user`` command uses.
        """
        query = """
            query myself {
                myself {
                    clientBalance
                    currentSpendPerHr
                    creditAlertThreshold
                }
            }
        """
        response = self._session.post(
            RUNPOD_GRAPHQL_URL,
            json={"query": query},
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(
                f"RunPod balance query failed: {data['errors']}"
            )
        return data["data"]["myself"]

    def check_balance(self, min_balance: float = None) -> float:
        """Fail early if the RunPod credit balance is too low.

        Returns the current credit balance. Threshold defaults to
        ``RUNPOD_MIN_BALANCE`` (2 USD) and can be overridden per call.
        """
        threshold = min_balance if min_balance is not None else float(
            os.getenv("RUNPOD_MIN_BALANCE", "2")
        )
        balance = self.get_balance()
        client_balance = float(balance.get("clientBalance", 0))
        print(f"RunPod balance: ${client_balance:.2f} "
              f"(threshold: ${threshold:.2f})")
        if client_balance < threshold:
            raise ValueError(
                f"RunPod balance insuficiente: ${client_balance:.2f} "
                f"< ${threshold:.2f}. No se dispara el training."
            )
        return client_balance

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

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job so it stops billing."""
        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/cancel/{job_id}"
        response = self._session.post(
            url, headers=self._headers(), timeout=30
        )
        response.raise_for_status()
        print(f"✓ RunPod job {job_id} cancelled")
        return response.json()

    def poll_job(self, job_id: str, timeout_s: int = 10800, interval_s: int = 45) -> dict:
        """Poll a job until it reaches a terminal state.

        Raises AirflowException on FAILED/CANCELLED/TIMED_OUT or if the
        polling loop itself exceeds timeout_s. The job is cancelled before
        raising, so a timed-out run stops billing immediately.
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
                    # Cancel before raising to stop billing
                    self._cancel_best_effort(job_id, status, status_data)
                    raise AirflowException(
                        f"RunPod job {job_id} ended with status {status}: "
                        f"{status_data.get('error')}"
                    )
            except requests.exceptions.RequestException as e:
                consecutive_errors += 1
                print(f"⚠️  Network error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    # Cancel before raising to stop billing
                    self._cancel_best_effort(job_id, "NETWORK_ERROR", {})
                    raise AirflowException(
                        f"RunPod job {job_id} polling failed after {max_consecutive_errors} "
                        f"consecutive network errors: {e}"
                    )
                # Wait a bit before retrying
                time.sleep(10)
                continue
            
            time.sleep(interval_s)
            elapsed += interval_s
        
        # Timeout reached - cancel before raising to stop billing
        self._cancel_best_effort(job_id, "TIMEOUT", {})
        raise AirflowException(
            f"RunPod job {job_id} did not complete within {timeout_s}s"
        )

    def _cancel_best_effort(self, job_id: str, status: str, status_data: dict):
        """Cancel a non-completed job, swallowing API errors.

        A job stuck in IN_QUEUE/IN_PROGRESS that hit the poll timeout would
        otherwise keep billing until RunPod's own timeout kicks in. Retrying
        the task would then submit a second job, doubling cost.
        """
        try:
            self.cancel_job(job_id)
        except Exception as exc:
            print(
                f"⚠ Could not cancel RunPod job {job_id} "
                f"({status}): {exc}"
            )
