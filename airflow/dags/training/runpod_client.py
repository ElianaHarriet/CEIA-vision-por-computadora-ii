"""Client for RunPod Serverless Endpoints."""
import os
import time
import requests
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
        response = requests.post(
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
        response = requests.post(
            url, json={"input": payload}, headers=self._headers(), timeout=30
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"✓ RunPod job submitted: {job_id}")
        return job_id

    def get_status(self, job_id: str, timeout_s: float = 60) -> dict:
        """Get the current status of a job, retrying transient API errors."""
        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/status/{job_id}"
        for attempt in range(3):
            try:
                response = requests.get(
                    url, headers=self._headers(), timeout=timeout_s
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                print(f"⚠ status poll attempt {attempt+1} failed: {exc}")
                time.sleep(10 * (attempt + 1))

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job so it stops billing."""
        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/cancel/{job_id}"
        response = requests.post(
            url, headers=self._headers(), timeout=30
        )
        response.raise_for_status()
        print(f"✗ RunPod job {job_id} cancelled")
        return response.json()

    def poll_job(self, job_id: str, timeout_s: int = 10800, interval_s: int = 45) -> dict:
        """Poll a job until it reaches a terminal state.

        Raises AirflowException on FAILED/CANCELLED/TIMED_OUT or if the
        polling loop itself exceeds timeout_s. The job is cancelled before
        raising, so a timed-out run stops billing immediately.
        """
        elapsed = 0
        while elapsed < timeout_s:
            status_data = self.get_status(job_id)
            status = status_data.get("status")
            print(f"RunPod job {job_id} status: {status} (elapsed: {elapsed}s)")
            if status == "COMPLETED":
                return status_data.get("output", {})
            if status in TERMINAL_FAILURE_STATUSES:
                self._cancel_best_effort(job_id, status, status_data)
                raise AirflowException(
                    f"RunPod job {job_id} ended with status {status}: "
                    f"{status_data.get('error')}"
                )
            time.sleep(interval_s)
            elapsed += interval_s
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
