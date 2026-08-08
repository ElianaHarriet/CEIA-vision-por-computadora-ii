"""Download a dataset archive from MinIO exposed via a Cloudflare Quick Tunnel."""
import os
import tarfile
import tempfile
import time
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError, ClientError

# Cloudflare Quick Tunnels reject single requests/responses above ~100MB
# (observed: a single GetObject on a 164MB file failed with HTTP 530).
# Forcing boto3 to fetch the archive in ranged chunks keeps each request
# well under that limit while still avoiding one-request-per-file.
_DOWNLOAD_CONFIG = TransferConfig(multipart_threshold=20 * 1024 * 1024, multipart_chunksize=20 * 1024 * 1024)


def _get_s3_client(endpoint_url: str):
    """Build a boto3 S3 client for the public tunnel endpoint.

    signature_version='s3v4' is required: SigV2 (boto3's default) fails
    authentication against requests that Cloudflare has rewritten headers for.
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            # Explicit timeouts: a stalled Cloudflare tunnel connection can
            # trickle bytes slowly enough to never trip botocore's default
            # read timeout, hanging a ranged chunk (and the whole job) far
            # longer than any single request should take. Fail fast instead
            # so a bad chunk gets retried rather than hanging indefinitely.
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def download_dataset(endpoint_url: str, bucket: str, prefix: str, local_dir: str) -> int:
    """Download s3://bucket/<prefix>.tar.gz and extract it into local_dir.

    Returns the number of files extracted.
    
    Retries on DNS/connection errors with exponential backoff to handle
    Cloudflare tunnel DNS propagation delays.
    """
    key = f"{prefix.rstrip('/')}.tar.gz"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Retry configuration for DNS/connection errors
    max_retries = 5
    base_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            client = _get_s3_client(endpoint_url)
            
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
                print(f"Downloading s3://{bucket}/{key} (attempt {attempt + 1}/{max_retries})...")
                client.download_file(bucket, key, tmp.name, Config=_DOWNLOAD_CONFIG)
                
                with tarfile.open(tmp.name, "r:gz") as tar:
                    tar.extractall(local_dir, filter="data")
                    count = sum(1 for m in tar.getmembers() if m.isfile())

            print(f"✓ Downloaded and extracted {count} files from s3://{bucket}/{key} to {local_dir}")
            return count
            
        except (EndpointConnectionError, ClientError, OSError) as e:
            error_msg = str(e)
            
            # Check if it's a DNS/connection error
            is_dns_error = (
                "Name or service not known" in error_msg or
                "gaierror" in error_msg or
                "Could not connect" in error_msg or
                isinstance(e, EndpointConnectionError)
            )
            
            if is_dns_error and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 10, 20, 40, 80s
                print(
                    f"⚠️  Connection/DNS error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                print(
                    f"   This often happens when Cloudflare tunnel DNS hasn't propagated yet."
                )
                print(f"   Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                # Re-raise if it's not a DNS error or we're out of retries
                print(f"✗ Failed after {attempt + 1} attempts: {e}")
                raise
