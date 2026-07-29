"""Download a dataset archive from MinIO exposed via a Cloudflare Quick Tunnel."""
import os
import tarfile
import tempfile
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

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
        config=Config(signature_version="s3v4"),
    )


def download_dataset(endpoint_url: str, bucket: str, prefix: str, local_dir: str) -> int:
    """Download s3://bucket/<prefix>.tar.gz and extract it into local_dir.

    Returns the number of files extracted.
    """
    client = _get_s3_client(endpoint_url)
    key = f"{prefix.rstrip('/')}.tar.gz"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        client.download_file(bucket, key, tmp.name, Config=_DOWNLOAD_CONFIG)
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(local_dir, filter="data")
            count = sum(1 for m in tar.getmembers() if m.isfile())

    print(f"✓ Downloaded and extracted {count} files from s3://{bucket}/{key} to {local_dir}")
    return count
