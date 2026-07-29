"""Download a dataset directory from MinIO exposed via a Cloudflare Quick Tunnel."""
import os
from pathlib import Path
import boto3
from botocore.config import Config


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
    """Recursively download s3://bucket/prefix into local_dir. Returns file count."""
    client = _get_s3_client(endpoint_url)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_key = key[len(prefix):].lstrip("/")
            if not relative_key:
                continue
            dest = Path(local_dir) / relative_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    print(f"✓ Downloaded {count} files from s3://{bucket}/{prefix} to {local_dir}")
    return count
