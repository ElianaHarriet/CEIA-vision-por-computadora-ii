"""Upload local dataset directories to MinIO (S3-compatible) storage.

Uploads a single tar.gz archive instead of one S3 object per file: with
~4600 files, per-file HTTP round-trips through the Cloudflare tunnel took
~1.5h even though MinIO itself answered each request in ~1ms. One archive
turns thousands of round-trips into one.
"""
import os
import tarfile
import tempfile
from pathlib import Path
import boto3


def _get_s3_client():
    """Build a boto3 S3 client pointed at the internal MinIO endpoint."""
    endpoint = os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def archive_key(prefix: str) -> str:
    """S3 key for the dataset archive of a given prefix."""
    return f"{prefix.rstrip('/')}.tar.gz"


def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str, force: bool = False) -> int:
    """Tar+gzip local_dir and upload it as s3://bucket/<prefix>.tar.gz.

    Skips upload if the archive already exists in S3 unless force=True.
    
    Returns the number of files packed into the archive, or 0 if skipped.
    """
    client = _get_s3_client()
    local_path = Path(local_dir)
    files = [p for p in local_path.rglob("*") if p.is_file()]
    key = archive_key(prefix)
    
    # Check if archive already exists in S3
    if not force:
        try:
            client.head_object(Bucket=bucket, Key=key)
            print(f"⚠️  Archive s3://{bucket}/{key} already exists, skipping upload")
            print(f"   (use force=True to overwrite)")
            return 0
        except client.exceptions.ClientError as e:
            # 404 means object doesn't exist, proceed with upload
            if e.response['Error']['Code'] != '404':
                raise

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            for file_path in files:
                tar.add(file_path, arcname=file_path.relative_to(local_path).as_posix())
        client.upload_file(tmp.name, bucket, key)

    print(f"✓ Uploaded {len(files)} files from {local_dir} to s3://{bucket}/{key}")
    return len(files)
