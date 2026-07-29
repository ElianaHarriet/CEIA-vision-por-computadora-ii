"""Upload local dataset directories to MinIO (S3-compatible) storage."""
import os
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


def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str) -> int:
    """Recursively upload local_dir to s3://bucket/prefix. Returns file count."""
    client = _get_s3_client()
    local_path = Path(local_dir)
    files = [p for p in local_path.rglob("*") if p.is_file()]
    for file_path in files:
        relative_key = file_path.relative_to(local_path).as_posix()
        key = f"{prefix.rstrip('/')}/{relative_key}"
        client.upload_file(str(file_path), bucket, key)
    print(f"✓ Uploaded {len(files)} files from {local_dir} to s3://{bucket}/{prefix}")
    return len(files)
