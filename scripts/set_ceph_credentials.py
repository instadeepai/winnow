"""Verify Ceph / S3-compatible credentials are available from the environment.

Do not write secrets to ``~/.aws`` — that duplicates credentials on disk (broader
exposure, backups, shared filesystems). On clusters where
``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` are injected into the job
environment, boto3, the AWS CLI, s3fs, and most S3 clients use them via the
default credential provider chain with no extra files.

Optional: set ``AWS_ENDPOINT_URL`` (or legacy ``AWS_ENDPOINT_URL_S3``) for a
custom S3 endpoint (typical for Ceph RGW).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

_REQUIRED = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
_ENDPOINT_VARS = "AWS_ENDPOINT_URL"


def check_credentials() -> None:
    """Ensure required AWS-compatible env vars are set; print a short summary.

    Raises:
        OSError: if required environment variables are not set
    """
    missing = [name for name in _REQUIRED if not os.environ.get(name)]
    if missing:
        msg = (
            "Ceph/S3 access needs these environment variables set (e.g. by your "
            "scheduler or shell): " + ", ".join(_REQUIRED)
        )
        raise OSError(msg)

    endpoint_set = next((v for v in _ENDPOINT_VARS if os.environ.get(v)), None)
    print("Ceph/S3 credentials: OK")
    if endpoint_set:
        print(f"Custom endpoint: {endpoint_set} is set.")
    else:
        print(
            "Note: no AWS_ENDPOINT_URL set — use that (or AWS_ENDPOINT_URL_S3) if you use a non-AWS S3 endpoint."
        )


if __name__ == "__main__":
    try:
        check_credentials()
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
