"""Set Ceph credentials for S3-compatible storage."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def set_credentials() -> None:
    """Set the Ceph credentials.

    - To access Ceph storage, the credentials are stored in AWS config and credentials files
    - The credentials are read from environment variables FOUNDATION_MODEL_AWS_ACCESS_KEY_ID
      and FOUNDATION_MODEL_AWS_SECRET_ACCESS_KEY

    Raises:
        OSError: if required environment variables are not set
    """
    try:
        access_key = os.environ["FOUNDATION_MODEL_AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["FOUNDATION_MODEL_AWS_SECRET_ACCESS_KEY"]
    except KeyError as e:
        msg = (
            "To use Ceph storage you should set both 'FOUNDATION_MODEL_AWS_ACCESS_KEY_ID' "
            "and 'FOUNDATION_MODEL_AWS_SECRET_ACCESS_KEY' environment variables."
        )
        raise OSError(msg) from e

    # Create credentials directory if it doesn't exist
    credentials_dir = Path.home() / ".aws"
    credentials_dir.mkdir(exist_ok=True)

    # Set up config file
    config_path = credentials_dir / "config"
    config_content = """[profile foundation_model]
output = json

[foundation_model]
aws_access_key_id = {access_key}
aws_secret_access_key = {secret_key}
""".format(access_key=access_key, secret_key=secret_key)

    with open(config_path, "w") as f:
        f.write(config_content)

    # Set up credentials file
    credentials_path = credentials_dir / "credentials"
    credentials_content = """[foundation_model]
aws_access_key_id = {access_key}
aws_secret_access_key = {secret_key}
""".format(access_key=access_key, secret_key=secret_key)

    with open(credentials_path, "w") as f:
        f.write(credentials_content)


if __name__ == "__main__":
    set_credentials()
