import huggingface_hub


def download_winnow_dataset() -> None:
    """Download the Winnow MS datasets from the Hugging Face dataset repository."""
    huggingface_hub.snapshot_download(
        repo_id="instadeepai/winnow-ms-datasets",
        repo_type="dataset",
        allow_patterns=["general_*.parquet", "celegans_*.parquet", "immuno2_*.parquet"],
        local_dir="winnow-ms-datasets",
    )


if __name__ == "__main__":
    download_winnow_dataset()
