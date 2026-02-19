from __future__ import annotations

import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download


def ensure_vector_db(
    repo_id: str = "nzeinali/DMP_vector_db",
    filename: str = "vector_db.zip",
    dest_root: str | Path = "data/vector_db",
) -> Path:
    """
    Download vector_db.zip from Hugging Face and extract into data/vector_db
    only if index files are missing.
    """

    dest_root = Path(dest_root).resolve()

    # Check if already exists
    if any(dest_root.glob("*/index.faiss")):
        print("Vector DB already exists locally. Skipping download.")
        return dest_root

    print("Downloading vector DB from Hugging Face...")

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
    )

    dest_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_root)

    print("Vector DB downloaded and extracted.")
    return dest_root