import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

# disable HF default progress bars to control CLI output formatting
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download


def parse_hf_link(link: str) -> Dict[str, Any]:
    """
    parse a hugging face link or repo shorthand to extract model information.
    supports:
    - https://huggingface.co/owner/repo/resolve/branch/path/to/file.gguf
    - https://huggingface.co/owner/repo
    - owner/repo
    - owner/repo/file.gguf
    """
    cleaned: str = link.strip().rstrip("/")
    result: Dict[str, Any] = {
        "repo_id": "",
        "filename": None,
        "is_gguf": False,
        "revision": "main",
    }

    if cleaned.startswith(("http://", "https://")):
        parsed = urlparse(cleaned)
        path_parts = [p for p in parsed.path.split("/") if p]
        
        # expect huggingface.co/owner/repo or huggingface.co/owner/repo/...
        if len(path_parts) >= 2:
            # check for keyword like resolve, blob, or raw
            keywords = {"resolve", "blob", "raw"}
            keyword_idx = -1
            for i, part in enumerate(path_parts):
                if part in keywords:
                    keyword_idx = i
                    break

            if keyword_idx != -1:
                # repo is everything before resolve/blob/raw
                result["repo_id"] = "/".join(path_parts[:keyword_idx])
                # revision is the element immediately following resolve/blob/raw
                if keyword_idx + 1 < len(path_parts):
                    result["revision"] = path_parts[keyword_idx + 1]
                # filename is everything after revision
                if keyword_idx + 2 < len(path_parts):
                    result["filename"] = "/".join(path_parts[keyword_idx + 2:])
            else:
                # no file specifier, just a repo ID
                result["repo_id"] = "/".join(path_parts[:2])
        elif len(path_parts) == 1:
            # single word repo like 'gpt2'
            result["repo_id"] = path_parts[0]
    else:
        # shorthand format
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 3 and parts[-1].endswith(".gguf"):
            result["repo_id"] = "/".join(parts[:-1])
            result["filename"] = parts[-1]
        elif len(parts) == 2:
            result["repo_id"] = "/".join(parts)
        elif len(parts) == 1:
            result["repo_id"] = parts[0]
        else:
            raise ValueError(f"invalid hugging face identifier format: {link}")

    if not result["repo_id"]:
        raise ValueError(f"could not parse repository id from: {link}")

    # determine if gguf
    if result["filename"] and result["filename"].endswith(".gguf"):
        result["is_gguf"] = True

    return result


def download_model(
    link: str,
    dest_dir: Path,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Path:
    """
    downloads a model based on hugging face link to the destination directory.
    handles both GGUF files and complete Transformers repositories.
    """
    parsed = parse_hf_link(link)
    repo_id: str = parsed["repo_id"]
    filename: Optional[str] = parsed["filename"]
    is_gguf: bool = parsed["is_gguf"]
    revision: str = parsed["revision"]

    # sanitize folder name for destination (replace slashes with double underscores)
    safe_repo_name = repo_id.replace("/", "--")
    repo_dest_dir = dest_dir / safe_repo_name

    if progress_callback:
        progress_callback("starting download process")

    try:
        if is_gguf and filename:
            # downloading a specific GGUF file
            repo_dest_dir.mkdir(parents=True, exist_ok=True)
            if progress_callback:
                progress_callback(f"downloading file {filename.lower()} from repository {repo_id.lower()}")
            
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=repo_dest_dir,
                local_dir_use_symlinks=False,
            )
            if progress_callback:
                progress_callback("download completed successfully")
            return Path(downloaded_file_path)
        else:
            # downloading full transformers model snapshot
            if progress_callback:
                progress_callback(f"downloading transformers repository snapshot for {repo_id.lower()}")
            
            downloaded_dir_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=repo_dest_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"] if not filename else []
            )
            if progress_callback:
                progress_callback("download completed successfully")
            return Path(downloaded_dir_path)
            
    except Exception as err:
        if progress_callback:
            progress_callback(f"download failed: {str(err).lower()}")
        raise RuntimeError(f"failed to download from {link}: {err}") from err
