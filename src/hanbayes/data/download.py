"""Dataset download helper.

The ChnSentiCorp dataset is NOT redistributed with this repository. The
canonical community copy used by the original experiments is hosted in the
``chnsenticorp`` GitHub repository (Hugging Face datasets mirror). We download
the three TSV splits and verify their SHA-256 checksums so that every user
reproduces the exact same data.

You may also place ``train.tsv`` / ``dev.tsv`` / ``test.tsv`` manually into
``data/raw/`` (header: ``label<TAB>text_a``) and skip downloading.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE_URL = "https://raw.githubusercontent.com/pengming617/chnsenticorp/main/data"

CHECKSUMS = {
    "train.tsv": "b85b8318588fcf68f58589a923a09b5bf09000e8f066388bb2a92dab7f3ba787",
    "dev.tsv": "1ecfbc62abe99b7170bdeacd64490febf150bc5a583701e7d9e72900f775059c",
    "test.tsv": "f47c7d28989b1de30634729327f8f90c9a203d337136ef982c9c14cc9ef33392",
}

_DOWNLOAD_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 3
_USER_AGENT = "hanbayes/2.0 (+https://github.com/GOOD-123-CPU/hanbayes)"


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Return True when the file's SHA-256 matches *expected_sha256*."""

    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_sha256


def _fetch(url: str, target: Path) -> None:
    """Download *url* to *target* with timeout, UA header and retries."""

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response, open(target, "wb") as file:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    file.write(chunk)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)  # exponential back-off: 2s, 4s

    raise RuntimeError(
        f"下载失败（已重试 {_MAX_RETRIES} 次）：{url}\n"
        f"最后一次错误：{last_error}\n"
        "请检查网络，或手动下载后放入 data/raw/ 目录。"
    )


def download_dataset(data_dir, show_progress: bool = True) -> Path:
    """Download the three TSV splits into *data_dir* and verify checksums.

    If all files already exist with correct checksums, nothing is downloaded.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, checksum in CHECKSUMS.items():
        target = data_dir / filename

        if target.exists() and verify_checksum(target, checksum):
            if show_progress:
                print(f"已存在且校验通过：{target}")
            continue

        url = f"{RAW_BASE_URL}/{filename}"
        if show_progress:
            print(f"下载 {url} -> {target}")
        _fetch(url, target)

        if not verify_checksum(target, checksum):
            target.unlink(missing_ok=True)  # don't leave corrupt files behind
            raise RuntimeError(
                f"下载文件校验失败：{filename}。"
                "已删除损坏文件。请检查网络或手动下载后放入 data/raw/ 目录。"
            )
        if show_progress:
            print(f"校验通过：{filename}")

    return data_dir
