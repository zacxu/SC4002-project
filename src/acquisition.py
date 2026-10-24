"""Acquire the original MASSIVE 1.0 English file without extracting other locales.

SHA-256 values record the bytes observed locally. Amazon does not publish an
expected digest here, so these values are not an independent authenticity check.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen

from src.runtime import project_root, relative_project_path


SOURCE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
ARCHIVE_PATH = "data/amazon-massive-dataset-1.0.tar.gz"
ENGLISH_PATH = "data/1.0/data/en-US.jsonl"
ENGLISH_MEMBER = "1.0/data/en-US.jsonl"
MANIFEST_PATH = "data/source_manifest.json"
CHUNK_BYTES = 1024 * 1024
MAX_ENGLISH_BYTES = 128 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".part", dir=parent)
    os.close(descriptor)
    return Path(name)


def _require_regular_local_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular local file: {path}")


def download_archive(destination: Path) -> dict[str, Any]:
    """Stream the official URL into a temporary file; never replace an archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Archive already exists; it will be reused: {destination}")
    temporary = _temporary_path(destination.parent, ".massive-download-")
    try:
        request = Request(SOURCE_URL, headers={"User-Agent": "SC4002-MASSIVE-acquisition/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            headers = {key: response.headers.get(key) for key in (
                "Content-Length", "Content-Type", "ETag", "Last-Modified"
            ) if response.headers.get(key) is not None}
            final_url = response.geturl()
            shutil.copyfileobj(response, output, length=CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        expected_size = headers.get("Content-Length")
        if expected_size is not None and temporary.stat().st_size != int(expected_size):
            raise ValueError("Downloaded archive size differs from HTTP Content-Length")
        if temporary.stat().st_size == 0:
            raise ValueError("The downloaded archive is empty")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Archive appeared during download: {destination}")
        temporary.replace(destination)
        return {
            "url": SOURCE_URL,
            "final_url": final_url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_headers": headers,
            "provenance": "Downloaded by this acquisition script over HTTPS.",
        }
    finally:
        temporary.unlink(missing_ok=True)


def _validate_member(member: tarfile.TarInfo) -> None:
    # No archive-controlled filesystem operation is ever performed. Reject
    # malicious names and links even for members that would otherwise be skipped.
    name = member.name
    pieces = PurePosixPath(name).parts
    if (not name or name.startswith("/") or "\\" in name or ".." in pieces
            or any(":" in piece for piece in pieces)):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"Archive links and special members are forbidden: {name!r}")


def extract_english(archive_path: Path, destination: Path) -> dict[str, Any]:
    """Inspect every tar member, then publish only the exact English regular file.

    Iteration uses streaming mode. No extract/extractall call or archive-supplied
    output path is used. Duplicates, links, traversal names, missing English, and
    conflicting existing English files fail without replacing existing data.
    """
    _require_regular_local_file(archive_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _require_regular_local_file(destination)
    temporary = _temporary_path(destination.parent, ".english-extraction-")
    found = False
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                _validate_member(member)
                if member.name != ENGLISH_MEMBER:
                    continue
                if found:
                    raise ValueError(f"Duplicate English member: {ENGLISH_MEMBER}")
                if not member.isfile():
                    raise ValueError("The English archive member must be a regular file")
                if not 0 < member.size <= MAX_ENGLISH_BYTES:
                    raise ValueError("English archive member size is outside the accepted range")
                found = True
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Cannot read the English archive member")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_BYTES)
                if temporary.stat().st_size != member.size:
                    raise ValueError("English extraction size differs from the tar header")
        if not found:
            raise ValueError(f"Required English member is missing: {ENGLISH_MEMBER}")
        digest = file_sha256(temporary)
        if destination.exists() or destination.is_symlink():
            _require_regular_local_file(destination)
            if file_sha256(destination) != digest:
                raise ValueError("Existing English data differs from the archive; refusing to overwrite")
        else:
            temporary.replace(destination)
        return {"relative_path": ENGLISH_PATH, "archive_member": ENGLISH_MEMBER,
                "sha256": digest, "size_bytes": destination.stat().st_size}
    finally:
        temporary.unlink(missing_ok=True)


def _verify_record(path: Path, record: dict[str, Any], expected_relative_path: str) -> None:
    if record["relative_path"] != expected_relative_path:
        raise ValueError("Source manifest contains an unexpected destination path")
    _require_regular_local_file(path)
    if file_sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Existing data differs from the source manifest: {path}")


def acquire_dataset(explicit_root: str | Path | None = None) -> dict[str, Any]:
    """Download once, verify prior recorded bytes, and retain original English data."""
    root = project_root(explicit_root)
    archive_path = relative_project_path(root, ARCHIVE_PATH)
    english_path = relative_project_path(root, ENGLISH_PATH)
    manifest_path = relative_project_path(root, MANIFEST_PATH)
    previous = None
    if manifest_path.exists() or manifest_path.is_symlink():
        _require_regular_local_file(manifest_path)
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["schema_version"] != 1 or previous["source"]["url"] != SOURCE_URL:
            raise ValueError("Unexpected source manifest version or URL")
        _verify_record(archive_path, previous["archive"], ARCHIVE_PATH)
        if english_path.exists() or english_path.is_symlink():
            _verify_record(english_path, previous["english"], ENGLISH_PATH)
    if archive_path.exists() or archive_path.is_symlink():
        _require_regular_local_file(archive_path)
        source = previous["source"] if previous is not None else {
            "url": SOURCE_URL, "final_url": None, "downloaded_at_utc": None,
            "http_headers": {},
            "provenance": "Pre-existing local archive; acquisition time and HTTP headers were not observed.",
        }
    else:
        source = download_archive(archive_path)
    archive_record = {"relative_path": ARCHIVE_PATH, "sha256": file_sha256(archive_path),
                      "size_bytes": archive_path.stat().st_size}
    english_record = extract_english(archive_path, english_path)
    if previous is not None and english_record != previous["english"]:
        raise ValueError("Extracted English metadata differs from the previous manifest")
    manifest = {
        "schema_version": 1,
        "source": source,
        "archive": archive_record,
        "english": english_record,
        "checksum_scope": "Locally observed SHA-256 digests for reproducibility and change detection; no independently published expected checksum was available.",
    }
    if previous is None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(manifest_path.parent, ".source-manifest-")
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            if manifest_path.exists() or manifest_path.is_symlink():
                raise FileExistsError(f"Manifest appeared during acquisition: {manifest_path}")
            temporary.replace(manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest
