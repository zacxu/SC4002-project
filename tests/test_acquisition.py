"""Offline acquisition tests: hostile archives, provenance, and immutable reuse."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.acquisition import (
    ARCHIVE_PATH, ENGLISH_MEMBER, ENGLISH_PATH, MANIFEST_PATH, SOURCE_URL,
    acquire_dataset, download_archive, extract_english, file_sha256,
)


ENGLISH_BYTES = b'{"id":"0","locale":"en-US","partition":"train","utt":"hello"}\n'


def make_archive(path: Path, entries: list[tuple[str, bytes | None, bytes | None]]) -> None:
    """Entries contain name, payload, and optional special tar type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, special_type in entries:
            member = tarfile.TarInfo(name)
            if special_type is not None:
                member.type = special_type
                if special_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    member.linkname = "../../outside"
            else:
                member.size = len(payload or b"")
            archive.addfile(member, io.BytesIO(payload) if payload is not None else None)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "dataset.tar.gz"
        self.destination = self.root / ENGLISH_PATH

    def test_exact_english_only_and_idempotent_existing_output(self):
        make_archive(self.archive, [
            ("1.0/data/fr-FR.jsonl", b"French data", None),
            (ENGLISH_MEMBER, ENGLISH_BYTES, None),
            ("1.0/data/en-GB.jsonl", b"Other English locale", None),
        ])
        result = extract_english(self.archive, self.destination)
        self.assertEqual(self.destination.read_bytes(), ENGLISH_BYTES)
        self.assertEqual(result["archive_member"], ENGLISH_MEMBER)
        self.assertEqual(result["sha256"], file_sha256(self.destination))
        self.assertEqual([p.name for p in self.destination.parent.iterdir()], ["en-US.jsonl"])
        original_mtime = self.destination.stat().st_mtime_ns
        self.assertEqual(extract_english(self.archive, self.destination), result)
        self.assertEqual(self.destination.stat().st_mtime_ns, original_mtime)

    def test_duplicate_exact_member_rejected_before_any_data_is_published(self):
        make_archive(self.archive, [(ENGLISH_MEMBER, ENGLISH_BYTES, None)] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate English"):
            extract_english(self.archive, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_traversal_absolute_windows_and_links_rejected_even_after_english(self):
        hostile = [
            ("../outside", b"bad", None),
            ("/outside", b"bad", None),
            ("C:/outside", b"bad", None),
            ("1.0\\data\\en-US.jsonl", b"bad", None),
            ("unused-symlink", None, tarfile.SYMTYPE),
            ("unused-hardlink", None, tarfile.LNKTYPE),
            ("unused-fifo", None, tarfile.FIFOTYPE),
        ]
        for entry in hostile:
            with self.subTest(name=entry[0]):
                make_archive(self.archive, [(ENGLISH_MEMBER, ENGLISH_BYTES, None), entry])
                with self.assertRaises(ValueError):
                    extract_english(self.archive, self.destination)
                self.assertFalse(self.destination.exists())
                self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_missing_nonexact_or_directory_english_is_rejected(self):
        for entries in (
            [("./" + ENGLISH_MEMBER, ENGLISH_BYTES, None)],
            [(ENGLISH_MEMBER, None, tarfile.DIRTYPE)],
            [("1.0/data/fr-FR.jsonl", b"French", None)],
        ):
            with self.subTest(entries=entries):
                make_archive(self.archive, entries)
                with self.assertRaises(ValueError):
                    extract_english(self.archive, self.destination)
                self.assertFalse(self.destination.exists())

    def test_existing_conflicting_data_is_preserved(self):
        make_archive(self.archive, [(ENGLISH_MEMBER, ENGLISH_BYTES, None)])
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"local modification")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            extract_english(self.archive, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"local modification")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / ARCHIVE_PATH
        make_archive(self.archive, [(ENGLISH_MEMBER, ENGLISH_BYTES, None)])

    def test_reuses_archive_without_network_and_records_unknown_original_download(self):
        with patch("src.acquisition.download_archive") as download:
            first = acquire_dataset(self.root)
            original = (self.root / MANIFEST_PATH).read_bytes()
            second = acquire_dataset(self.root)
        download.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(original, (self.root / MANIFEST_PATH).read_bytes())
        self.assertEqual(first["source"]["url"], SOURCE_URL)
        self.assertIsNone(first["source"]["downloaded_at_utc"])
        self.assertEqual(first["archive"]["sha256"], file_sha256(self.archive))
        self.assertEqual(json.loads(original)["english"]["relative_path"], ENGLISH_PATH)

    def test_modified_archive_or_english_fails_without_overwrite_or_redownload(self):
        for target in (ARCHIVE_PATH, ENGLISH_PATH):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_archive(root / ARCHIVE_PATH, [(ENGLISH_MEMBER, ENGLISH_BYTES, None)])
                acquire_dataset(root)
                original_manifest = (root / MANIFEST_PATH).read_bytes()
                (root / target).write_bytes(b"modified data")
                with patch("src.acquisition.download_archive") as download:
                    with self.assertRaisesRegex(ValueError, "differs from the source manifest"):
                        acquire_dataset(root)
                download.assert_not_called()
                self.assertEqual((root / target).read_bytes(), b"modified data")
                self.assertEqual((root / MANIFEST_PATH).read_bytes(), original_manifest)

    def test_missing_recorded_archive_fails_without_silent_redownload(self):
        acquire_dataset(self.root)
        self.archive.unlink()
        with patch("src.acquisition.download_archive") as download:
            with self.assertRaisesRegex(ValueError, "regular local file"):
                acquire_dataset(self.root)
        download.assert_not_called()

    def test_missing_english_can_be_reconstructed_from_verified_archive(self):
        manifest = acquire_dataset(self.root)
        (self.root / ENGLISH_PATH).unlink()
        self.assertEqual(acquire_dataset(self.root), manifest)
        self.assertEqual((self.root / ENGLISH_PATH).read_bytes(), ENGLISH_BYTES)


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, declared_size: int | None = None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if declared_size is None else declared_size),
                        "ETag": '"observed-etag"', "Last-Modified": "observed-date"}

    def geturl(self):
        return SOURCE_URL


class DownloadTests(unittest.TestCase):
    def test_download_records_observed_headers_and_never_replaces_existing_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive.tar.gz"
            with patch("src.acquisition.urlopen", return_value=FakeResponse(b"archive bytes")) as opener:
                source = download_archive(destination)
            self.assertEqual(destination.read_bytes(), b"archive bytes")
            self.assertEqual(source["url"], SOURCE_URL)
            self.assertEqual(source["http_headers"]["ETag"], '"observed-etag"')
            self.assertIsNotNone(source["downloaded_at_utc"])
            self.assertEqual(opener.call_args.args[0].full_url, SOURCE_URL)
            with patch("src.acquisition.urlopen") as opener:
                with self.assertRaises(FileExistsError):
                    download_archive(destination)
            opener.assert_not_called()

    def test_incomplete_download_leaves_no_archive_or_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive.tar.gz"
            with patch("src.acquisition.urlopen", return_value=FakeResponse(b"short", declared_size=999)):
                with self.assertRaisesRegex(ValueError, "Content-Length"):
                    download_archive(destination)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
