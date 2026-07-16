from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import (
    ArchiveConflictError,
    ChecksumMismatchError,
    RawArchive,
)

OBSERVED_AT = datetime(2026, 7, 16, 1, 30, tzinfo=UTC)


class DataLayoutTests(unittest.TestCase):
    def test_ensure_creates_all_data_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            layout = DataLayout(Path(temporary_directory)).ensure()

            self.assertTrue(layout.raw.is_dir())
            self.assertTrue(layout.canonical.is_dir())
            self.assertTrue(layout.derived.is_dir())


class RawArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.layout = DataLayout(Path(self.temporary_directory.name))
        self.archive = RawArchive(self.layout)
        self.content = b'{"fixture": "arsenal-chelsea"}'

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def archive_content(self, **overrides: object):
        arguments = {
            "source": "fbref",
            "source_id": "match-123",
            "url": "https://fbref.com/en/matches/match-123",
            "observed_at": OBSERVED_AT,
            "target_event_time": OBSERVED_AT + timedelta(days=1),
            "collector_version": "fbref-collector/0.1.0",
            "media_type": "text/html",
        }
        arguments.update(overrides)
        return self.archive.archive(self.content, **arguments)  # type: ignore[arg-type]

    def test_archive_is_idempotent_and_manifest_contains_provenance(self) -> None:
        first = self.archive_content()
        second = self.archive_content()

        self.assertEqual(first, second)
        self.assertEqual(self.archive.read(first.id), self.content)
        self.archive.verify(first)

        manifest_path = self.layout.raw_manifest_path(first.id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "source",
            "source_id",
            "url",
            "observed_at",
            "target_event_time",
            "checksum",
            "collector_version",
            "media_type",
        ):
            self.assertIn(field, manifest)
        self.assertEqual(manifest["observed_at"], "2026-07-16T01:30:00Z")

    def test_same_content_deduplicates_bytes_without_losing_observations(self) -> None:
        first = self.archive_content()
        second = self.archive_content(observed_at=OBSERVED_AT + timedelta(minutes=5))

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(
            self.layout.raw_object_path(first.checksum),
            self.layout.raw_object_path(second.checksum),
        )
        object_files = [path for path in (self.layout.raw / "objects").rglob("*") if path.is_file()]
        self.assertEqual(len(object_files), 1)

    def test_expected_and_stored_checksums_are_verified(self) -> None:
        with self.assertRaises(ChecksumMismatchError):
            self.archive_content(expected_checksum="0" * 64)

        asset = self.archive_content()
        object_path = self.layout.raw_object_path(asset.checksum)
        object_path.write_bytes(b"modified")

        with self.assertRaises(ChecksumMismatchError):
            self.archive.read(asset)

    def test_checksum_is_sha256_of_content(self) -> None:
        asset = self.archive_content()

        self.assertEqual(asset.checksum, hashlib.sha256(self.content).hexdigest())

    def test_read_rejects_a_modified_manifest(self) -> None:
        asset = self.archive_content()
        manifest_path = self.layout.raw_manifest_path(asset.id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["url"] = "https://example.invalid/tampered"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(ArchiveConflictError):
            self.archive.read(asset)


if __name__ == "__main__":
    unittest.main()
