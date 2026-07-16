"""Filesystem layout and immutable raw evidence storage."""

from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import (
    ArchiveConflictError,
    ChecksumMismatchError,
    RawArchive,
)

__all__ = [
    "ArchiveConflictError",
    "ChecksumMismatchError",
    "DataLayout",
    "RawArchive",
]
