"""Per-request verification primitives for immutable local evidence.

The public storage loaders remain the authority for domain validation.  This module only
provides bounded, request-local lookup state and byte/directory proofs so those loaders do not
rescan the same catalog for every reference.  Nothing in this module is cached at module scope.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DERIVED_ARTIFACT = re.compile(r"^derived-artifact:([0-9a-f]{64})$")
_TRAINING_DATASET = re.compile(r"^training-dataset:([0-9a-f]{64})$")
_ENTITY_NAMESPACES = frozenset({"competition", "season", "team", "player", "match"})
_ACTIVE_SESSION: ContextVar[VerificationSession | None] = ContextVar(
    "football_data_platform_verification_session",
    default=None,
)


class VerificationConflict(ArchiveConflictError):
    """Raised when a read proof or immutable catalog cannot be established."""


class IdentityState(StrEnum):
    """The result of a canonical identity lookup.

    ``UNKNOWN`` is deliberately not cached: it represents an unavailable or malformed catalog,
    rather than a durable negative identity result.
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FileProof:
    """A content and metadata proof for one immutable file read."""

    path: Path
    sha256: str
    size_bytes: int
    mtime_ns: int

    @property
    def reference(self) -> str:
        return f"file-sha256:{self.sha256}"

    @classmethod
    def capture(cls, path: str | Path, *, expected_sha256: str | None = None) -> FileProof:
        resolved = Path(path).resolve()
        expected = _normalize_digest(expected_sha256)
        try:
            before = _file_signature(resolved)
            digest = _hash_file(resolved)
            after = _file_signature(resolved)
        except OSError as error:
            raise VerificationConflict(f"cannot prove file bytes: {resolved}") from error
        if before != after:
            raise VerificationConflict(f"file changed while being proved: {resolved}")
        if expected is not None and digest != expected:
            raise VerificationConflict(
                f"file checksum mismatch for {resolved}: expected {expected}, got {digest}"
            )
        return cls(
            path=resolved,
            sha256=digest,
            size_bytes=before[0],
            mtime_ns=before[1],
        )

    def verify(self) -> None:
        """Re-read and verify the file, rejecting even same-size tampering."""

        try:
            before = _file_signature(self.path)
        except OSError as error:
            raise VerificationConflict(f"proved file is missing: {self.path}") from error
        if before[0] != self.size_bytes or before[1] != self.mtime_ns:
            raise VerificationConflict(f"proved file changed: {self.path}")
        digest = _hash_file(self.path)
        if digest != self.sha256:
            raise VerificationConflict(f"proved file bytes changed: {self.path}")
        try:
            after = _file_signature(self.path)
        except OSError as error:
            raise VerificationConflict(f"proved file disappeared: {self.path}") from error
        if before != after:
            raise VerificationConflict(f"proved file changed while being checked: {self.path}")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Untrusted catalog metadata for one derived manifest file.

    The entry is not a domain manifest.  Consumers must still call the normal archive loader,
    which performs the typed manifest and lineage verifier.
    """

    artifact_id: str
    path: Path
    output_refs: tuple[str, ...]
    proof: FileProof


@dataclass(frozen=True, slots=True)
class SampleEntry:
    """Untrusted index entry connecting a sample ID to a dataset file."""

    sample_id: str
    dataset_id: str
    path: Path
    proof: FileProof


class VerificationSession:
    """Bound all verification lookup state to one public request.

    A session owns at most one read-only SQLite connection and one scan of each derived catalog.
    Successful present/absent identity answers may be reused inside the session.  Unknown
    answers and all failed scans are never cached, so a later request cannot inherit a transient
    failure or stale bytes.
    """

    def __init__(
        self,
        layout: DataLayout | str | Path,
        *,
        canonical: Any | None = None,
    ) -> None:
        self.layout = layout if isinstance(layout, DataLayout) else DataLayout(Path(layout))
        self.canonical = canonical
        self._connection: sqlite3.Connection | None = None
        self._entered = False
        self._closed = False
        self._identity_cache: dict[tuple[str, str], IdentityState] = {}
        self._manifest_catalog: dict[str, tuple[ManifestEntry, ...]] | None = None
        self._sample_catalog: dict[str, tuple[SampleEntry, ...]] | None = None
        self._proofs: dict[Path, FileProof] = {}
        self._directory_snapshots: dict[Path, tuple[str, ...]] = {}
        self._sidecar_snapshots: dict[Path, tuple[Path, ...]] = {}
        self._artifact_manifests: dict[str, Any] = {}
        self._team_observations: dict[str, Any] = {}
        self._match_results: dict[str, Any] = {}
        self._match_report_replays: dict[str, Any] = {}
        self._resolving_manifests: set[str] = set()

    def __enter__(self) -> VerificationSession:
        if self._closed:
            raise VerificationConflict("verification session is closed")
        if self._entered:
            raise VerificationConflict("verification session is already entered")
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            self.close()
        except BaseException:
            # Never mask the operation's original exception.  A clean request still receives
            # the stability failure from ``close``.
            if exc_type is None:
                raise
        return False

    def close(self) -> None:
        """Run the final stability check and close the read snapshot."""

        if self._closed:
            return
        if not self._entered:
            self._closed = True
            return
        stability_error: BaseException | None = None
        try:
            self.assert_stable()
        except BaseException as error:  # close must release SQLite even on proof failure
            stability_error = error
        finally:
            connection = self._connection
            self._connection = None
            if connection is not None:
                try:
                    connection.rollback()
                finally:
                    connection.close()
            self._closed = True
            self._entered = False
        if stability_error is not None:
            raise stability_error

    def canonical_connection(self) -> sqlite3.Connection:
        """Return the session's single read-only SQLite snapshot connection."""

        self._ensure_open()
        if self._connection is not None:
            return self._connection
        path = self._canonical_path()
        if not path.is_file():
            raise VerificationConflict(f"canonical store is unavailable: {path}")
        # Capture the database bytes before opening the transaction.  The final proof catches a
        # writer changing the file while this request is still reading its snapshot.
        self.file_proof(path)
        sidecars_before = _sqlite_sidecars(path)
        for sidecar in sidecars_before:
            self.file_proof(sidecar)
        uri = f"{path.resolve().as_uri()}?mode=ro"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(uri, uri=True, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN")
            connection.execute("SELECT name FROM sqlite_schema ORDER BY name LIMIT 1").fetchone()
        except (OSError, sqlite3.Error) as error:
            if connection is not None:
                connection.close()
            # Do not retain a failed connection or a failed identity result.
            raise VerificationConflict(f"cannot open canonical read snapshot: {path}") from error
        try:
            sidecars_after = _sqlite_sidecars(path)
        except VerificationConflict:
            connection.close()
            raise
        if sidecars_after != sidecars_before:
            connection.close()
            raise VerificationConflict(f"canonical SQLite sidecars changed while opening: {path}")
        self._connection = connection
        self._sidecar_snapshots[path.resolve()] = sidecars_before
        return connection

    def canonical_query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        if not isinstance(sql, str) or not sql.lstrip().lower().startswith("select"):
            raise ValueError("canonical_query only permits SELECT statements")
        return list(self.canonical_connection().execute(sql, parameters).fetchall())

    def lookup_identity(
        self,
        reference: str,
        *,
        namespace: str,
    ) -> IdentityState:
        """Look up a canonical ID with present/absent/unknown semantics."""

        self._ensure_open()
        if (
            namespace not in _ENTITY_NAMESPACES
            or not isinstance(reference, str)
            or not reference
            or reference.strip() != reference
            or reference.partition(":")[0] != namespace
        ):
            return IdentityState.UNKNOWN
        key = (namespace, reference)
        cached = self._identity_cache.get(key)
        if cached is not None:
            return cached
        try:
            connection = self.canonical_connection()
            found = connection.execute(
                "SELECT 1 FROM entities WHERE entity_id = ? AND entity_type = ? LIMIT 1",
                (reference, namespace),
            ).fetchone()
            state = IdentityState.PRESENT if found is not None else IdentityState.ABSENT
        except (OSError, sqlite3.Error, VerificationConflict, ValueError, TypeError):
            # UNKNOWN must remain uncached.  A later request/session can retry after a transient
            # file or database condition is repaired.
            return IdentityState.UNKNOWN
        self._identity_cache[key] = state
        return state

    def require_identity(self, reference: str, *, namespace: str) -> None:
        state = self.lookup_identity(reference, namespace=namespace)
        if state is not IdentityState.PRESENT:
            raise VerificationConflict(
                f"canonical identity {reference!r} is {state.value}; refusing to continue"
            )

    def manifests_for_output_ref(self, output_ref: str) -> tuple[ManifestEntry, ...]:
        self._ensure_open()
        self._ensure_manifest_catalog()
        assert self._manifest_catalog is not None
        return self._manifest_catalog.get(output_ref, ())

    def require_unique_manifest(self, output_ref: str) -> ManifestEntry:
        entries = self.manifests_for_output_ref(output_ref)
        if len(entries) != 1:
            raise VerificationConflict(
                f"{output_ref} requires exactly one derived artifact manifest; found {len(entries)}"
            )
        entries[0].proof.verify()
        return entries[0]

    def sample_entries(self, sample_id: str | None = None) -> tuple[SampleEntry, ...]:
        self._ensure_open()
        self._ensure_sample_catalog()
        assert self._sample_catalog is not None
        if sample_id is None:
            return tuple(entry for entries in self._sample_catalog.values() for entry in entries)
        return self._sample_catalog.get(sample_id, ())

    def require_unique_sample(self, sample_id: str) -> SampleEntry:
        entries = self.sample_entries(sample_id)
        if len(entries) != 1:
            raise VerificationConflict(
                f"{sample_id} requires exactly one training sample; found {len(entries)}"
            )
        entries[0].proof.verify()
        return entries[0]

    def file_proof(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_reference: str | None = None,
    ) -> FileProof:
        """Capture a content proof and retain it for the session stability check."""

        self._ensure_open()
        if expected_reference is not None:
            if not expected_reference.startswith("file-sha256:"):
                raise ValueError("expected_reference must use file-sha256 namespace")
            expected_sha256 = expected_reference.removeprefix("file-sha256:")
        resolved = Path(path).resolve()
        existing = self._proofs.get(resolved)
        if existing is not None:
            existing.verify()
            if expected_sha256 is not None and existing.sha256 != _normalize_digest(
                expected_sha256
            ):
                raise VerificationConflict(f"file checksum mismatch for {resolved}")
            return existing
        proof = FileProof.capture(resolved, expected_sha256=expected_sha256)
        self._proofs[resolved] = proof
        return proof

    def cached_artifact_manifest(self, artifact_id: str) -> Any | None:
        self._ensure_open()
        return self._artifact_manifests.get(artifact_id)

    def remember_artifact_manifest(self, artifact_id: str, manifest: Any) -> None:
        self._ensure_open()
        self._artifact_manifests[artifact_id] = manifest

    def cached_team_observation(self, source_ref: str) -> Any | None:
        self._ensure_open()
        return self._team_observations.get(source_ref)

    def remember_team_observation(self, source_ref: str, observation: Any) -> None:
        self._ensure_open()
        self._team_observations[source_ref] = observation

    def cached_match_result(self, source_ref: str) -> Any | None:
        self._ensure_open()
        return self._match_results.get(source_ref)

    def remember_match_result(self, source_ref: str, result: Any) -> None:
        self._ensure_open()
        self._match_results[source_ref] = result

    def cached_match_report_replay(self, contract_id: str) -> Any | None:
        self._ensure_open()
        return self._match_report_replays.get(contract_id)

    def remember_match_report_replay(self, contract_id: str, replay: Any) -> None:
        self._ensure_open()
        self._match_report_replays[contract_id] = replay

    @contextmanager
    def resolving_artifact_manifest(self, artifact_id: str) -> Iterator[None]:
        self._ensure_open()
        if artifact_id in self._resolving_manifests:
            raise VerificationConflict(f"recursive derived artifact lineage: {artifact_id}")
        self._resolving_manifests.add(artifact_id)
        try:
            yield
        finally:
            self._resolving_manifests.discard(artifact_id)

    def assert_stable(self) -> None:
        """Verify every read file and catalog directory before request completion."""

        self._ensure_open()
        for path, expected in tuple(self._sidecar_snapshots.items()):
            if _sqlite_sidecars(path) != expected:
                raise VerificationConflict(f"canonical SQLite sidecars changed: {path}")
        for proof in tuple(self._proofs.values()):
            proof.verify()
        for root, expected in tuple(self._directory_snapshots.items()):
            actual = _directory_listing(root)
            if actual != expected:
                raise VerificationConflict(f"verified catalog directory changed: {root}")

    def _ensure_manifest_catalog(self) -> None:
        if self._manifest_catalog is not None:
            return
        root = self.layout.derived / "manifests" / "artifacts"
        initial_listing, paths = _catalog_files(root)
        catalog: dict[str, list[ManifestEntry]] = defaultdict(list)
        # Assign the cache only after the complete scan succeeds.  A malformed file therefore
        # cannot poison this request or a later request with a partial index.
        for path in paths:
            proof = self.file_proof(path)
            payload = _load_json(proof.path, "derived artifact manifest")
            artifact_id = payload.get("id")
            match = (
                _DERIVED_ARTIFACT.fullmatch(artifact_id) if isinstance(artifact_id, str) else None
            )
            if match is None or path.stem != match.group(1):
                raise VerificationConflict(f"invalid derived artifact manifest identity: {path}")
            output_refs = payload.get("output_refs")
            if not isinstance(output_refs, list) or any(
                not isinstance(item, str) or not item or item.strip() != item
                for item in output_refs
            ):
                raise VerificationConflict(f"invalid derived artifact manifest outputs: {path}")
            if len(output_refs) != len(set(output_refs)):
                raise VerificationConflict(f"duplicate derived artifact manifest outputs: {path}")
            entry = ManifestEntry(
                artifact_id=artifact_id,
                path=proof.path,
                output_refs=tuple(sorted(output_refs)),
                proof=proof,
            )
            for output_ref in entry.output_refs:
                catalog[output_ref].append(entry)
        if _directory_listing(root) != initial_listing:
            raise VerificationConflict(
                f"derived manifest catalog changed while being scanned: {root}"
            )
        self._manifest_catalog = {
            reference: tuple(sorted(entries, key=lambda item: item.artifact_id))
            for reference, entries in catalog.items()
        }
        self._directory_snapshots[root.resolve()] = initial_listing

    def _ensure_sample_catalog(self) -> None:
        if self._sample_catalog is not None:
            return
        root = self.layout.derived / "training-datasets"
        initial_listing, paths = _catalog_files(root)
        catalog: dict[str, list[SampleEntry]] = defaultdict(list)
        for path in paths:
            proof = self.file_proof(path)
            payload = _load_json(proof.path, "training dataset")
            dataset_id = payload.get("id")
            match = _TRAINING_DATASET.fullmatch(dataset_id) if isinstance(dataset_id, str) else None
            if match is None or path.stem != match.group(1):
                raise VerificationConflict(f"invalid training dataset identity: {path}")
            samples = payload.get("samples")
            if not isinstance(samples, list):
                raise VerificationConflict(f"invalid training dataset samples: {path}")
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
                    raise VerificationConflict(f"invalid training sample entry: {path}")
                sample_id = sample["sample_id"]
                if not sample_id or sample_id.strip() != sample_id:
                    raise VerificationConflict(f"invalid training sample ID: {path}")
                catalog[sample_id].append(
                    SampleEntry(
                        sample_id=sample_id,
                        dataset_id=dataset_id,
                        path=proof.path,
                        proof=proof,
                    )
                )
        if _directory_listing(root) != initial_listing:
            raise VerificationConflict(
                f"training dataset catalog changed while being scanned: {root}"
            )
        self._sample_catalog = {
            reference: tuple(sorted(entries, key=lambda item: (item.dataset_id, str(item.path))))
            for reference, entries in catalog.items()
        }
        self._directory_snapshots[root.resolve()] = initial_listing

    def _canonical_path(self) -> Path:
        candidate = getattr(self.canonical, "path", self.canonical)
        if candidate is None:
            candidate = self.layout.canonical / "platform.sqlite3"
        return Path(candidate)

    def _ensure_open(self) -> None:
        if self._closed:
            raise VerificationConflict("verification session is closed")
        if not self._entered:
            raise VerificationConflict("verification session must be entered before reading")


def active_verification_session(layout: DataLayout | str | Path) -> VerificationSession | None:
    session = _ACTIVE_SESSION.get()
    if session is None or not session._entered or session._closed:
        return None
    candidate = layout if isinstance(layout, DataLayout) else DataLayout(Path(layout))
    if session.layout.root.resolve() != candidate.root.resolve():
        return None
    return session


@contextmanager
def verification_session_scope(session: VerificationSession) -> Iterator[VerificationSession]:
    existing = active_verification_session(session.layout)
    if existing is not None:
        yield existing
        return
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    if not path.is_file():
        raise OSError("path is not a regular file")
    return stat.st_size, stat.st_mtime_ns


def _sqlite_sidecars(path: Path) -> tuple[Path, ...]:
    prefix = path.name + "-"
    try:
        return tuple(
            sorted(
                item.resolve()
                for item in path.parent.iterdir()
                if item.name.startswith(prefix) and item.is_file()
            )
        )
    except OSError as error:
        raise VerificationConflict(f"cannot inspect canonical SQLite sidecars: {path}") from error


def _normalize_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected SHA-256 digest")
    if value.startswith("file-sha256:"):
        value = value.removeprefix("file-sha256:")
    if _DIGEST.fullmatch(value) is None:
        raise ValueError("expected SHA-256 digest")
    return value


def _catalog_files(root: Path) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    if not root.exists():
        return (), ()
    if not root.is_dir():
        raise VerificationConflict(f"verification catalog is not a directory: {root}")
    resolved = root.resolve()
    listing = _directory_listing(resolved)
    paths = tuple(
        sorted(
            (resolved / relative for relative in listing if relative.casefold().endswith(".json")),
            key=str,
        )
    )
    return listing, paths


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationConflict(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise VerificationConflict(f"{description} must be an object: {path}")
    return payload


def _directory_listing(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    if not root.is_dir():
        raise VerificationConflict(f"verified catalog is not a directory: {root}")
    try:
        return tuple(
            sorted(
                str(path.relative_to(root).as_posix()) for path in root.rglob("*") if path.is_file()
            )
        )
    except OSError as error:
        raise VerificationConflict(f"cannot inspect verified catalog: {root}") from error
