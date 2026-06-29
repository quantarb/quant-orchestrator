from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

DEFAULT_ARTIFACT_ROOT = Path("artifacts/orchestrator")
ARTIFACT_ROOT_ENV = "QUANT_ORCHESTRATOR_ARTIFACT_ROOT"

RunStatus = Literal["running", "completed", "failed"]


@dataclass(frozen=True)
class RunRecord:
    id: str
    run_type: str
    name: str
    status: RunStatus
    created_at: str
    updated_at: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    tags: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    run_id: str
    kind: str
    name: str
    path: Path
    format: str
    created_at: str
    metadata: dict[str, Any]

    @property
    def uri(self) -> str:
        return f"artifact:{self.id}"


class ArtifactStore:
    """Filesystem artifact store with a SQLite registry.

    The orchestrator owns this registry so downstream applications can request work,
    receive artifact URIs or paths, and load outputs without maintaining separate ML storage.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv(ARTIFACT_ROOT_ENV) or DEFAULT_ARTIFACT_ROOT).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "registry.sqlite"
        self._initialize()

    def create_run(
        self,
        *,
        run_type: str,
        name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RunRecord:
        run_id = _new_id("run")
        now = _now()
        record = RunRecord(
            id=run_id,
            run_type=_clean_token(run_type),
            name=str(name),
            status="running",
            created_at=now,
            updated_at=now,
            params=dict(params or {}),
            metrics={},
            tags=dict(tags or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, run_type, name, status, created_at, updated_at,
                    params_json, metrics_json, tags_json, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.run_type,
                    record.name,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    _dumps(record.params),
                    _dumps(record.metrics),
                    _dumps(record.tags),
                    record.error,
                ),
            )
        return record

    def complete_run(
        self,
        run_id: str,
        *,
        status: RunStatus = "completed",
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RunRecord:
        if status not in {"completed", "failed"}:
            raise ValueError("Completed runs must use status='completed' or status='failed'")
        existing = self.get_run(run_id)
        merged_metrics = {**existing.metrics, **dict(metrics or {})}
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, updated_at = ?, metrics_json = ?, error = ?
                WHERE id = ?
                """,
                (status, now, _dumps(merged_metrics), error, run_id),
            )
        return self.get_run(run_id)

    def fail_run(self, run_id: str, error: BaseException | str) -> RunRecord:
        return self.complete_run(run_id, status="failed", error=str(error))

    def update_run(
        self,
        run_id: str,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RunRecord:
        existing = self.get_run(run_id)
        updated_params = {**existing.params, **dict(params or {})}
        updated_metrics = {**existing.metrics, **dict(metrics or {})}
        updated_tags = {**existing.tags, **dict(tags or {})}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET updated_at = ?, params_json = ?, metrics_json = ?, tags_json = ?
                WHERE id = ?
                """,
                (
                    _now(),
                    _dumps(updated_params),
                    _dumps(updated_metrics),
                    _dumps(updated_tags),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown orchestrator run: {run_id}")
        return _run_from_row(row)

    def list_runs(
        self,
        *,
        run_type: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        clauses = []
        values: list[Any] = []
        if run_type is not None:
            clauses.append("run_type = ?")
            values.append(_clean_token(run_type))
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [_run_from_row(row) for row in rows]

    def delete_run(self, run_id: str, *, delete_files: bool = False) -> None:
        artifacts = self.list_artifacts(run_id=run_id, limit=1_000_000)
        with self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        if delete_files:
            for artifact in artifacts:
                _remove_path(artifact.path)

    def save_dataframe(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        frame: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        index: bool = False,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id, kind, name, "csv")
        frame.to_csv(path, index=index)
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=path,
            format="csv",
            metadata=metadata,
        )

    def load_dataframe(self, artifact_id_or_uri: str) -> pd.DataFrame:
        artifact = self.get_artifact(artifact_id_or_uri)
        if artifact.format != "csv":
            raise ValueError(f"Artifact {artifact.id} is not a CSV dataframe artifact")
        return pd.read_csv(artifact.path)

    def save_json(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id, kind, name, "json")
        path.write_text(_dumps(payload, indent=2), encoding="utf-8")
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=path,
            format="json",
            metadata=metadata,
        )

    def load_json(self, artifact_id_or_uri: str) -> dict[str, Any]:
        artifact = self.get_artifact(artifact_id_or_uri)
        if artifact.format != "json":
            raise ValueError(f"Artifact {artifact.id} is not a JSON artifact")
        return json.loads(artifact.path.read_text(encoding="utf-8"))

    def save_text(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        text: str,
        extension: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id, kind, name, extension)
        path.write_text(text, encoding="utf-8")
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=path,
            format=_clean_extension(extension),
            metadata=metadata,
        )

    def load_text(self, artifact_id_or_uri: str) -> str:
        artifact = self.get_artifact(artifact_id_or_uri)
        return artifact.path.read_text(encoding="utf-8")

    def save_bytes(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        data: bytes,
        extension: str = "bin",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id, kind, name, extension)
        path.write_bytes(data)
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=path,
            format=_clean_extension(extension),
            metadata=metadata,
        )

    def load_bytes(self, artifact_id_or_uri: str) -> bytes:
        artifact = self.get_artifact(artifact_id_or_uri)
        return artifact.path.read_bytes()

    def save_pickle(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id, kind, name, "pkl")
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=path,
            format="pickle",
            metadata=metadata,
        )

    def load_pickle(self, artifact_id_or_uri: str) -> Any:
        artifact = self.get_artifact(artifact_id_or_uri)
        if artifact.format != "pickle":
            raise ValueError(f"Artifact {artifact.id} is not a pickle artifact")
        with artifact.path.open("rb") as handle:
            return pickle.load(handle)

    def register_file(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        path: str | Path,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
        copy: bool = True,
    ) -> ArtifactRecord:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        extension = "directory" if source.is_dir() else source.suffix.lstrip(".") or (format or "artifact")
        destination = self._artifact_path(run_id, kind, name, extension)
        if copy:
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            registered_path = destination
        else:
            registered_path = source
        return self._register_artifact(
            run_id=run_id,
            kind=kind,
            name=name,
            path=registered_path,
            format=format or extension,
            metadata=metadata,
        )

    def get_artifact(self, artifact_id_or_uri: str) -> ArtifactRecord:
        artifact_id = _artifact_id_from_uri(artifact_id_or_uri)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown orchestrator artifact: {artifact_id_or_uri}")
        return _artifact_from_row(row)

    def latest_artifact(
        self,
        *,
        kind: str,
        name: str | None = None,
    ) -> ArtifactRecord:
        query = "SELECT * FROM artifacts WHERE kind = ?"
        values: list[Any] = [_clean_token(kind)]
        if name is not None:
            query += " AND name = ?"
            values.append(str(name))
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()
        if row is None:
            name_text = "" if name is None else f" named {name!r}"
            raise FileNotFoundError(f"No artifact found for kind {kind!r}{name_text}")
        return _artifact_from_row(row)

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[ArtifactRecord]:
        query = "SELECT * FROM artifacts"
        clauses = []
        values: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            values.append(_clean_token(kind))
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if name is not None:
            clauses.append("name = ?")
            values.append(str(name))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [_artifact_from_row(row) for row in rows]

    def update_artifact_metadata(
        self,
        artifact_id_or_uri: str,
        metadata: dict[str, Any],
    ) -> ArtifactRecord:
        artifact = self.get_artifact(artifact_id_or_uri)
        merged = {**artifact.metadata, **dict(metadata)}
        with self._connect() as conn:
            conn.execute(
                "UPDATE artifacts SET metadata_json = ? WHERE id = ?",
                (_dumps(merged), artifact.id),
            )
        return self.get_artifact(artifact.id)

    def delete_artifact(self, artifact_id_or_uri: str, *, delete_file: bool = False) -> None:
        artifact = self.get_artifact(artifact_id_or_uri)
        with self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact.id,))
        if delete_file:
            _remove_path(artifact.path)

    def _artifact_path(self, run_id: str, kind: str, name: str, extension: str) -> Path:
        directory = self.root / "files" / _clean_token(kind) / _clean_token(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_clean_filename(name)}.{_clean_extension(extension)}"
        if not path.exists():
            return path
        return directory / f"{_clean_filename(name)}-{uuid4().hex[:8]}.{_clean_extension(extension)}"

    def _register_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        name: str,
        path: Path,
        format: str,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        self.get_run(run_id)
        record = ArtifactRecord(
            id=_new_id("art"),
            run_id=run_id,
            kind=_clean_token(kind),
            name=str(name),
            path=path.expanduser().resolve(),
            format=str(format),
            created_at=_now(),
            metadata=dict(metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, run_id, kind, name, path, format, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.run_id,
                    record.kind,
                    record.name,
                    str(record.path),
                    record.format,
                    record.created_at,
                    _dumps(record.metadata),
                ),
            )
        return record

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    error TEXT
                )
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    format TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """,
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_kind_name ON artifacts(kind, name)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id)",
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_type_status ON runs(run_type, status)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def get_artifact_store(root: str | Path | None = None) -> ArtifactStore:
    return ArtifactStore(root=root)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, default=str)


def _loads(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Registry JSON payload must be an object")
    return payload


def _artifact_id_from_uri(value: str) -> str:
    text = str(value)
    return text.removeprefix("artifact:")


def _clean_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned.strip("_").lower() or "artifact"


def _clean_filename(value: str) -> str:
    return _clean_token(value)


def _clean_extension(value: str) -> str:
    return _clean_token(value).replace("_", "") or "artifact"


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        run_type=row["run_type"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        params=_loads(row["params_json"]),
        metrics=_loads(row["metrics_json"]),
        tags=_loads(row["tags_json"]),
        error=row["error"],
    )


def _artifact_from_row(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        id=row["id"],
        run_id=row["run_id"],
        kind=row["kind"],
        name=row["name"],
        path=Path(row["path"]),
        format=row["format"],
        created_at=row["created_at"],
        metadata=_loads(row["metadata_json"]),
    )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
