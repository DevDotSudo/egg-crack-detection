from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default image output directory: %USERPROFILE%\Pictures\detected_eggs
# ---------------------------------------------------------------------------
_DEFAULT_PICTURES_DIR = (
    Path(os.environ.get('USERPROFILE', Path.home())) / 'Pictures' / 'detected_eggs'
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id                    TEXT PRIMARY KEY,
    timestamp             TEXT NOT NULL,
    source_name           TEXT NOT NULL DEFAULT 'camera',
    is_crack              INTEGER NOT NULL,
    confidence            REAL,
    egg_size              TEXT,
    egg_size_confidence   REAL,
    egg_size_score        REAL,
    crack_size            TEXT,
    crack_size_confidence REAL,
    contour_length        REAL,
    area_ratio            REAL,
    egg_width_pixels      REAL,
    egg_length_pixels     REAL,
    processing_time_ms    INTEGER,
    original_image_path   TEXT,
    overlay_image_path    TEXT,
    metadata_json         TEXT
);
"""

# Fields stripped from the metadata JSON before storage (already saved as files).
_IMAGE_FIELDS = {'original_image_b64', 'overlay_image_b64', 'intermediate_steps', 'crack_mask_b64'}


class DetectionDB:
    """SQLite-backed detection history.

    Images (original JPEG + overlay PNG) are written to *pictures_dir*.
    Only the absolute file paths are stored in the database; no base-64
    blobs are persisted to disk.

    Thread-safety: a single ``threading.Lock`` serialises all writes.
    Reads open a short-lived read-only connection to avoid blocking writes.
    """

    def __init__(
        self,
        db_path: Path,
        pictures_dir: Path = _DEFAULT_PICTURES_DIR,
    ) -> None:
        self.db_path = db_path
        self.pictures_dir = pictures_dir
        self._lock = threading.Lock()
        self._init_db()
        self.pictures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA foreign_keys=ON;')
            conn.execute('PRAGMA busy_timeout=5000;')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE_SQL)

    def _save_images(self, item: dict[str, Any]) -> tuple[str | None, str | None]:
        """Decode and write image files; return (original_path, overlay_path)."""
        record_id = item.get('id', 'unknown')
        original_path: str | None = None
        overlay_path: str | None = None

        orig_b64: str | None = item.get('original_image_b64')
        if orig_b64:
            try:
                img_bytes = base64.b64decode(orig_b64)
                dest = self.pictures_dir / f'{record_id}_original.jpg'
                dest.write_bytes(img_bytes)
                original_path = str(dest)
            except Exception:
                pass

        ov_b64: str | None = item.get('overlay_image_b64')
        if ov_b64:
            try:
                img_bytes = base64.b64decode(ov_b64)
                dest = self.pictures_dir / f'{record_id}_overlay.png'
                dest.write_bytes(img_bytes)
                overlay_path = str(dest)
            except Exception:
                pass

        return original_path, overlay_path

    @staticmethod
    def _strip_images(item: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in item.items() if k not in _IMAGE_FIELDS}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        meta_raw = d.pop('metadata_json', None)
        if meta_raw:
            try:
                d.update(json.loads(meta_raw))
            except (json.JSONDecodeError, TypeError):
                pass
        d['is_crack'] = bool(d.get('is_crack', 0))
        return d

    # ------------------------------------------------------------------
    # Public CRUD API
    # ------------------------------------------------------------------

    def add(self, item: dict[str, Any]) -> bool:
        """Insert or replace a detection record.

        Images are saved to *pictures_dir* first; then only paths are
        stored in the database.

        Returns ``True`` if this is a new record, ``False`` if replaced.
        """
        record_id: str = item.get('id', '')
        if not record_id:
            return False

        with self._lock:
            original_path, overlay_path = self._save_images(item)
            clean = self._strip_images(item)
            metadata_json = json.dumps(clean)

            ts = item.get('timestamp') or datetime.now(timezone.utc).isoformat()
            row = (
                record_id,
                ts,
                str(item.get('source_name', 'camera')),
                int(bool(item.get('is_crack', False))),
                item.get('confidence'),
                item.get('egg_size'),
                item.get('egg_size_confidence'),
                item.get('egg_size_score'),
                item.get('crack_size'),
                item.get('crack_size_confidence'),
                item.get('contour_length'),
                item.get('area_ratio'),
                item.get('egg_width_pixels'),
                item.get('egg_length_pixels'),
                item.get('processing_time_ms'),
                original_path,
                overlay_path,
                metadata_json,
            )

            with self._connect() as conn:
                existing = conn.execute(
                    'SELECT id FROM detections WHERE id = ?', (record_id,),
                ).fetchone()
                conn.execute(
                    '''INSERT OR REPLACE INTO detections
                       (id, timestamp, source_name, is_crack, confidence,
                        egg_size, egg_size_confidence, egg_size_score,
                        crack_size, crack_size_confidence,
                        contour_length, area_ratio,
                        egg_width_pixels, egg_length_pixels,
                        processing_time_ms,
                        original_image_path, overlay_image_path,
                        metadata_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    row,
                )
            return existing is None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the *limit* most-recent detection records (metadata only)."""
        limit = max(1, min(limit, 1000))
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, item_id: str) -> dict[str, Any] | None:
        """Return a single detection record by ID, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM detections WHERE id = ?', (item_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def delete(self, item_id: str) -> bool:
        """Delete a detection record AND its image files.

        Returns ``True`` if the record existed and was deleted.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    'SELECT original_image_path, overlay_image_path FROM detections WHERE id = ?',
                    (item_id,),
                ).fetchone()
                if row is None:
                    return False
                conn.execute('DELETE FROM detections WHERE id = ?', (item_id,))

            # Remove image files (best-effort).
            for path_str in (row['original_image_path'], row['overlay_image_path']):
                if path_str:
                    try:
                        Path(path_str).unlink(missing_ok=True)
                    except OSError:
                        pass
        return True

    def clear(self) -> int:
        """Delete all records and their image files. Returns count deleted."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    'SELECT original_image_path, overlay_image_path FROM detections',
                ).fetchall()
                count = len(rows)
                conn.execute('DELETE FROM detections')

            for row in rows:
                for path_str in (row['original_image_path'], row['overlay_image_path']):
                    if path_str:
                        try:
                            Path(path_str).unlink(missing_ok=True)
                        except OSError:
                            pass
        return count

    def get_image_paths(self, item_id: str) -> tuple[str | None, str | None]:
        """Return (original_path, overlay_path) for an ID, or (None, None)."""
        with self._connect() as conn:
            row = conn.execute(
                'SELECT original_image_path, overlay_image_path FROM detections WHERE id = ?',
                (item_id,),
            ).fetchone()
        if row is None:
            return None, None
        return row['original_image_path'], row['overlay_image_path']
