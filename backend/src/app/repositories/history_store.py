from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text('[]', encoding='utf-8')

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def add(self, item: dict[str, Any]) -> bool:
        with self.lock:
            items = self._read()
            summary = {k: v for k, v in item.items() if k not in {'original_image_b64', 'overlay_image_b64', 'intermediate_steps'}}
            item_id = summary.get('id')
            replaced = False
            if item_id:
                for index, existing in enumerate(items):
                    if existing.get('id') == item_id:
                        items[index] = summary
                        replaced = True
                        break
            if not replaced:
                items.insert(0, summary)
            self._write(items[:1000])
            return not replaced

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            return self._read()[:max(1, min(limit, 1000))]

    def get(self, item_id: str) -> dict[str, Any] | None:
        with self.lock:
            return next((x for x in self._read() if x.get('id') == item_id), None)

    def delete(self, item_id: str) -> bool:
        with self.lock:
            items = self._read()
            filtered = [x for x in items if x.get('id') != item_id]
            if len(filtered) == len(items):
                return False
            self._write(filtered)
            return True

    def clear(self) -> int:
        with self.lock:
            items = self._read()
            count = len(items)
            self._write([])
            return count

    def _write(self, items: list[dict[str, Any]]) -> None:
        """Replace the history file atomically so a crash cannot truncate it."""
        payload = json.dumps(items, indent=2)
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f'{self.path.name}.',
            suffix='.tmp',
            text=True,
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
