"""One trace per case, one span per stage attempt. JSONL sink (+ optional OTel).

The same trace_id flows A -> B -> C and is copied into every contract's trace_id field.
Never log raw patient/case payloads - hash them with hash_payload().
"""
import hashlib
import json
import time
import uuid
from pathlib import Path

from common.settings import settings


def hash_payload(obj) -> str:
    try:
        blob = json.dumps(obj, default=str, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
    except Exception:
        return "na"


class LoopTrace:
    """Carried through every stage. Open one at the orchestrator, pass children down."""

    def __init__(self, case_id: str, trace_id: str | None = None, stage: str | None = None):
        self.case_id = case_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.stage = stage
        self._dir = Path(settings.OSTEON_TRACE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{self.trace_id}.jsonl"

    def child(self, stage: str) -> "LoopTrace":
        return LoopTrace(self.case_id, self.trace_id, stage)

    def emit(self, **fields) -> dict:
        span = {
            "trace_id": self.trace_id,
            "span_id": uuid.uuid4().hex[:8],
            "stage": self.stage,
            "ts": round(time.time(), 3),
        }
        span.update(fields)
        with open(self._path, "a") as f:
            print(json.dumps(span, default=str), file=f)
        return span
