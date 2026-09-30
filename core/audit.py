import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = str(run_id or uuid.uuid4())
        self.log_file = Path(AUDIT_DIR) / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def get_logs(self) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []

        with self.log_file.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]