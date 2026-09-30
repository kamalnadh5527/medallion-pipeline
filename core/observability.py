import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    def __init__(self, agent_name: str, run_id: str) -> None:
        self.agent_name = agent_name
        self.run_id = run_id
        self._start_time = time.time()
        self.trace: dict[str, Any] = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": None,
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs: Any) -> "AgentTrace":
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str: str) -> "AgentTrace":
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs: Any) -> "AgentTrace":
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages: list[Any]) -> "AgentTrace":
        for message in messages:
            message_type = type(message).__name__
            content = getattr(message, "content", "")
            tool_calls = getattr(message, "tool_calls", None) or []

            if message_type == "HumanMessage":
                self.trace["reasoning_steps"].append({
                    "type": "task_input",
                    "content": content,
                })

            elif message_type == "AIMessage":
                if tool_calls:
                    self.trace["tool_calls"].extend(tool_calls)

                if content:
                    self.trace["reasoning_steps"].append({
                        "type": "ai_message",
                        "content": content,
                    })
                    if self.trace["plan"] is None and str(content).strip():
                        self.trace["plan"] = content

            elif message_type == "ToolMessage":
                self.trace["reasoning_steps"].append({
                    "type": "tool_result",
                    "tool": getattr(message, "name", None),
                    "content": content,
                })

        return self

    def complete(self, status: str = "success") -> None:
        self.trace["duration_seconds"] = round(time.time() - self._start_time, 6)
        self.trace["status"] = status

        safe_agent_name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.agent_name)
        trace_path = TRACES_DIR / f"trace_{safe_agent_name}_{self.run_id[:8]}.json"
        trace_path.write_text(
            json.dumps(self.trace, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(
            f"Trace saved: agent={self.agent_name}, "
            f"status={status}, duration={self.trace['duration_seconds']}s"
        )

    def fail(self, error: Any) -> None:
        self.trace["error"] = str(error)
        self.complete(status="failed")