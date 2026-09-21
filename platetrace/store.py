"""Local case history without credentials or hidden model reasoning."""
import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


class Run:
    def __init__(self, data: dict, directory: Path):
        self.data = data
        self.directory = directory
        self.changed = asyncio.Condition()
        self.task: asyncio.Task | None = None

    @property
    def id(self):
        return self.data["id"]

    def save(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.directory / f"{self.id}.json"
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(self.data, file, ensure_ascii=False)
        tmp.replace(path)

    async def emit(self, kind: str, data: dict):
        async with self.changed:
            event = {"id": len(self.data["events"]) + 1, "type": kind, "time": now(), "data": data}
            self.data["events"].append(event)
            self.save()
            self.changed.notify_all()

    async def status(self, status: str):
        self.data["status"] = status
        await self.emit("status", {"status": status})


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        self.runs: dict[str, Run] = {}

    def load(self):
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)[-100:]:
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict) or not all(key in data for key in (
                    "id", "status", "plate", "jurisdiction", "provider", "model_id", "created_at", "events", "sources"
                )):
                    continue
                if not isinstance(data["events"], list) or not isinstance(data["sources"], list):
                    continue
                if not re.fullmatch(r"[a-f0-9]{32}", data["id"]):
                    continue
                if data["status"] in ("queued", "running", "cancelling"):
                    data["status"] = "interrupted"
                    data["error"] = "The server stopped before this run completed. Start a new run to continue."
                run = Run(data, self.directory)
                self.runs[run.id] = run
                run.save()
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def add(self, data: dict) -> Run:
        if len(self.runs) >= 100:
            oldest = next((key for key, run in self.runs.items()
                           if run.data["status"] not in ("queued", "running", "cancelling")), None)
            if oldest:
                del self.runs[oldest]
                (self.directory / f"{oldest}.json").unlink(missing_ok=True)
        run = Run(data, self.directory)
        self.runs[run.id] = run
        run.save()
        return run


def markdown_report(data: dict) -> str:
    report = data.get("report") or {}
    lines = ["# PlateTrace research report", "", f"Plate: {data['plate']}",
             f"Jurisdiction: {data['jurisdiction']}", f"Status: {data['status']}",
             f"Provider: {data['provider']} / {data['model_id']}", f"Created: {data['created_at']}", "",
             report.get("summary", "No completed report is available."), ""]
    for finding in report.get("findings", []):
        lines += [f"## {finding['title']}", "", finding["detail"], "",
                  "Sources: " + ", ".join(finding["source_ids"]), ""]
    for title, key in (("Limitations", "limitations"), ("Next steps", "next_steps")):
        lines += [f"## {title}", ""]
        lines += [f"- {item}" for item in report.get(key, [])]
        lines += [""]
    lines += ["## Sources", ""]
    for source in data["sources"]:
        lines += [f"- {source['id']}: {source['title']} — {source.get('url', 'User-provided data')}",
                  f"  Retrieved: {source['retrieved_at']}; type: {source['kind']}"]
    if data.get("issue_reports"):
        lines += ["", "## GitHub issue reports", ""]
        for issue in data["issue_reports"]:
            lines += [f"- #{issue['number']} ({issue['status']}): {issue['url']}"]
    memory = data.get("memory")
    if memory:
        lines += ["", "## Research memory", "",
                  "Research memory supplies planning context only. Findings require this run's sources.",
                  f"Memory enabled: {'yes' if memory.get('enabled') else 'no'}.",
                  f"New memory saved: {'yes' if memory.get('saved') else 'no'}."]
        for entry in memory.get("recalled", []):
            lines += [f"- Recalled run {entry['id']} from {entry['created_at']}."]
        if memory.get("error"):
            lines += [memory["error"]]
        for lesson in data.get("research_lessons", []):
            lines += [f"- Research lesson: {lesson}"]
    lines += ["", "AI-generated research requires independent verification. A plate is not a unique global vehicle identifier."]
    return "\n".join(lines) + "\n"
