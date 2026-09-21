"""Bounded local research lessons, never a second store of vehicle case data."""
import copy
import ipaddress
import json
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from .issues import _sanitize
from .models import RunRequest

MAX_ENTRIES = 200
MAX_RECALLED = 5
MAX_AGE_DAYS = 90
MAX_LESSONS = 5
MAX_LESSON_LENGTH = 1000
MAX_SOURCES = 6
_MAX_FILE_BYTES = 2_000_000
_STATUSES = {"completed", "incomplete", "failed"}
_TOOLS = {"search_web", "fetch_url", "read_records", "terminal", "report_issue"}
_OUTCOMES = {
    "completed": "The research run completed a report; vehicle claims still require fresh evidence.",
    "incomplete": "The research run reached its turn limit without a completed report.",
    "failed": "The research run stopped after an error; observed tool outcomes may guide a retry.",
}
_PURPOSE_PATTERNS = {
    "recalls": r"\brecalls?\b",
    "specifications": r"\b(?:specifications?|specs?|dimensions?|engine|transmission|trim)\b",
    "vehicle_details": r"\b(?:vehicle details|vehicle facts|make|model|decode|decoding|identify)\b",
    "fuel": r"\b(?:fuel|mpg|economy|electric|battery|range)\b",
    "emissions": r"\b(?:emissions?|pollution|co2)\b",
    "safety": r"\b(?:safety|crash|ratings?)\b",
    "maintenance": r"\b(?:maintenance|service|repair|manuals?)\b",
    "history": r"\b(?:history|registration|inspection|mot)\b",
}
_VIN = re.compile(r"(?i)\b[A-HJ-NPR-Z0-9]{17}\b")
_COMMAND = re.compile(
    r"```|`[^`]+`|\$\(|(?:^|\n)\s*(?:\$|>|Traceback|HTTP/\d)|"
    r"(?:^|[\s;|])(?:curl|wget|sudo|bash|sh|zsh|python[\d.]*|pip[\d.]*|npm|docker|"
    r"git|ssh|export|cat|echo|printf)\s+(?:-|https?://|[\w./])", re.IGNORECASE
)


def _decoded(value: str) -> str:
    for _ in range(3):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    return value


def _clean_text(value: str, request: RunRequest, limit: int) -> str:
    # Reuse the existing identifier/credential redactor without its issue schema.
    value = _sanitize(value, request)
    value = _VIN.sub("[REDACTED VEHICLE ID]", value)
    value = re.sub(r"(?i)https?://[^\s<>\"']+", "[source URL omitted]", value)
    value = re.sub(r"\bS\d+\b", "[prior source]", value)
    return " ".join(value.split())[:limit]


def prepare_lessons(request: RunRequest, values: list[str]) -> list[str]:
    """Scrub model lessons before they enter run events or durable memory."""
    if not isinstance(values, list):
        return []
    lessons = []
    for value in values[:MAX_LESSONS]:
        if not isinstance(value, str) or len(value) > MAX_LESSON_LENGTH or _COMMAND.search(value):
            continue
        lesson = _clean_text(value, request, MAX_LESSON_LENGTH)
        if lesson and lesson not in lessons:
            lessons.append(lesson)
    return lessons


def _purpose_tags(request: RunRequest) -> list[str]:
    text = _clean_text(f"{request.purpose} {request.objective}", request, 2200).casefold()
    return [tag for tag, pattern in _PURPOSE_PATTERNS.items() if re.search(pattern, text)]


def _context(value: str, request: RunRequest) -> str:
    value = _clean_text(value, request, 80)
    # A redacted context is useless for matching and should not become a label.
    return "" if "[" in value or _COMMAND.search(value) else value


def _public_url(value: str, request: RunRequest | None = None) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 2000:
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (parsed.scheme not in ("http", "https") or not host or parsed.username or parsed.password
                or any(char.isspace() for char in value) or "\\" in value):
            return None
        if host.casefold() == "localhost" or host.casefold().endswith((".localhost", ".local", ".internal")):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            if "." not in host:
                return None
        path = _decoded(parsed.path)
        if (_VIN.search(path) or re.search(r"(?i)/(?:plates?|vins?|license[-_]?plates?)(?:/|$)", path)
                or re.search(r"[\x00-\x1f\x7f]", path)):
            return None
        clean = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, "", ""))
        # A known identifier anywhere in the URL makes it unsuitable as a lead.
        if request is not None and _sanitize(_decoded(clean), request) != _decoded(clean):
            return None
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return None
        return clean
    except (ValueError, OverflowError):
        return None


def _timestamp(value) -> datetime | None:
    if not isinstance(value, str) or len(value) > 60:
        return None
    try:
        result = datetime.fromisoformat(value)
        return result.astimezone(UTC) if result.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _fresh(entry: dict) -> bool:
    created = _timestamp(entry.get("created_at"))
    current = datetime.now(UTC)
    return created is not None and current - timedelta(days=MAX_AGE_DAYS) <= created <= current + timedelta(minutes=5)


def _validated_entry(value) -> dict | None:
    """Allow only the exact persisted schema; ignore corrupt or foreign entries."""
    keys = {"id", "created_at", "status", "jurisdiction", "make", "model", "year", "purpose_tags",
            "lessons", "sources", "tool_outcomes", "outcome"}
    if not isinstance(value, dict) or set(value) != keys:
        return None
    if (not isinstance(value["id"], str) or not re.fullmatch(r"[a-f0-9]{32}", value["id"])
            or not isinstance(value["status"], str) or value["status"] not in _STATUSES or not _fresh(value)
            or value["outcome"] != _OUTCOMES[value["status"]]):
        return None
    if any(not isinstance(value[key], str) or len(value[key]) > 80 for key in ("jurisdiction", "make", "model")):
        return None
    if value["year"] is not None and (type(value["year"]) is not int or not 1886 <= value["year"] <= 2100):
        return None
    tags = value["purpose_tags"]
    if (not isinstance(tags, list) or len(tags) > len(_PURPOSE_PATTERNS)
            or any(not isinstance(tag, str) or tag not in _PURPOSE_PATTERNS for tag in tags)):
        return None
    lessons = value["lessons"]
    if (not isinstance(lessons, list) or len(lessons) > MAX_LESSONS
            or any(not isinstance(item, str) or not 1 <= len(item) <= MAX_LESSON_LENGTH for item in lessons)):
        return None
    sources = value["sources"]
    if (not isinstance(sources, list) or len(sources) > MAX_SOURCES
            or any(not isinstance(source, dict) or set(source) != {"url"}
                   or not isinstance(source["url"], str)
                   or _public_url(source["url"]) != source["url"] for source in sources)):
        return None
    outcomes = value["tool_outcomes"]
    if not isinstance(outcomes, dict) or not set(outcomes) <= _TOOLS:
        return None
    for counts in outcomes.values():
        if (not isinstance(counts, dict) or set(counts) != {"success", "failure"}
                or any(type(count) is not int or not 0 <= count <= 1000 for count in counts.values())):
            return None
    return copy.deepcopy(value)


class MemoryStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.entries: list[dict] = []
        self.generation = 0

    @property
    def path(self) -> Path:
        return self.directory / "entries.json"

    def load(self):
        self.entries = []
        self.generation = 0
        try:
            if not self.path.exists() or self.path.stat().st_size > _MAX_FILE_BYTES:
                return
            self.directory.chmod(0o700)
            self.path.chmod(0o600)
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1
                    or type(data.get("generation")) is not int or data["generation"] < 0
                    or not isinstance(data.get("entries"), list)):
                return
            self.generation = data["generation"]
            unique = {}
            for raw in data["entries"]:
                entry = _validated_entry(raw)
                if entry is not None:
                    unique[entry["id"]] = entry
            self.entries = sorted(unique.values(), key=lambda entry: entry["created_at"], reverse=True)[:MAX_ENTRIES]
        except (OSError, ValueError, TypeError, KeyError):
            # Memory is optional; corrupt data must not prevent the app starting.
            self.entries = []

    def _save(self):
        """Replace atomically, with private permissions and no partial JSON writes."""
        temporary = None
        try:
            payload = json.dumps({"version": 1, "generation": self.generation, "entries": self.entries},
                                 ensure_ascii=False).encode("utf-8")
            while len(payload) > _MAX_FILE_BYTES and self.entries:
                self.entries = self.entries[:-1]
                payload = json.dumps({"version": 1, "generation": self.generation, "entries": self.entries},
                                     ensure_ascii=False).encode("utf-8")
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.chmod(0o700)
            fd, temporary = tempfile.mkstemp(prefix=".entries-", suffix=".tmp", dir=self.directory)
            with os.fdopen(fd, "wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, UnicodeError):
            raise OSError("Research memory could not be saved locally.") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def snapshot(self) -> dict:
        entries = [entry for entry in self.entries if _fresh(entry)]
        return {"entries": copy.deepcopy(entries), "count": len(entries), "generation": self.generation}

    def bootstrap(self, history) -> int:
        """Import only old observed outcomes and public origins, exactly once."""
        if self.path.exists():
            return 0
        imported = 0
        fields = {"plate", "vin", "jurisdiction", "make", "model", "year", "provider", "model_id",
                  "purpose", "objective", "authorized", "use_memory"}
        for raw in history:
            if not isinstance(raw, dict):
                continue
            try:
                request = RunRequest(**{key: raw[key] for key in fields if key in raw})
            except ValueError:
                continue
            sources = []
            for source in raw.get("sources", []):
                if (not isinstance(source, dict) or source.get("kind") != "web"
                        or not isinstance(source.get("url"), str)):
                    continue
                try:
                    parsed = urlsplit(source.get("url", ""))
                    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
                except (ValueError, TypeError):
                    continue
                url = _public_url(origin, request)
                if url:
                    parsed = urlsplit(url)
                    sources.append({"kind": "web", "url": urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))})
            data = {key: raw[key] for key in ("id", "created_at", "status", "provider", "events") if key in raw}
            data["sources"] = sources
            if self.remember(data, request, generation=self.generation) is not None:
                imported += 1
        # Even an empty import is durable, so deleting memory never restores history.
        self._save()
        return imported

    def clear(self):
        old_entries, old_generation = self.entries, self.generation
        self.entries = []
        self.generation += 1
        try:
            self._save()
        except OSError:
            self.entries, self.generation = old_entries, old_generation
            raise

    def recall(self, request: RunRequest, *, exclude_run_id: str = "") -> list[dict]:
        if not getattr(request, "use_memory", True) or request.provider == "demo":
            return []
        context = {key: _context(getattr(request, key), request).casefold()
                   for key in ("jurisdiction", "make", "model")}
        tags = set(_purpose_tags(request))
        ranked = []
        for entry in self.entries:
            if entry["id"] == exclude_run_id or not _fresh(entry):
                continue
            score = sum(weight for key, weight in (("jurisdiction", 5), ("make", 3), ("model", 4))
                        if context[key] and context[key] == entry[key].casefold())
            score += 2 * len(tags.intersection(entry["purpose_tags"]))
            if not score:
                continue
            if request.year is not None and request.year == entry["year"]:
                score += 1
            ranked.append((score, entry["created_at"], entry["id"], entry))
        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [copy.deepcopy(item[3]) for item in ranked[:MAX_RECALLED]]

    def remember(self, run_data: dict, request: RunRequest, *, generation: int) -> dict | None:
        if (generation != self.generation or not getattr(request, "use_memory", True)
                or request.provider == "demo" or run_data.get("provider") == "demo"
                or not isinstance(run_data.get("status"), str) or run_data["status"] not in _STATUSES):
            return None
        run_id = run_data.get("id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            return None
        previous = next((entry for entry in self.entries if entry["id"] == run_id), None)
        if previous is not None:
            return copy.deepcopy(previous)
        outcomes = {}
        for event in run_data.get("events", []):
            if not isinstance(event, dict) or event.get("type") != "tool_result":
                continue
            data = event.get("data")
            if (not isinstance(data, dict) or not isinstance(data.get("name"), str)
                    or data["name"] not in _TOOLS):
                continue
            result = data.get("result")
            if not isinstance(result, dict):
                continue
            failed = (bool(result.get("error")) or bool(result.get("timed_out"))
                      or bool(result.get("cancelled"))
                      or (data["name"] == "terminal" and result.get("exit_code") != 0))
            counts = outcomes.setdefault(data["name"], {"success": 0, "failure": 0})
            key = "failure" if failed else "success"
            counts[key] = min(counts[key] + 1, 1000)
        sources = []
        for source in run_data.get("sources", []):
            if not isinstance(source, dict) or source.get("kind") != "web":
                continue
            url = _public_url(source.get("url"), request)
            if url and {"url": url} not in sources:
                sources.append({"url": url})
            if len(sources) == MAX_SOURCES:
                break
        lessons = prepare_lessons(request, run_data.get("research_lessons", []))
        if not outcomes and not sources:
            return None
        created = _timestamp(run_data.get("created_at"))
        if created is None:
            return None
        entry = {
            "id": run_id, "created_at": created.isoformat(),
            "status": run_data["status"],
            **{key: _context(getattr(request, key), request) for key in ("jurisdiction", "make", "model")},
            "year": request.year, "purpose_tags": _purpose_tags(request), "lessons": lessons,
            "sources": sources, "tool_outcomes": outcomes, "outcome": _OUTCOMES[run_data["status"]],
        }
        if not _fresh(entry):
            return None
        old_entries = self.entries
        self.entries = sorted([entry, *(item for item in self.entries if _fresh(item))],
                              key=lambda item: item["created_at"], reverse=True)[:MAX_ENTRIES]
        try:
            self._save()
        except OSError:
            self.entries = old_entries
            raise
        return copy.deepcopy(entry) if any(item["id"] == run_id for item in self.entries) else None
