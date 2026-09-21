"""Terminal-independent, privacy-filtered GitHub issue reporting."""
import asyncio
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Annotated
from urllib.parse import quote, quote_plus, unquote
from weakref import WeakKeyDictionary

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import RunRequest
from .store import Run, now

DEFAULT_REPOSITORY = "culpen90/PlateTrace"
MAX_SUBMISSION_ATTEMPTS = 3
_CACHE_LIMIT = 256
_SECRET_ENV = ("OPENROUTER_API_KEY", "OLLAMA_API_KEY", "BRAVE_SEARCH_API_KEY", "PLATETRACE_GITHUB_TOKEN")
_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9_.-]{1,100}\Z")
# The server uses one event loop. Per-loop locks also support isolated test/app lifetimes.
_locks: WeakKeyDictionary = WeakKeyDictionary()
_recent_submissions: OrderedDict = OrderedDict()
_REDACTIONS = {"[REDACTED]": "\x00\x01", "[REDACTED VEHICLE ID]": "\x00\x02",
               "[REDACTED CREDENTIAL]": "\x00\x03"}


class IssueReportingError(RuntimeError):
    """An actionable reporting error that never includes remote bodies or credentials."""


class IssueReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=2000)
    steps_to_reproduce: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        min_length=1, max_length=10
    )
    expected_behavior: str = Field(min_length=1, max_length=2000)
    actual_behavior: str = Field(min_length=1, max_length=2000)


def issue_reporting_status() -> dict:
    """Describe configuration without exposing or testing the server-only credential."""
    repository = os.getenv("PLATETRACE_GITHUB_REPOSITORY", DEFAULT_REPOSITORY).strip()
    if not _REPOSITORY.fullmatch(repository) or repository.split("/")[-1] in (".", ".."):
        return {"available": False, "repository": "", "url": "", "reason": (
            "Set PLATETRACE_GITHUB_REPOSITORY to a GitHub owner/repository name."
        )}
    status = {"available": True, "repository": repository,
              "url": f"https://github.com/{repository}/issues", "reason": ""}
    token = os.getenv("PLATETRACE_GITHUB_TOKEN", "").strip()
    if not token:
        status.update(available=False, reason=(
            "Set PLATETRACE_GITHUB_TOKEN on the server with Issues write permission for this repository."
        ))
    elif not re.fullmatch(r"[A-Za-z0-9_]+", token) or len(token) > 500:
        status.update(available=False, reason="PLATETRACE_GITHUB_TOKEN has an invalid format.")
    return status


def _decoded(value: str) -> str:
    # Decode common URL encodings, including double-encoded copied identifiers.
    for _ in range(3):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    return value


def _sanitize(value: str, request: RunRequest) -> str:
    # Query strings and fragments often hold unrelated case data; omit them entirely.
    # Decode each original token as a unit, so encoded query spaces cannot leave
    # private query fragments behind and fully encoded URL schemes are recognized.
    def without_url_details(match):
        decoded = _decoded(match[0])
        url = re.search(r"https?://", decoded, flags=re.IGNORECASE)
        if url:
            return decoded[:url.start()] + re.split(r"[?#]", decoded[url.start():], maxsplit=1)[0]
        return decoded

    value = re.sub(r"[^\s<>\"'()]+", without_url_details, value)
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", value)
    # Temporary non-text markers keep short plates such as ID or REDACTED from
    # matching replacement labels, including on the agent's second validation pass.
    for label, marker in _REDACTIONS.items():
        value = value.replace(label, marker)
    for secret in sorted({request.api_key, *(os.getenv(key, "") for key in _SECRET_ENV)}, key=len, reverse=True):
        if secret:
            for variant in {secret, quote(secret, safe=""), quote_plus(secret), json.dumps(secret)[1:-1]}:
                value = value.replace(variant, _REDACTIONS["[REDACTED]"])
    identifiers = {request.plate, request.vin}
    for record in request.records:
        identifiers.update((record.plate, record.vin))
    for identifier in sorted(identifiers, key=len, reverse=True):
        normalized = re.sub(r"[^A-Za-z0-9]", "", identifier)
        if normalized:
            pattern = r"(?<![A-Za-z0-9])" + r"[\s+._:/-]*".join(map(re.escape, normalized))
            value = re.sub(pattern + r"(?![A-Za-z0-9])", _REDACTIONS["[REDACTED VEHICLE ID]"],
                           value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}|"
                   r"sk-(?:or-v1-)?[A-Za-z0-9_-]{8,})\b", _REDACTIONS["[REDACTED]"], value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+",
                   _REDACTIONS["[REDACTED CREDENTIAL]"], value)
    value = re.sub(r"(?i)\b(?:[A-Z_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|GITHUB_TOKEN)|api[- ]?key|"
                   r"access[- ]?token|token|password|secret|authorization|x-subscription-token)"
                   r"[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)",
                   _REDACTIONS["[REDACTED CREDENTIAL]"], value)
    value = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@",
                   lambda match: match[1] + _REDACTIONS["[REDACTED]"] + "@", value)
    for label, marker in _REDACTIONS.items():
        value = value.replace(marker, label)
    return value.strip()


def prepare_issue_report(request: RunRequest, args: dict) -> dict:
    """Validate and scrub BEFORE model-supplied arguments enter persisted tool events."""
    try:
        report = IssueReport.model_validate(args).model_dump()
        sanitized = {key: [_sanitize(item, request) for item in value] if isinstance(value, list)
                     else _sanitize(value, request) for key, value in report.items()}
        sanitized["title"] = " ".join(sanitized["title"].split())
        return IssueReport.model_validate(sanitized).model_dump()
    except ValidationError:
        raise IssueReportingError(
            "Provide only a nonempty title (up to 160 characters), summary (2000), 1–10 "
            "steps_to_reproduce (1000 each), expected_behavior (2000), and actual_behavior (2000)."
        ) from None


def _fingerprint(title: str) -> str:
    normalized = " ".join(re.findall(r"\w+", title.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _marker(fingerprint: str) -> str:
    return f"<!-- platetrace-issue:{fingerprint} -->"


def _body(report: dict, fingerprint: str) -> str:
    # Only explicitly supplied, sanitized report fields leave the server. Never attach run data.
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(report["steps_to_reproduce"], 1))
    return (f"## Summary\n\n{report['summary']}\n\n## Steps to reproduce\n\n{steps}\n\n"
            f"## Expected behavior\n\n{report['expected_behavior']}\n\n"
            f"## Actual behavior\n\n{report['actual_behavior']}\n\n{_marker(fingerprint)}")


def _json(response: httpx.Response, expected_status: int) -> dict:
    if response.status_code != expected_status:
        if response.status_code == 401:
            message = "GitHub authentication failed. Check PLATETRACE_GITHUB_TOKEN on the server."
        elif response.status_code in (403, 429):
            message = "GitHub refused the request. Check Issues write permission and GitHub rate limits."
        elif response.status_code in (404, 410):
            message = "GitHub repository or issues are unavailable. Check the repository and token access."
        else:
            message = "GitHub could not complete issue reporting. Check the tracker before trying again."
        raise IssueReportingError(message)
    try:
        if len(response.content) > 1_000_000:
            raise ValueError
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError
        return data
    except (TypeError, ValueError, UnicodeError):
        raise IssueReportingError("GitHub returned an invalid issue-reporting response. Check the tracker.") from None


def _verified_issue(data: dict, repository: str, fingerprint: str) -> dict:
    if not isinstance(data, dict):
        raise IssueReportingError("GitHub returned an invalid issue record. Check the tracker.")
    number, url, state, body = (data.get(key) for key in ("number", "html_url", "state", "body"))
    if (type(number) is not int or number < 1 or not isinstance(url, str)
            or url.casefold() != f"https://github.com/{repository}/issues/{number}".casefold()
            or state not in ("open", "closed") or not isinstance(body, str)
            or _marker(fingerprint) not in body or "pull_request" in data):
        raise IssueReportingError("GitHub issue details could not be verified. Check the tracker.")
    return {"number": number, "url": url, "state": state}


def _remember(key: tuple, result: dict | None):
    _recent_submissions[key] = result
    _recent_submissions.move_to_end(key)
    while len(_recent_submissions) > _CACHE_LIMIT:
        _recent_submissions.popitem(last=False)


def _save_result(run: Run, result: dict) -> dict:
    reports = run.data.setdefault("issue_reports", [])
    if not any(item.get("url") == result["url"] for item in reports):
        reports.append(result.copy())
    for attempt in run.data.get("issue_report_attempts", []):
        if (attempt["repository"].casefold() == result["repository"].casefold()
                and attempt["fingerprint"] == result["fingerprint"]):
            attempt["status"] = "confirmed"
            attempt["url"] = result["url"]
    run.save()
    return result


async def report_issue(run: Run, request: RunRequest, args: dict) -> dict:
    """Deduplicate and file an issue directly over HTTPS, independent of terminal access."""
    report = prepare_issue_report(request, args)
    config = issue_reporting_status()
    if not config["available"]:
        raise IssueReportingError(config["reason"])
    repository = config["repository"]
    fingerprint = _fingerprint(report["title"])
    key = (repository.casefold(), fingerprint)
    result = {"repository": repository, "title": report["title"], "fingerprint": fingerprint}
    lock = _locks.setdefault(asyncio.get_running_loop(), asyncio.Lock())
    async with lock:
        cached = _recent_submissions.get(key)
        if cached:
            return _save_result(run, {**cached, "status": "existing"})
        for previous in run.data.get("issue_reports", []):
            if (previous.get("repository", "").casefold() == repository.casefold()
                    and previous.get("fingerprint") == fingerprint):
                return {**previous, "status": "existing"}
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2026-03-10",
                   "Authorization": f"Bearer {os.environ['PLATETRACE_GITHUB_TOKEN'].strip()}",
                   "User-Agent": "PlateTrace-Issue-Reporter"}
        try:
            async with httpx.AsyncClient(base_url="https://api.github.com", headers=headers,
                                         timeout=httpx.Timeout(20, connect=10), trust_env=False,
                                         follow_redirects=False) as client:
                response = await client.get("/search/issues", params={
                    "q": f'repo:{repository} is:issue in:body "platetrace-issue:{fingerprint}"',
                    "per_page": 100,
                })
                search = _json(response, 200)
                items, total = search.get("items"), search.get("total_count")
                if (search.get("incomplete_results") is not False or not isinstance(items, list)
                        or type(total) is not int or total < 0 or total != len(items)):
                    raise IssueReportingError("GitHub duplicate checking was incomplete. No issue was submitted.")
                for item in items:
                    verified = _verified_issue(item, repository, fingerprint)
                    existing = {**result, **verified, "status": "existing"}
                    _remember(key, existing)
                    return _save_result(run, existing)
                attempts = run.data.setdefault("issue_report_attempts", [])
                attempted = any(item.get("fingerprint") == fingerprint
                                and item.get("repository", "").casefold() == repository.casefold()
                                for item in attempts)
                if attempted or key in _recent_submissions:
                    raise IssueReportingError(
                        "This issue submission was already attempted. Check the tracker; automatic resubmission "
                        "is disabled because the previous outcome may be uncertain."
                    )
                if len(attempts) >= MAX_SUBMISSION_ATTEMPTS:
                    raise IssueReportingError("This run reached its limit of 3 new issue submission attempts.")
                attempt = {"repository": repository, "fingerprint": fingerprint,
                           "status": "pending", "attempted_at": now()}
                attempts.append(attempt)
                run.save()  # Record BEFORE POST so cancellation/restarts cannot blindly submit twice.
                _remember(key, None)
                try:
                    response = await client.post(f"/repos/{repository}/issues", json={
                        "title": report["title"], "body": _body(report, fingerprint),
                    })
                    created = {**result, **_verified_issue(_json(response, 201), repository, fingerprint),
                               "status": "created"}
                except BaseException:
                    attempt["status"] = "uncertain"
                    run.save()
                    raise
                _remember(key, created)
                return _save_result(run, created)
        except httpx.HTTPError:
            raise IssueReportingError(
                "GitHub could not be reached or timed out. Check the tracker; an attempted submission "
                "will not be sent again automatically."
            ) from None
