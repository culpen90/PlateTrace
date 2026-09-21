import asyncio
import json
import subprocess
from unittest.mock import Mock

import httpx
import pytest

from platetrace import issues
from platetrace.models import RunRequest
from platetrace.store import Run


@pytest.fixture
def case(tmp_path, monkeypatch):
    for key in issues._SECRET_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PLATETRACE_GITHUB_TOKEN", "github_test_token")
    monkeypatch.delenv("PLATETRACE_GITHUB_REPOSITORY", raising=False)
    issues._recent_submissions.clear()
    request = RunRequest(plate="ABC-123", jurisdiction="US / CA", authorized=True,
                         api_key="private-api-key", enable_terminal=False)
    return Run({"id": "a" * 32, "events": []}, tmp_path), request


@pytest.fixture
def report():
    return {"title": "Report export fails after cancellation", "summary": "Export raises an internal error.",
            "steps_to_reproduce": ["Start a run.", "Cancel, then export its report."],
            "expected_behavior": "Export an incomplete report.", "actual_behavior": "Export fails."}


@pytest.fixture
def mock_http(monkeypatch):
    real_client = httpx.AsyncClient
    requests = []

    def install(handler):
        async def transport(request):
            requests.append(request)
            assert request.url.host == "api.github.com"
            assert request.url.scheme == "https"
            assert request.headers["authorization"] == "Bearer github_test_token"
            response = handler(request)
            return await response if asyncio.iscoroutine(response) else response

        def client(**kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            return real_client(transport=httpx.MockTransport(transport), **kwargs)

        monkeypatch.setattr(issues.httpx, "AsyncClient", client)
        return requests

    return install


def search(items=(), **kwargs):
    return httpx.Response(200, json={"incomplete_results": False, "total_count": len(items),
                                    "items": list(items), **kwargs})


def issue(report, number=42, **kwargs):
    fingerprint = issues._fingerprint(report["title"])
    return {"number": number, "html_url": f"https://github.com/culpen90/PlateTrace/issues/{number}",
            "body": issues._marker(fingerprint), "state": "open", **kwargs}


async def test_create_without_terminal_or_subprocess_and_persist(case, report, mock_http, monkeypatch):
    run, request = case
    forbidden = Mock(side_effect=AssertionError("No shell credentials or terminal access"))
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)

    def handler(req):
        if req.method == "GET":
            query = req.url.params["q"]
            assert "repo:culpen90/PlateTrace" in query and "is:issue" in query
            assert "is:open" not in query and "state:open" not in query
            return search()
        assert req.url.path == "/repos/culpen90/PlateTrace/issues"
        body = json.loads(req.content)
        assert set(body) == {"title", "body"}
        assert "## Steps to reproduce" in body["body"]
        saved = json.loads((run.directory / f"{run.id}.json").read_text())
        assert saved["issue_report_attempts"][0]["status"] == "pending"
        return httpx.Response(201, json=issue(report))

    requests = mock_http(handler)
    result = await issues.report_issue(run, request, report)
    assert result["status"] == "created"
    assert result["state"] == "open"
    assert len(requests) == 2
    assert run.data["issue_reports"] == [result]
    assert run.data["issue_report_attempts"][0]["status"] == "confirmed"
    persisted = (run.directory / f"{run.id}.json").read_text()
    assert "github_test_token" not in persisted and "private-api-key" not in persisted
    forbidden.assert_not_called()


async def test_reuses_closed_duplicate(case, report, mock_http):
    run, request = case
    requests = mock_http(lambda _: search([issue(report, state="closed")]))
    result = await issues.report_issue(run, request, report)
    assert result["status"] == "existing"
    assert result["state"] == "closed"
    assert len(requests) == 1
    assert not run.data.get("issue_report_attempts")


async def test_same_run_and_concurrent_runs_create_once_before_search_index_updates(case, report, mock_http):
    run, request = case
    other = Run({"id": "b" * 32, "events": []}, run.directory)

    async def handler(req):
        await asyncio.sleep(0)
        return search() if req.method == "GET" else httpx.Response(201, json=issue(report))

    requests = mock_http(handler)
    results = await asyncio.gather(issues.report_issue(run, request, report),
                                   issues.report_issue(other, request, report))
    repeated = await issues.report_issue(run, request, {**report, "title": "REPORT export fails after cancellation!"})
    assert [result["status"] for result in results] == ["created", "existing"]
    assert repeated["status"] == "existing"
    assert sum(req.method == "POST" for req in requests) == 1
    assert len(other.data["issue_reports"]) == 1


@pytest.mark.parametrize("code,message", [(401, "authentication"), (403, "permission"),
                                          (404, "repository"), (429, "rate limits"), (500, "could not")])
async def test_api_failures_hide_bodies_and_never_create_after_search_failure(case, report, mock_http, code, message):
    run, request = case
    requests = mock_http(lambda _: httpx.Response(code, text="github_test_token private-api-key"))
    with pytest.raises(issues.IssueReportingError, match=message) as error:
        await issues.report_issue(run, request, report)
    assert "github_test_token" not in str(error.value) and "private-api-key" not in str(error.value)
    assert len(requests) == 1 and requests[0].method == "GET"


@pytest.mark.parametrize("repository", ["https://github.com/a/b", "a/b/c", "../r", "owner/..", "owner/repo?x",
                                        "owner/repo\nX-Test: bad", "", "owner name/repo"])
def test_invalid_repository_is_not_echoed_or_used(case, monkeypatch, repository):
    monkeypatch.setenv("PLATETRACE_GITHUB_REPOSITORY", repository)
    status = issues.issue_reporting_status()
    assert status["available"] is False
    assert status["repository"] == status["url"] == ""


async def test_missing_token_is_actionable_without_network(case, report, monkeypatch):
    run, request = case
    monkeypatch.delenv("PLATETRACE_GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "unrelated_credential_must_not_be_used")
    network = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr(issues.httpx, "AsyncClient", network)
    status = issues.issue_reporting_status()
    assert status["repository"] == "culpen90/PlateTrace"
    assert status["available"] is False
    with pytest.raises(issues.IssueReportingError, match="PLATETRACE_GITHUB_TOKEN"):
        await issues.report_issue(run, request, report)
    network.assert_not_called()


@pytest.mark.parametrize("change", [{"title": " "}, {"title": "x" * 161}, {"summary": "x" * 2001},
                                      {"steps_to_reproduce": []}, {"steps_to_reproduce": [" " ]},
                                      {"steps_to_reproduce": [1]}, {"steps_to_reproduce": ["x"] * 11},
                                      {"actual_behavior": 1}, {"repository": "attacker/other"}])
async def test_invalid_report_inputs_are_not_logged_or_submitted(case, report, monkeypatch, change):
    run, request = case
    network = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr(issues.httpx, "AsyncClient", network)
    with pytest.raises(issues.IssueReportingError):
        await issues.report_issue(run, request, {**report, **change})
    assert run.data == {"id": "a" * 32, "events": []}
    network.assert_not_called()


def test_missing_field_is_rejected_without_echoing_input(case, report):
    _, request = case
    report.pop("summary")
    report["title"] = "private-api-key"
    with pytest.raises(issues.IssueReportingError) as error:
        issues.prepare_issue_report(request, report)
    assert "private-api-key" not in str(error.value)


async def test_privacy_filters_case_identifiers_and_credentials_without_attaching_context(case, report, monkeypatch,
                                                                                       mock_http):
    run, _ = case
    request = RunRequest(plate="ABC-123", jurisdiction="US / CA", authorized=True, api_key="private-api-key",
                         vin="1HGCM82633A004352", objective="PRIVATE_OBJECTIVE_DO_NOT_ATTACH",
                         records=[{"plate": "XYZ987", "jurisdiction": "US / NY", "vin": "2HGCM82633A004352",
                                   "make": "PRIVATE_RECORD_DO_NOT_ATTACH"}])
    monkeypatch.setenv("OPENROUTER_API_KEY", "known_openrouter_secret")
    monkeypatch.setenv("OLLAMA_API_KEY", "known_ollama_secret")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "known_brave_secret")
    secrets = ["private-api-key", "github_test_token", "known_openrouter_secret", "known_ollama_secret",
               "known_brave_secret", "abc 123", "AbC_1.2/3", "x-y-z-9-8-7", "1hgcm82633a004352",
               "2HGCM82633A004352", "ghp_unrecognizedcredential123", "unrelated_bearer_secret",
               "unrelated_password_secret", "query_secret", "url_user:url_password"]
    report["actual_behavior"] = (
        "Observed " + " | ".join(secrets[:11]) + " | Bearer unrelated_bearer_secret | "
        "password=unrelated_password_secret | https://site.example/?api_key=query_secret | "
        "https://url_user:url_password@site.example/"
    )
    prepared = issues.prepare_issue_report(request, report)
    assert issues.prepare_issue_report(request, prepared) == prepared
    requests = mock_http(lambda req: search() if req.method == "GET" else httpx.Response(201, json=issue(prepared)))
    await issues.report_issue(run, request, report)
    sent = requests[-1].content.decode()
    for secret in secrets:
        assert secret not in sent
    assert "PRIVATE_OBJECTIVE_DO_NOT_ATTACH" not in sent
    assert "PRIVATE_RECORD_DO_NOT_ATTACH" not in sent
    assert "[REDACTED" in sent


async def test_post_timeout_never_posts_twice_and_persists_uncertainty(case, report, mock_http):
    run, request = case

    def handler(req):
        if req.method == "GET":
            return search()
        raise httpx.ReadTimeout("github_test_token private-api-key", request=req)

    requests = mock_http(handler)
    with pytest.raises(issues.IssueReportingError, match="timed out"):
        await issues.report_issue(run, request, report)
    assert run.data["issue_report_attempts"][0]["status"] == "uncertain"
    with pytest.raises(issues.IssueReportingError, match="already attempted"):
        await issues.report_issue(run, request, report)
    other = Run({"id": "b" * 32, "events": []}, run.directory)
    with pytest.raises(issues.IssueReportingError, match="already attempted"):
        await issues.report_issue(other, request, report)
    assert sum(req.method == "POST" for req in requests) == 1


async def test_retry_can_confirm_timed_out_submission_after_restart(case, report, mock_http):
    run, request = case
    run.data["issue_report_attempts"] = [{"repository": "culpen90/PlateTrace",
                                        "fingerprint": issues._fingerprint(report["title"]), "status": "pending"}]
    mock_http(lambda _: search([issue(report)]))
    result = await issues.report_issue(run, request, report)
    assert result["status"] == "existing"
    assert run.data["issue_report_attempts"][0]["status"] == "confirmed"


async def test_cancellation_preserves_attempt_and_propagates(case, report, mock_http):
    run, request = case
    started = asyncio.Event()

    async def handler(req):
        if req.method == "GET":
            return search()
        started.set()
        await asyncio.Event().wait()

    mock_http(handler)
    task = asyncio.create_task(issues.report_issue(run, request, report))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.data["issue_report_attempts"][0]["status"] == "uncertain"


@pytest.mark.parametrize("response", [search(incomplete_results=True), search(total_count=1),
                                       httpx.Response(200, text="not json"), httpx.Response(200, json=[]),
                                       search(total_count=True), httpx.Response(302, headers={
                                           "location": "https://attacker.example/stolen"})])
async def test_unreliable_search_or_redirect_never_creates(case, report, mock_http, response):
    run, request = case
    requests = mock_http(lambda _: response)
    with pytest.raises(issues.IssueReportingError):
        await issues.report_issue(run, request, report)
    assert len(requests) == 1


@pytest.mark.parametrize("change", [{"html_url": "https://github.com/attacker/repo/issues/42"},
                                      {"html_url": "https://github.com/culpen90/PlateTrace/issues/42?token=secret"},
                                      {"number": True}, {"body": "No report marker"}, {"body": None},
                                      {"state": "unknown"}, {"pull_request": {}}])
async def test_unverified_duplicate_is_not_reused_or_created(case, report, mock_http, change):
    run, request = case
    requests = mock_http(lambda _: search([issue(report, **change)]))
    with pytest.raises(issues.IssueReportingError, match="verified"):
        await issues.report_issue(run, request, report)
    assert not run.data.get("issue_reports") and len(requests) == 1


async def test_invalid_create_response_is_uncertain_and_not_retried(case, report, mock_http):
    run, request = case
    requests = mock_http(lambda req: search() if req.method == "GET" else httpx.Response(201, json={}))
    with pytest.raises(issues.IssueReportingError, match="verified"):
        await issues.report_issue(run, request, report)
    with pytest.raises(issues.IssueReportingError, match="already attempted"):
        await issues.report_issue(run, request, report)
    assert sum(req.method == "POST" for req in requests) == 1
    assert run.data["issue_report_attempts"][0]["status"] == "uncertain"


async def test_attempt_limit_counts_unsuccessful_posts_but_allows_deduplication(case, report, mock_http):
    run, request = case
    existing = False

    def handler(req):
        if req.method == "GET":
            return search([issue(report)]) if existing else search()
        return httpx.Response(403, json={"message": "private-api-key"})

    requests = mock_http(handler)
    for index in range(3):
        with pytest.raises(issues.IssueReportingError, match="permission"):
            await issues.report_issue(run, request, {**report, "title": f"Export error variant {index}"})
    with pytest.raises(issues.IssueReportingError, match="limit of 3"):
        await issues.report_issue(run, request, report)
    assert sum(req.method == "POST" for req in requests) == 3
    existing = True
    assert (await issues.report_issue(run, request, report))["status"] == "existing"


@pytest.mark.parametrize("plate", ["ID", "REDACTED", "VEHICLE"])
def test_redaction_placeholders_are_idempotent_for_short_plate_values(case, report, plate):
    _, request = case
    request.plate = plate
    report["title"] = f"Export fails for {plate}"
    report["actual_behavior"] = f"{plate}: private-api-key; password=unknown_secret_value"
    once = issues.prepare_issue_report(request, report)
    twice = issues.prepare_issue_report(request, once)
    assert once == twice
    assert issues._fingerprint(once["title"]) == issues._fingerprint(twice["title"])
    assert once["title"] == "Export fails for [REDACTED VEHICLE ID]"


@pytest.mark.parametrize("encoded", ["ABC%20123", "abc%2d123", "ABC%252D123", "ABC+123",
                                        "%41%42%43%2D123"])
def test_encoded_known_identifiers_are_redacted(case, report, encoded):
    _, request = case
    report["actual_behavior"] = f"Observed {encoded} and https://example.com/{encoded}/lookup"
    clean = issues.prepare_issue_report(request, report)
    assert encoded not in clean["actual_behavior"]
    assert clean["actual_behavior"].count("[REDACTED VEHICLE ID]") == 2
    assert issues.prepare_issue_report(request, clean) == clean


@pytest.mark.parametrize("url", [
    "https://example.com/lookup?plate=ABC%2D123&case=private%20case%20data#secret-fragment",
    "https://example.com/lookup#private%20case%20data",
    "https%3A%2F%2Fexample.com%2Flookup%3Fplate%3DABC%252D123%26case%3Dprivate%20case%20data",
    "%68ttps%3A%2F%2Fexample.com%2Flookup%3Fcase%3Dprivate%20case%20data",
])
def test_url_queries_and_fragments_removed_including_encoded_protocols(case, report, url):
    _, request = case
    report["actual_behavior"] = f"Request failed: {url} with an export error."
    clean = issues.prepare_issue_report(request, report)
    assert clean["actual_behavior"] == "Request failed: https://example.com/lookup with an export error."
    assert issues.prepare_issue_report(request, clean) == clean
