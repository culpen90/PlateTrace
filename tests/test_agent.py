import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from platetrace import agent
from platetrace.models import RunRequest
from platetrace.providers import ProviderError
from platetrace.store import Run, now
from platetrace.webtools import WebError


def tool_call(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def report(source_id="S1"):
    return {
        "summary": "The fetched source describes the vehicle specification.",
        "findings": [
            {
                "title": "Vehicle detail",
                "detail": "The source describes a blue vehicle.",
                "source_ids": [source_id],
            }
        ],
        "limitations": ["No plate match has been independently established."],
        "next_steps": [],
    }


@pytest.fixture
def investigation(tmp_path, monkeypatch):
    request = RunRequest(
        plate="abc-123",
        jurisdiction="US / CA",
        authorized=True,
        provider="openrouter",
        api_key="private-api-key",
        max_steps=6,
    )
    data = request.model_dump(exclude={"api_key", "records"})
    data.update(id="a" * 32, created_at=now(), status="queued", events=[], sources=[], report=None)
    run = Run(data, tmp_path)
    cleanup = AsyncMock()
    monkeypatch.setattr(agent, "cancel_terminal", cleanup)
    return run, request, cleanup


async def test_autonomous_search_fetch_report_and_citation_recovery(investigation, monkeypatch):
    run, request, cleanup = investigation
    responses = iter(
        [
            tool_call("search_web", {"query": "ABC123 vehicle specs"}),
            tool_call("fetch_url", {"url": "https://vehicles.example/specs"}),
            tool_call("finish_report", report("invented-source")),
            tool_call("finish_report", report()),
        ]
    )
    histories = []

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        assert api_key == "private-api-key"
        assert "private-api-key" not in json.dumps(messages)
        message = next(responses)
        message["reasoning_details"] = [{"data": "PRIVATE_MODEL_REASONING"}]
        return message

    monkeypatch.setattr(agent.providers, "complete", complete)
    search = AsyncMock(return_value={"results": [{"url": "https://vehicles.example/specs"}]})
    fetch = AsyncMock(
        return_value={
            "title": "Public specifications",
            "url": "https://vehicles.example/specs",
            "text": "The vehicle is blue.",
        }
    )
    monkeypatch.setattr(agent, "search_web", search)
    monkeypatch.setattr(agent, "fetch_page", fetch)
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    assert len(run.data["sources"]) == 1
    assert run.data["report"]["findings"][0]["source_ids"] == ["S1"]
    failed_tool = histories[3][-1]
    assert failed_tool["role"] == "tool"
    assert "Every finding must cite" in json.loads(failed_tool["content"])["error"]
    assert "PRIVATE_MODEL_REASONING" in json.dumps(histories[1])
    assert "PRIVATE_MODEL_REASONING" not in json.dumps(run.data)
    assert "private-api-key" not in (run.directory / f"{run.id}.json").read_text()
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)
    search.assert_awaited_once()
    fetch.assert_awaited_once()
    assert run.data["events"][-1]["type"] == "done"


async def test_uncited_final_prose_is_never_published(investigation, monkeypatch):
    run, request, cleanup = investigation
    request.max_steps = 2
    complete = AsyncMock(return_value={"role": "assistant", "content": "FABRICATED_FACT", "tool_calls": []})
    monkeypatch.setattr(agent.providers, "complete", complete)
    await agent.research(run, request)
    assert complete.await_count == 2
    assert run.data["status"] == "incomplete"
    assert run.data["report"]["findings"] == []
    assert "FABRICATED_FACT" not in json.dumps(run.data)
    cleanup.assert_awaited_once_with(run.id)


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ProviderError("Check model permissions."), "Check model permissions."),
        (TimeoutError(), "15 minute deadline"),
        (RuntimeError("sensitive unexpected failure"), "unexpected research error"),
    ],
)
async def test_provider_failure_and_timeout_always_cleanup(investigation, monkeypatch, failure, expected):
    run, request, cleanup = investigation
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=failure))
    await agent.research(run, request)
    assert run.data["status"] == "failed"
    assert expected in run.data["error"]
    assert "sensitive unexpected failure" not in json.dumps(run.data)
    assert request.api_key == ""
    assert run.data["events"][-1]["type"] == "done"
    cleanup.assert_awaited_once_with(run.id)


async def test_cancellation_retains_evidence_clears_key_and_removes_terminal(investigation, monkeypatch):
    run, request, cleanup = investigation
    started = asyncio.Event()

    async def blocked(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.providers, "complete", blocked)
    agent.add_source(run, "Earlier evidence", "https://vehicles.example/evidence", "Earlier finding")
    task = asyncio.create_task(agent.research(run, request))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    await asyncio.wait_for(task, timeout=1)
    assert run.data["status"] == "cancelled"
    assert run.data["sources"][0]["id"] == "S1"
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)
    assert run.data["events"][-1]["data"] == {"status": "cancelled"}


async def test_terminal_receives_only_evidence_and_is_cleaned_up(investigation, monkeypatch):
    run, request, cleanup = investigation
    request.enable_terminal = True
    terminal = AsyncMock(return_value={"stdout": "Found a public source", "exit_code": 0})
    monkeypatch.setattr(agent, "run_terminal", terminal)
    monkeypatch.setattr(
        agent.providers,
        "complete",
        AsyncMock(
            side_effect=[
                tool_call("terminal", {"command": "python -c 'print(1)'"}),
                tool_call(
                    "finish_report",
                    {
                        "summary": "The result is inconclusive.",
                        "findings": [],
                        "limitations": ["No original source was fetched."],
                        "next_steps": [],
                    },
                ),
            ]
        ),
    )
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    evidence = terminal.await_args.args[1]
    assert set(evidence) == {"plate", "jurisdiction", "sources", "records"}
    assert "private-api-key" not in json.dumps(evidence)
    cleanup.assert_awaited_once_with(run.id)


async def test_disabled_terminal_and_unknown_citations_cannot_execute_or_finish(investigation, monkeypatch):
    run, request, _ = investigation
    terminal = AsyncMock()
    monkeypatch.setattr(agent, "run_terminal", terminal)
    with pytest.raises(ValueError, match="disabled"):
        await agent.execute_tool(run, request, "terminal", {"command": "echo test"})
    with pytest.raises(ValueError, match="Every finding must cite"):
        await agent.execute_tool(run, request, "finish_report", report())
    assert run.data["report"] is None
    terminal.assert_not_awaited()


async def test_supplied_records_match_plate_and_jurisdiction_exactly(investigation):
    run, _, _ = investigation
    request = RunRequest(
        plate="abc-123",
        jurisdiction="US / CA",
        authorized=True,
        records=[
            {"plate": "ABC 123", "jurisdiction": "us / ca", "make": "Matched"},
            {"plate": "ABC123", "jurisdiction": "US / NY", "make": "Wrong jurisdiction"},
            {"plate": "ABC124", "jurisdiction": "US / CA", "make": "Wrong plate"},
        ],
    )
    result = await agent.execute_tool(run, request, "read_records", {})
    assert [record["make"] for record in result["records"]] == ["Matched"]
    assert result["source_id"] == "S1"
    assert run.data["sources"][0]["kind"] == "provided"
    assert "Unverified" in result["notice"]


async def test_excessive_model_tool_batch_fails_and_cleans_up(investigation, monkeypatch):
    run, request, cleanup = investigation
    response = tool_call("read_records", {})
    response["tool_calls"] *= 9
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(return_value=response))
    await agent.research(run, request)
    assert run.data["status"] == "failed"
    assert "maximum 8" in run.data["error"]
    assert not any(event["type"] == "tool_start" for event in run.data["events"])
    cleanup.assert_awaited_once_with(run.id)


def issue_arguments(**overrides):
    return {
        "title": "Web reader fails to parse an ordinary JSON source",
        "summary": "The source reader fails consistently while processing a public JSON document.",
        "steps_to_reproduce": ["Fetch a public JSON source using fetch_url.", "Observe the parser error."],
        "expected_behavior": "The reader returns readable source content and a citation identifier.",
        "actual_behavior": "The reader reports a parser error and does not register a source.",
        **overrides,
    }


def inconclusive_report():
    return {
        "summary": "The result is inconclusive because the source reader failed.",
        "findings": [],
        "limitations": ["The original source could not be read."],
        "next_steps": [],
    }


async def test_agent_reports_encountered_issue_without_terminal_and_finishes(investigation, monkeypatch):
    run, request, cleanup = investigation
    monkeypatch.setenv("PLATETRACE_GITHUB_TOKEN", "ISSUE_TOKEN_MUST_NOT_LEAK")
    monkeypatch.setenv("PLATETRACE_GITHUB_REPOSITORY", "example/vehicle-research")
    terminal = AsyncMock()
    monkeypatch.setattr(agent, "run_terminal", terminal)
    monkeypatch.setattr(agent, "fetch_page", AsyncMock(side_effect=WebError("The JSON parser failed.")))
    reported_issue = {
        "status": "created", "url": "https://github.com/example/vehicle-research/issues/17",
        "number": 17, "title": issue_arguments()["title"], "state": "open",
    }

    async def report_issue(received_run, received_request, arguments):
        assert received_run is run and received_request is request
        assert received_request.enable_terminal is False
        assert arguments == issue_arguments()
        received_run.data.setdefault("issue_reports", []).append(reported_issue)
        return reported_issue

    reporter = AsyncMock(side_effect=report_issue)
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    responses = iter([
        tool_call("fetch_url", {"url": "https://vehicles.example/specs.json"}),
        tool_call("report_issue", issue_arguments()),
        tool_call("finish_report", inconclusive_report()),
    ])
    histories = []

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        tool_names = {item["function"]["name"] for item in tools}
        assert "report_issue" in tool_names
        assert "terminal" not in tool_names
        issue_schema = next(item["function"]["parameters"] for item in tools
                            if item["function"]["name"] == "report_issue")
        assert set(issue_schema["required"]) == set(issue_arguments())
        assert issue_schema["additionalProperties"] is False
        context = json.loads(messages[1]["content"])
        assert context["issue_reporting"]["available"] is True
        assert context["issue_reporting"]["repository"] == "example/vehicle-research"
        assert "ISSUE_TOKEN_MUST_NOT_LEAK" not in json.dumps(messages)
        assert "private-api-key" not in json.dumps(messages)
        return next(responses)

    monkeypatch.setattr(agent.providers, "complete", complete)
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    assert run.data["issue_reports"] == [reported_issue]
    assert json.loads(histories[1][-1]["content"])["error"] == "The JSON parser failed."
    assert json.loads(histories[2][-1]["content"])["url"] == reported_issue["url"]
    assert "ISSUE_TOKEN_MUST_NOT_LEAK" not in (run.directory / f"{run.id}.json").read_text()
    reporter.assert_awaited_once()
    terminal.assert_not_awaited()
    cleanup.assert_awaited_once_with(run.id)


@pytest.mark.parametrize("unexpected", [False, True])
async def test_issue_reporting_failure_does_not_abort_research(investigation, monkeypatch, unexpected):
    run, request, _ = investigation
    failure = (RuntimeError("UNEXPECTED_ISSUE_SECRET") if unexpected
               else agent.issues.IssueReportingError("GitHub issue reporting is unavailable."))
    reporter = AsyncMock(side_effect=failure)
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    histories = []
    responses = iter([
        tool_call("report_issue", issue_arguments()),
        tool_call("finish_report", inconclusive_report()),
    ])

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        return next(responses)

    monkeypatch.setattr(agent.providers, "complete", complete)
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    error = json.loads(histories[1][-1]["content"])["error"]
    if not unexpected:
        assert error == "GitHub issue reporting is unavailable."
    else:
        assert error
        assert "UNEXPECTED_ISSUE_SECRET" not in json.dumps(histories)
        assert "UNEXPECTED_ISSUE_SECRET" not in json.dumps(run.data)
    assert not run.data.get("issue_reports")
    reporter.assert_awaited_once()


@pytest.mark.parametrize("arguments", [
    '{"title": "MALFORMED_ISSUE_SECRET"',
    '["MALFORMED_ISSUE_SECRET"]',
    '{"title": {"nested": "MALFORMED_ISSUE_SECRET"}}',
])
async def test_invalid_issue_arguments_are_private_and_recoverable(investigation, monkeypatch, arguments):
    run, request, _ = investigation
    malformed = tool_call("report_issue", {})
    malformed["tool_calls"][0]["function"]["arguments"] = arguments
    reporter = AsyncMock()
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=[
        malformed, tool_call("finish_report", inconclusive_report()),
    ]))
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    issue_results = [event["data"]["result"] for event in run.data["events"]
                     if event["type"] == "tool_result" and event["data"]["name"] == "report_issue"]
    assert len(issue_results) == 1 and "error" in issue_results[0]
    assert "MALFORMED_ISSUE_SECRET" not in json.dumps(run.data)
    assert "MALFORMED_ISSUE_SECRET" not in (run.directory / f"{run.id}.json").read_text()
    reporter.assert_not_awaited()


async def test_issue_arguments_are_sanitized_before_being_logged(investigation, monkeypatch):
    run, request, _ = investigation
    monkeypatch.setenv("PLATETRACE_GITHUB_TOKEN", "ISSUE_TOKEN_MUST_NOT_LEAK")
    arguments = issue_arguments(actual_behavior=(
        "The reader failed for ABC123 using private-api-key and ISSUE_TOKEN_MUST_NOT_LEAK."
    ))
    reporter = AsyncMock(return_value={"status": "existing", "url": "https://github.com/example/repo/issues/3",
                                      "number": 3, "title": arguments["title"], "state": "open"})
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=[
        tool_call("report_issue", arguments), tool_call("finish_report", inconclusive_report()),
    ]))
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    reporter.assert_awaited_once()
    submitted = json.dumps(reporter.await_args.args[2])
    assert "private-api-key" not in submitted and "ISSUE_TOKEN_MUST_NOT_LEAK" not in submitted
    assert "ABC123" not in submitted
    issue_starts = [event["data"]["arguments"] for event in run.data["events"]
                    if event["type"] == "tool_start" and event["data"]["name"] == "report_issue"]
    assert issue_starts == [reporter.await_args.args[2]]
    persisted = (run.directory / f"{run.id}.json").read_text()
    assert "private-api-key" not in persisted and "ISSUE_TOKEN_MUST_NOT_LEAK" not in persisted


async def test_demo_never_invokes_github_issue_reporting(investigation, monkeypatch):
    run, request, _ = investigation
    request.provider = "demo"
    request.enable_terminal = True
    monkeypatch.setenv("PLATETRACE_GITHUB_TOKEN", "configured-demo-token")
    reporter = AsyncMock()
    provider = AsyncMock()
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    monkeypatch.setattr(agent.providers, "complete", provider)
    with pytest.raises(ValueError, match="[Dd]emo"):
        await agent.execute_tool(run, request, "report_issue", issue_arguments())
    await agent.research(run, request)
    assert run.data["status"] == "completed"
    reporter.assert_not_awaited()
    provider.assert_not_awaited()
