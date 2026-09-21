"""Behavioral checks for a bounded research and report-writing budget."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from platetrace import agent
from platetrace.models import RunRequest
from platetrace.store import Run, now


def tool_call(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def report(source_id="S1"):
    return {
        "summary": "The public source describes a blue vehicle.",
        "findings": [{
            "title": "Vehicle specification",
            "detail": "The source describes the vehicle as blue.",
            "source_ids": [source_id],
        }],
        "limitations": ["An individual plate match has not been verified."],
        "next_steps": [],
    }


def tool_names(tools):
    return {item["function"]["name"] for item in tools}


def system_text(messages):
    return "\n".join(item["content"] for item in messages if item["role"] == "system")


@pytest.fixture
def investigation(tmp_path, monkeypatch):
    request = RunRequest(
        plate="abc-123",
        jurisdiction="US / CA",
        authorized=True,
        provider="openrouter",
        api_key="private-api-key",
        max_steps=4,
        use_memory=False,
    )
    data = request.model_dump(exclude={"api_key", "records"})
    data.update(id="b" * 32, created_at=now(), status="queued", events=[], sources=[], report=None)
    run = Run(data, tmp_path)
    cleanup = AsyncMock()
    monkeypatch.setattr(agent, "cancel_terminal", cleanup)
    return run, request, cleanup


def test_default_budget_and_hard_request_bounds():
    arguments = {"plate": "ABC123", "jurisdiction": "US / CA", "authorized": True}
    assert RunRequest(**arguments).max_steps == 24
    for limit in (2, 40):
        assert RunRequest(**arguments, max_steps=limit).max_steps == limit
    for limit in (1, 41):
        with pytest.raises(ValidationError):
            RunRequest(**arguments, max_steps=limit)


@pytest.mark.parametrize("limit", [2, 3, 6, 40])
async def test_report_turns_are_reserved_inside_the_hard_total(investigation, monkeypatch, limit):
    run, request, cleanup = investigation
    request.max_steps = limit
    histories = []
    offered_tools = []

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        offered_tools.append(tool_names(tools))
        return {"role": "assistant", "content": "UNCITED_PRIVATE_PROSE", "tool_calls": []}

    provider = AsyncMock(side_effect=complete)
    monkeypatch.setattr(agent.providers, "complete", provider)
    await agent.research(run, request)

    reserved = min(2, limit - 1)
    assert provider.await_count == limit
    assert all("fetch_url" in names and "finish_report" in names
               for names in offered_tools[:-reserved])
    assert offered_tools[-reserved:] == [{"finish_report"}] * reserved
    budget_messages = [system_text(history) for history in histories]
    assert len(set(budget_messages)) == limit
    for turn, prompt in enumerate(budget_messages, 1):
        assert str(turn) in prompt and str(limit) in prompt
    assert run.data["status"] == "incomplete"
    assert run.data["report"]["findings"] == []
    assert "UNCITED_PRIVATE_PROSE" not in json.dumps(run.data)
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)


@pytest.mark.parametrize("failure", ["missing_citations", "empty_citations", "unknown_source"])
async def test_last_research_evidence_can_be_reported_and_repaired_within_budget(
    investigation, monkeypatch, failure,
):
    run, request, cleanup = investigation
    agent.add_source(run, "Earlier source", "https://vehicles.example/earlier", "Earlier observation")
    invalid_report = report("S2")
    if failure == "missing_citations":
        del invalid_report["findings"][0]["source_ids"]
        invalid_report["summary"] = "REJECTED_REPORT_SECRET"
    elif failure == "empty_citations":
        invalid_report["findings"][0]["source_ids"] = []
        invalid_report["summary"] = "REJECTED_REPORT_SECRET"
    else:
        invalid_report["findings"][0]["source_ids"] = ["invented-source"]
    responses = iter([
        tool_call("search_web", {"query": "public vehicle specification"}),
        tool_call("fetch_url", {"url": "https://vehicles.example/last-source"}),
        tool_call("finish_report", invalid_report),
        tool_call("finish_report", report("S2")),
    ])
    histories = []
    offered_tools = []

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        offered_tools.append(tool_names(tools))
        return next(responses)

    provider = AsyncMock(side_effect=complete)
    search = AsyncMock(return_value={"results": [{"url": "https://vehicles.example/last-source"}]})
    fetch = AsyncMock(return_value={
        "title": "Final research source",
        "url": "https://vehicles.example/last-source",
        "text": "LAST_RESEARCH_EVIDENCE: The vehicle is blue.",
    })
    monkeypatch.setattr(agent.providers, "complete", provider)
    monkeypatch.setattr(agent, "search_web", search)
    monkeypatch.setattr(agent, "fetch_page", fetch)
    await agent.research(run, request)

    assert provider.await_count == request.max_steps == 4
    assert offered_tools[2:] == [{"finish_report"}, {"finish_report"}]
    for history in histories[2:]:
        prompt = system_text(history)
        assert "S1" in prompt and "S2" in prompt
        assert "LAST_RESEARCH_EVIDENCE" in json.dumps(history)
    feedback = json.loads(histories[3][-1]["content"])["error"]
    if failure in ("missing_citations", "empty_citations"):
        assert "findings.0.source_ids" in feedback or "findings[0].source_ids" in feedback
        assert "REJECTED_REPORT_SECRET" not in json.dumps(histories[3])
        assert "REJECTED_REPORT_SECRET" not in json.dumps(run.data)
    else:
        assert "source" in feedback.lower()
    assert run.data["status"] == "completed"
    assert run.data["report"]["findings"][0]["source_ids"] == ["S2"]
    assert [source["id"] for source in run.data["sources"]] == ["S1", "S2"]
    assert request.api_key == ""
    search.assert_awaited_once()
    fetch.assert_awaited_once()
    cleanup.assert_awaited_once_with(run.id)


async def test_report_phase_refuses_tool_calls_the_model_invents(investigation, monkeypatch):
    run, request, cleanup = investigation
    request.max_steps = 3
    request.enable_terminal = True
    forbidden = [
        ("terminal", {"command": "echo should-not-run"}),
        ("fetch_url", {"url": "https://vehicles.example/should-not-fetch"}),
        ("report_issue", {
            "title": "Should not be submitted",
            "summary": "The model tried to keep researching during report writing.",
            "steps_to_reproduce": ["Attempt reporting after the research budget is exhausted."],
            "expected_behavior": "Report writing finishes within the existing turn budget.",
            "actual_behavior": "An issue submission was requested during report writing.",
        }),
    ]
    malicious = {"role": "assistant", "content": "", "tool_calls": [
        tool_call(name, arguments, f"forbidden_{index}")["tool_calls"][0]
        for index, (name, arguments) in enumerate(forbidden)
    ]}
    responses = iter([
        tool_call("fetch_url", {"url": "https://vehicles.example/source"}),
        malicious,
        tool_call("finish_report", report()),
    ])
    histories = []

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        if len(histories) > 1:
            assert tool_names(tools) == {"finish_report"}
        return next(responses)

    provider = AsyncMock(side_effect=complete)
    fetch = AsyncMock(return_value={
        "title": "Public source", "url": "https://vehicles.example/source", "text": "The vehicle is blue.",
    })
    terminal = AsyncMock()
    reporter = AsyncMock()
    monkeypatch.setattr(agent.providers, "complete", provider)
    monkeypatch.setattr(agent, "fetch_page", fetch)
    monkeypatch.setattr(agent, "run_terminal", terminal)
    monkeypatch.setattr(agent.issues, "report_issue", reporter)
    await agent.research(run, request)

    assert provider.await_count == 3
    fetch.assert_awaited_once_with("https://vehicles.example/source")
    terminal.assert_not_awaited()
    reporter.assert_not_awaited()
    denied = [message for message in histories[2]
              if message.get("tool_call_id", "").startswith("forbidden_")]
    assert len(denied) == len(forbidden)
    assert all("error" in json.loads(message["content"]) for message in denied)
    assert run.data["status"] == "completed"
    assert len(run.data["sources"]) == 1
    assert not run.data.get("issue_reports")
    cleanup.assert_awaited_once_with(run.id)


async def test_perpetually_invalid_reports_stop_at_the_hard_cap(investigation, monkeypatch):
    run, request, cleanup = investigation
    agent.add_source(run, "Retained source", "https://vehicles.example/source", "The vehicle is blue.")
    malformed = report()
    malformed["findings"][0]["source_ids"] = []
    provider = AsyncMock(side_effect=lambda *args: tool_call("finish_report", malformed))
    monkeypatch.setattr(agent.providers, "complete", provider)
    await agent.research(run, request)

    assert provider.await_count == request.max_steps
    assert run.data["status"] == "incomplete"
    assert run.data["report"]["findings"] == []
    assert run.data["sources"][0]["id"] == "S1"
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)


@pytest.mark.parametrize("limit", [2, 24])
async def test_early_finish_does_not_spend_reserved_turns(investigation, monkeypatch, limit):
    run, request, cleanup = investigation
    request.max_steps = limit
    final_report = report()
    final_report["findings"] = []
    provider = AsyncMock(return_value=tool_call("finish_report", final_report))
    monkeypatch.setattr(agent.providers, "complete", provider)
    await agent.research(run, request)

    provider.assert_awaited_once()
    assert run.data["status"] == "completed"
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)


async def test_cancellation_during_report_phase_retains_evidence_and_cleans_up(investigation, monkeypatch):
    run, request, cleanup = investigation
    request.max_steps = 3
    started_report = asyncio.Event()
    calls = 0

    async def complete(provider, model, messages, tools, api_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tool_call("fetch_url", {"url": "https://vehicles.example/source"})
        assert tool_names(tools) == {"finish_report"}
        started_report.set()
        await asyncio.Event().wait()

    fetch = AsyncMock(return_value={
        "title": "Retained source", "url": "https://vehicles.example/source", "text": "The vehicle is blue.",
    })
    monkeypatch.setattr(agent.providers, "complete", complete)
    monkeypatch.setattr(agent, "fetch_page", fetch)
    task = asyncio.create_task(agent.research(run, request))
    try:
        await asyncio.wait_for(started_report.wait(), timeout=1)
    finally:
        task.cancel()
        await asyncio.wait_for(task, timeout=1)

    assert calls == 2
    assert run.data["status"] == "cancelled"
    assert run.data["sources"][0]["id"] == "S1"
    assert request.api_key == ""
    cleanup.assert_awaited_once_with(run.id)
    assert run.data["events"][-1]["data"] == {"status": "cancelled"}
