import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from platetrace import agent
from platetrace.models import RunRequest
from platetrace.providers import ProviderError
from platetrace.store import Run, now


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
