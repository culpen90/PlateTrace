"""Behavioral coverage for learning across real agent loops and server restarts."""
import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from platetrace import agent, server
from platetrace.models import RunRequest
from platetrace.providers import ProviderError
from platetrace.store import Run, now

SOURCE_URL = "https://vehicles.example/specifications"
LESSON = "Consult the manufacturer specifications page before broad web searches."


def payload(**overrides):
    return {
        "plate": "abc123",
        "jurisdiction": "US / CA",
        "provider": "ollama",
        "authorized": True,
        "max_steps": 6,
        **overrides,
    }


def tool_call(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def report(*, cited=True, lessons=None):
    result = {
        "summary": "The fetched manufacturer page lists specifications.",
        "findings": ([{
            "title": "Specifications",
            "detail": "A manufacturer specification page was fetched.",
            "source_ids": ["S1"],
        }] if cited else []),
        "limitations": ["A plate match has not been independently established."],
        "next_steps": [],
    }
    if lessons is not None:
        result["research_lessons"] = lessons
    return result


@pytest.fixture(autouse=True)
def isolated_tools(monkeypatch):
    monkeypatch.setattr(agent, "cancel_terminal", AsyncMock())
    monkeypatch.setattr(server, "terminal_status", AsyncMock(return_value={"available": False}))
    monkeypatch.setattr(agent, "fetch_page", AsyncMock(return_value={
        "title": "Manufacturer specifications",
        "url": SOURCE_URL,
        "text": "CURRENT_SOURCE_EXCERPT: vehicle specifications are documented here.",
    }))


@asynccontextmanager
async def application(directory):
    app = server.create_app(directory)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://127.0.0.1:8000",
        ) as client,
    ):
        yield app, client


async def start_run(app, client, **overrides):
    response = await client.post("/api/runs", json=payload(**overrides))
    assert response.status_code == 201, response.text
    return app.state.store.runs[response.json()["id"]]


async def finish_run(app, client, **overrides):
    run = await start_run(app, client, **overrides)
    await asyncio.wait_for(run.task, timeout=2)
    return run


def successful_provider(monkeypatch):
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=[
        tool_call("fetch_url", {"url": SOURCE_URL}),
        tool_call("finish_report", report(lessons=[LESSON])),
    ]))


async def test_completed_run_teaches_later_run_after_restart_without_reusing_evidence(tmp_path, monkeypatch):
    assert RunRequest(**payload()).use_memory is True
    successful_provider(monkeypatch)
    async with application(tmp_path) as (app, client):
        first = await finish_run(app, client)
        assert first.data["status"] == "completed"
        assert first.data["memory"]["saved"] is True
        assert first.data["research_lessons"] == [LESSON]
        assert "research_lessons" not in first.data["report"]
        markdown = await client.get(f"/api/runs/{first.id}/export?format=markdown")
        assert "Research memory" in markdown.text
        assert LESSON in markdown.text
        before_restart = (await client.get("/api/memory")).json()
        assert before_restart["count"] == 1
        assert len(before_restart["entries"]) == 1

    histories = []
    responses = iter([
        # A recalled source must not make S1 valid before this run fetches it.
        tool_call("finish_report", report()),
        tool_call("fetch_url", {"url": SOURCE_URL}),
        tool_call("finish_report", report()),
    ])

    async def complete(provider, model, messages, tools, api_key):
        histories.append(json.loads(json.dumps(messages)))
        return next(responses)

    monkeypatch.setattr(agent.providers, "complete", complete)
    async with application(tmp_path) as (app, client):
        assert (await client.get("/api/memory")).json() == before_restart
        second = await finish_run(app, client, plate="xyz987")
        assert second.data["status"] == "completed"
        recalled = json.loads(histories[0][1]["content"])["research_memory"]
        assert len(recalled) == 1
        assert LESSON in json.dumps(recalled)
        assert SOURCE_URL in json.dumps(recalled)
        assert "CURRENT_SOURCE_EXCERPT" not in json.dumps(recalled)
        assert "untrusted" in histories[0][0]["content"].lower()
        rejected_report = json.loads(histories[1][-1]["content"])
        assert "Every finding must cite" in rejected_report["error"]
        assert [source["id"] for source in second.data["sources"]] == ["S1"]
        assert second.data["report"]["findings"][0]["source_ids"] == ["S1"]
        assert second.data["memory"]["recalled"]
        assert first.data["id"] in json.dumps(second.data["memory"]["recalled"])
        markdown = await client.get(f"/api/runs/{second.id}/export?format=markdown")
        assert first.id in markdown.text
        assert (await client.get("/api/memory")).json()["count"] == 2


@pytest.mark.parametrize("overrides", [{"use_memory": False}, {"provider": "demo"}])
async def test_disabled_and_demo_runs_neither_recall_nor_learn(tmp_path, monkeypatch, overrides):
    async with application(tmp_path) as (app, client):
        recall = Mock(side_effect=AssertionError("Recall must not run"))
        remember = Mock(side_effect=AssertionError("Remember must not run"))
        monkeypatch.setattr(app.state.memory, "recall", recall)
        monkeypatch.setattr(app.state.memory, "remember", remember)
        successful_provider(monkeypatch)
        run = await finish_run(app, client, **overrides)
        assert run.data["status"] == "completed"
        assert run.data["memory"]["enabled"] is False
        assert run.data["memory"]["saved"] is False
        assert not run.data["memory"]["recalled"]
        recall.assert_not_called()
        remember.assert_not_called()
        assert (await client.get("/api/memory")).json()["count"] == 0
        if overrides.get("provider") != "demo":
            messages = agent.providers.complete.await_args_list[0].args[2]
            assert "research_memory" not in json.loads(messages[1]["content"])
            assert not run.data.get("research_lessons")


async def test_cancelled_run_does_not_learn_even_after_fetching_a_source(tmp_path, monkeypatch):
    blocked = asyncio.Event()
    calls = 0

    async def complete(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tool_call("fetch_url", {"url": SOURCE_URL})
        blocked.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.providers, "complete", complete)
    async with application(tmp_path) as (app, client):
        run = await start_run(app, client)
        await asyncio.wait_for(blocked.wait(), timeout=2)
        assert run.data["sources"]
        cancelled = await client.post(f"/api/runs/{run.id}/cancel")
        assert cancelled.status_code == 200
        await asyncio.wait_for(run.task, timeout=2)
        assert run.data["status"] == "cancelled"
        assert run.data["memory"]["saved"] is False
        assert (await client.get("/api/memory")).json()["count"] == 0


@pytest.mark.parametrize("outcome", ["incomplete", "failed"])
async def test_unsuccessful_runs_learn_from_observed_tools(tmp_path, monkeypatch, outcome):
    finish = (ProviderError("The provider is unavailable.") if outcome == "failed"
              else {"role": "assistant", "content": "No report", "tool_calls": []})
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=[
        tool_call("fetch_url", {"url": SOURCE_URL}), finish,
    ]))
    async with application(tmp_path) as (app, client):
        run = await finish_run(app, client, max_steps=2)
        assert run.data["status"] == outcome
        assert run.data["memory"]["saved"] is True
        snapshot = (await client.get("/api/memory")).json()
        assert snapshot["count"] == 1
        assert snapshot["entries"][0]["status"] == outcome
        assert SOURCE_URL in json.dumps(snapshot["entries"])


async def test_clear_during_run_prevents_relearning_and_history_backfill(tmp_path, monkeypatch):
    successful_provider(monkeypatch)
    async with application(tmp_path) as (app, client):
        first = await finish_run(app, client)
        previous = (await client.get("/api/memory")).json()
        blocked, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def complete(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return tool_call("fetch_url", {"url": SOURCE_URL})
            blocked.set()
            await release.wait()
            return tool_call("finish_report", report(lessons=[LESSON]))

        monkeypatch.setattr(agent.providers, "complete", complete)
        running = await start_run(app, client)
        await asyncio.wait_for(blocked.wait(), timeout=2)
        cleared = await client.delete("/api/memory")
        assert cleared.status_code == 200
        snapshot = (await client.get("/api/memory")).json()
        assert snapshot["count"] == 0
        assert snapshot["generation"] != previous["generation"]
        release.set()
        await asyncio.wait_for(running.task, timeout=2)
        assert running.data["status"] == "completed"
        assert running.data["memory"]["saved"] is False
        assert (await client.get("/api/memory")).json()["count"] == 0
        assert (await client.get(f"/api/runs/{first.id}")).status_code == 200

    async with application(tmp_path) as (_, client):
        assert (await client.get("/api/memory")).json()["count"] == 0
        assert len((await client.get("/api/runs")).json()["runs"]) == 2


async def test_queued_run_accepted_before_clear_cannot_repopulate_memory(tmp_path, monkeypatch):
    successful_provider(monkeypatch)
    async with application(tmp_path) as (app, client):
        await finish_run(app, client)
        assert (await client.get("/api/memory")).json()["count"] == 1
        release = asyncio.Event()

        async def deferred_research(*args, **kwargs):
            # Keep research itself unstarted until after its accepted queue slot is cleared.
            await release.wait()
            await agent.research(*args, **kwargs)

        monkeypatch.setattr(server, "research", deferred_research)
        successful_provider(monkeypatch)
        queued = await start_run(app, client)
        assert queued.data["status"] == "queued"
        agent.providers.complete.assert_not_awaited()
        assert (await client.delete("/api/memory")).status_code == 200
        release.set()
        await asyncio.wait_for(queued.task, timeout=2)

        assert queued.data["status"] == "completed"
        assert queued.data["memory"]["saved"] is False
        assert queued.data["memory"]["recalled"] == []
        assert agent.providers.complete.await_count == 2
        messages = agent.providers.complete.await_args_list[0].args[2]
        assert json.loads(messages[1]["content"])["research_memory"] == []
        assert (await client.get("/api/memory")).json()["count"] == 0


@pytest.mark.parametrize("method", ["get", "delete"])
@pytest.mark.parametrize("headers,status", [
    ({"Origin": "https://attacker.example"}, 403),
    ({"Sec-Fetch-Site": "cross-site"}, 403),
    ({"Host": "attacker.example"}, 400),
])
async def test_memory_endpoints_keep_local_access_guards(tmp_path, method, headers, status):
    async with application(tmp_path) as (_, client):
        response = await getattr(client, method)("/api/memory", headers=headers)
        assert response.status_code == status


@pytest.mark.parametrize("operation", ["recall", "remember"])
async def test_memory_failure_is_nonfatal_and_does_not_expose_error_details(tmp_path, monkeypatch, operation):
    successful_provider(monkeypatch)
    async with application(tmp_path) as (app, client):
        failing = Mock(side_effect=OSError("PRIVATE_MEMORY_ERROR_DETAIL"))
        monkeypatch.setattr(app.state.memory, operation, failing)
        run = await finish_run(app, client)
        assert run.data["status"] == "completed"
        assert run.data["report"]["findings"]
        failing.assert_called_once()
        assert run.data["memory"]["saved"] is False
        assert "PRIVATE_MEMORY_ERROR_DETAIL" not in json.dumps(run.data)
        assert any(event["type"] == "memory" for event in run.data["events"])
        assert run.data["events"][-1]["type"] == "done"


async def test_memory_clear_storage_error_is_redacted(tmp_path, monkeypatch):
    async with application(tmp_path) as (app, client):
        monkeypatch.setattr(app.state.memory, "clear", Mock(side_effect=OSError("PRIVATE_MEMORY_PATH")))
        response = await client.delete("/api/memory")
        assert response.status_code == 503
        assert "PRIVATE_MEMORY_PATH" not in response.text
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("lessons", [["one"] * 6, ["x" * 1001], [7], "a string is not a list"])
async def test_invalid_research_lessons_cannot_finalize_report(tmp_path, lessons):
    request = RunRequest(**payload())
    data = request.model_dump(exclude={"api_key", "records"})
    data.update(id="a" * 32, created_at=now(), status="running", events=[], sources=[], report=None)
    run = Run(data, tmp_path)
    with pytest.raises(ValueError):
        await agent.execute_tool(run, request, "finish_report", report(cited=False, lessons=lessons))
    assert run.data["report"] is None
    assert not run.data.get("research_lessons")


async def test_research_lessons_are_scrubbed_before_events_and_memory(tmp_path, monkeypatch):
    secret_lesson = "For ABC123 using PRIVATE_API_KEY, consult manufacturer specifications before searching."
    final = tool_call("finish_report", report(lessons=[secret_lesson]))
    final["reasoning_details"] = [{"data": "PRIVATE_MODEL_REASONING"}]
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=[
        tool_call("fetch_url", {"url": SOURCE_URL}), final,
    ]))
    async with application(tmp_path) as (app, client):
        run = await finish_run(app, client, api_key="PRIVATE_API_KEY")
        assert run.data["status"] == "completed"
        assert run.data["research_lessons"]
        starts = [event["data"]["arguments"] for event in run.data["events"]
                  if event["type"] == "tool_start" and event["data"]["name"] == "finish_report"]
        assert starts[0]["research_lessons"] == run.data["research_lessons"]
        snapshot = (await client.get("/api/memory")).json()
        memory_and_events = json.dumps([snapshot, starts])
        assert "ABC123" not in memory_and_events
        assert "PRIVATE_API_KEY" not in memory_and_events
        assert "PRIVATE_MODEL_REASONING" not in json.dumps(run.data)
        assert "PRIVATE_API_KEY" not in (tmp_path / f"{run.id}.json").read_text()


async def test_run_without_research_observations_does_not_create_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.providers, "complete", AsyncMock(side_effect=ProviderError("Unavailable.")))
    async with application(tmp_path) as (app, client):
        run = await finish_run(app, client)
        assert run.data["status"] == "failed"
        assert run.data["memory"]["saved"] is False
        assert (await client.get("/api/memory")).json()["count"] == 0


async def test_startup_learns_only_safe_legacy_observations_once_and_respects_clear(tmp_path, monkeypatch):
    request = RunRequest(**payload())
    data = request.model_dump(exclude={"api_key", "records", "use_memory"})
    data.update(
        id="a" * 32,
        created_at=now(),
        status="completed",
        objective="LEGACY_OBJECTIVE_MUST_NOT_IMPORT",
        events=[
            {"id": 1, "type": "tool_result", "time": now(), "data": {
                "name": "fetch_url", "result": {"source_id": "S1", "text": "LEGACY_TOOL_BODY"},
            }},
            {"id": 2, "type": "tool_result", "time": now(), "data": {
                "name": "terminal", "result": {"exit_code": 1, "stderr": "LEGACY_ERROR_BODY"},
            }},
        ],
        sources=[{
            "id": "S1", "kind": "web", "retrieved_at": now(), "title": "LEGACY_SOURCE_TITLE",
            "url": "https://vehicles.example/LEGACY_SECRET_PATH?session=LEGACY_SECRET_QUERY#private",
            "excerpt": "LEGACY_SOURCE_EXCERPT",
        }],
        report={"summary": "LEGACY_REPORT_MUST_NOT_IMPORT", "findings": [],
                "limitations": [], "next_steps": []},
        research_lessons=["LEGACY_LESSON_MUST_NOT_IMPORT"],
    )
    Run(data, tmp_path).save()
    Run({**data, "id": "b" * 32, "use_memory": False}, tmp_path).save()
    provider = AsyncMock()
    monkeypatch.setattr(agent.providers, "complete", provider)

    async with application(tmp_path) as (_, client):
        snapshot = (await client.get("/api/memory")).json()
        assert snapshot["count"] == 1
        entry = snapshot["entries"][0]
        assert entry["id"] == data["id"]
        assert [source["url"].rstrip("/") for source in entry["sources"]] == ["https://vehicles.example"]
        assert entry["tool_outcomes"] == {
            "fetch_url": {"success": 1, "failure": 0},
            "terminal": {"success": 0, "failure": 1},
        }
        assert not entry["lessons"]
        assert "LEGACY_" not in json.dumps(snapshot)
        assert "ABC123" not in json.dumps(snapshot)

    async with application(tmp_path) as (_, client):
        assert (await client.get("/api/memory")).json() == snapshot
        assert (await client.delete("/api/memory")).status_code == 200

    async with application(tmp_path) as (_, client):
        assert (await client.get("/api/memory")).json()["count"] == 0
        assert len((await client.get("/api/runs")).json()["runs"]) == 2
    provider.assert_not_awaited()
