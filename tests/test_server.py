import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from platetrace import agent, server
from platetrace.models import RunRequest
from platetrace.store import now


def payload(**overrides):
    return {
        "plate": "abc123",
        "jurisdiction": "US / CA",
        "provider": "demo",
        "model_id": "demo-fixture",
        "authorized": True,
        **overrides,
    }


@pytest.fixture
async def application(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        server, "terminal_status", AsyncMock(return_value={"available": False, "reason": "No Docker"})
    )
    monkeypatch.setattr(agent, "cancel_terminal", AsyncMock())
    app = server.create_app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://127.0.0.1:8000",
        ) as client,
    ):
        yield app, client, tmp_path


async def test_create_demo_replay_sse_and_export_are_consistent_and_secret_free(application):
    app, client, directory = application
    response = await client.post("/api/runs", json=payload(api_key="not-for-storage"))
    assert response.status_code == 201
    run_id = response.json()["id"]
    await asyncio.wait_for(app.state.store.runs[run_id].task, timeout=2)
    detail = await client.get(f"/api/runs/{run_id}")
    data = detail.json()
    assert data["plate"] == "ABC123"
    assert data["status"] == "completed"
    assert data["sources"][0]["kind"] == "demo"
    assert "synthetic" in json.dumps(data).lower()
    assert "not-for-storage" not in detail.text
    assert "not-for-storage" not in (directory / f"{run_id}.json").read_text()
    history = (await client.get("/api/runs")).json()["runs"]
    assert history[0]["id"] == run_id

    first_event_id = data["events"][0]["id"]
    stream = await client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(first_event_id)})
    assert stream.status_code == 200 and "text/event-stream" in stream.headers["content-type"]
    assert f"id: {first_event_id}\n" not in stream.text
    assert "event: report" in stream.text and "event: done" in stream.text
    assert stream.headers["x-accel-buffering"] == "no"
    completed_stream = await client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "99999"})
    assert "event: done" in completed_stream.text
    assert "event: report" not in completed_stream.text

    exported = await client.get(f"/api/runs/{run_id}/export?format=json")
    assert exported.json()["report"] == data["report"]
    markdown = await client.get(f"/api/runs/{run_id}/export?format=markdown")
    assert "Sources: S1" in markdown.text and "Synthetic demonstration record" in markdown.text
    assert "attachment; filename=" in markdown.headers["content-disposition"]
    assert (await client.get(f"/api/runs/{run_id}/export?format=html")).status_code == 400


async def test_turn_budget_config_matches_request_schema_and_default(application):
    app, client, _ = application
    config = (await client.get("/api/config")).json()
    schema = RunRequest.model_json_schema()["properties"]["max_steps"]
    assert config["defaults"]["max_steps"] == schema["default"] == 24
    assert config["limits"]["max_steps"] == schema["maximum"] == 40
    assert config["limits"]["report_turns"] == 2
    assert schema["minimum"] == 2

    response = await client.post("/api/runs", json=payload())
    assert response.status_code == 201
    run = app.state.store.runs[response.json()["id"]]
    await asyncio.wait_for(run.task, timeout=2)
    assert run.data["max_steps"] == config["defaults"]["max_steps"]


@pytest.mark.parametrize("max_steps", [2, 40])
async def test_turn_budget_accepts_supported_boundaries(application, max_steps):
    app, client, _ = application
    response = await client.post("/api/runs", json=payload(max_steps=max_steps))
    assert response.status_code == 201
    run = app.state.store.runs[response.json()["id"]]
    await asyncio.wait_for(run.task, timeout=2)
    assert run.data["max_steps"] == max_steps


@pytest.mark.parametrize(
    "overrides",
    [
        {"authorized": False},
        {"plate": "../../etc/passwd"},
        {"vin": "INVALID"},
        {"objective": "find the owner's address"},
        {"source_urls": ["file:///etc/passwd"]},
        {"source_urls": ["https://user:password@example.com"]},
        {"max_steps": 1},
        {"max_steps": 41},
    ],
)
async def test_validation_rejects_unsupported_input_without_echoing_keys(application, overrides):
    _, client, _ = application
    response = await client.post("/api/runs", json=payload(api_key="KEY_MUST_NOT_LEAK", **overrides))
    assert response.status_code == 422
    assert "KEY_MUST_NOT_LEAK" not in response.text
    assert "input" not in response.json()["detail"][0]


async def test_provider_and_terminal_preflight_validation(application):
    _, client, _ = application
    missing_key = await client.post("/api/runs", json=payload(provider="openrouter"))
    assert missing_key.status_code == 400 and "API key" in missing_key.text
    no_terminal = await client.post("/api/runs", json=payload(provider="ollama", enable_terminal=True))
    assert no_terminal.status_code == 400 and "No Docker" in no_terminal.text
    assert (await client.get("/api/models?provider=invalid")).status_code == 400
    assert (await client.get("/api/runs/missing")).status_code == 404


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Origin": "https://attacker.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Host": "attacker.example"}, 400),
        ({"Origin": "http://127.0.0.1:8000"}, 200),
        ({"Origin": "http://[bad"}, 403),
    ],
)
async def test_host_and_origin_guards(application, headers, expected):
    _, client, _ = application
    response = await client.get("/api/health", headers=headers)
    assert response.status_code == expected


async def test_security_headers_config_and_model_errors_do_not_expose_credentials(application, monkeypatch):
    _, client, _ = application
    monkeypatch.setenv("OPENROUTER_API_KEY", "ENV_KEY_MUST_NOT_LEAK")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BRAVE_KEY_MUST_NOT_LEAK")
    config = await client.get("/api/config")
    assert config.json()["openrouter_key_configured"] is True
    assert config.json()["search"]["provider"] == "Brave"
    assert "MUST_NOT_LEAK" not in config.text
    assert config.headers["cache-control"] == "no-store"
    assert config.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in config.headers["content-security-policy"]
    monkeypatch.setattr(server, "list_models", AsyncMock(side_effect=server.ProviderError("Start Ollama.")))
    models = await client.get("/api/models?provider=ollama")
    assert models.json() == {"models": [], "error": "Start Ollama."}


async def test_issue_reporting_configuration_is_independent_of_terminal_and_secret_free(application, monkeypatch):
    _, client, _ = application
    monkeypatch.setenv("PLATETRACE_GITHUB_REPOSITORY", "example/vehicle-research")
    monkeypatch.setenv("PLATETRACE_GITHUB_TOKEN", "GITHUB_TOKEN_MUST_NOT_LEAK")
    configured = await client.get("/api/config")
    assert configured.status_code == 200
    config = configured.json()
    assert config["terminal"]["available"] is False
    assert config["issue_reporting"]["available"] is True
    assert config["issue_reporting"]["repository"] == "example/vehicle-research"
    assert config["issue_reporting"]["url"] == "https://github.com/example/vehicle-research/issues"
    assert "GITHUB_TOKEN_MUST_NOT_LEAK" not in configured.text

    monkeypatch.delenv("PLATETRACE_GITHUB_TOKEN")
    missing_token = await client.get("/api/config")
    assert missing_token.json()["issue_reporting"]["available"] is False
    assert missing_token.json()["issue_reporting"]["reason"]
    assert missing_token.json()["terminal"]["available"] is False


async def test_reported_issue_links_survive_api_exports_and_history_reload(application):
    app, client, directory = application
    response = await client.post("/api/runs", json=payload())
    run_id = response.json()["id"]
    run = app.state.store.runs[run_id]
    await asyncio.wait_for(run.task, timeout=2)
    issues = [{"status": "created", "url": "https://github.com/example/vehicle-research/issues/17",
               "number": 17, "title": "Source reader fails for JSON", "state": "open"}]
    run.data["issue_reports"] = issues
    run.save()

    detail = await client.get(f"/api/runs/{run_id}")
    exported = await client.get(f"/api/runs/{run_id}/export?format=json")
    markdown = await client.get(f"/api/runs/{run_id}/export?format=markdown")
    assert detail.json()["issue_reports"] == issues
    assert exported.json()["issue_reports"] == issues
    assert issues[0]["url"] in markdown.text
    assert "#17" in markdown.text
    assert "created" in markdown.text

    reloaded = server.create_app(directory)
    async with reloaded.router.lifespan_context(reloaded):
        assert reloaded.state.store.runs[run_id].data["issue_reports"] == issues


async def test_request_body_size_is_enforced_for_declared_and_chunked_bodies(application):
    _, client, _ = application
    declared = await client.post("/api/runs", content=b"x" * 1_000_001)
    assert declared.status_code == 413

    async def chunks():
        yield b"x" * 600_000
        yield b"x" * 600_000

    chunked = await client.post("/api/runs", content=chunks(), headers={"Content-Type": "application/json"})
    assert chunked.status_code == 413


def photo_payload(**overrides):
    return {
        "provider": "ollama", "model_id": "vision-model", "image_data_url": "data:image/png;base64,AAAA",
        **overrides,
    }


async def test_photo_read_is_transient_and_does_not_start_research(application, monkeypatch):
    app, client, directory = application
    result = {"plate": "ABC123", "jurisdiction": "California, US", "warnings": []}
    reader = AsyncMock(return_value=result)
    monkeypatch.setattr(server, "read_plate", reader)
    response = await client.post("/api/photos/read", json=photo_payload(api_key="PHOTO_KEY_NOT_FOR_STORAGE"))
    assert response.status_code == 200
    assert response.json() == result
    assert reader.await_args.args[0].api_key == "PHOTO_KEY_NOT_FOR_STORAGE"
    assert "PHOTO_KEY_NOT_FOR_STORAGE" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert not app.state.store.runs
    assert not list(directory.glob("*.json"))


@pytest.mark.parametrize("overrides", [{"provider": "demo"}, {"model_id": ""}, {"unexpected": "value"}])
async def test_photo_validation_does_not_echo_credentials_or_image(application, overrides):
    _, client, _ = application
    response = await client.post("/api/photos/read", json=photo_payload(
        api_key="PHOTO_SECRET", image_data_url="PRIVATE_PHOTO_BYTES", **overrides,
    ))
    assert response.status_code == 422
    assert "PHOTO_SECRET" not in response.text
    assert "PRIVATE_PHOTO_BYTES" not in response.text


@pytest.mark.parametrize("error,status", [
    (server.PhotoError("Choose a readable JPEG, PNG, or WebP photo."), 422),
    (server.ProviderError("Choose a vision-capable model."), 502),
])
async def test_photo_errors_are_actionable(application, monkeypatch, error, status):
    _, client, _ = application
    monkeypatch.setattr(server, "read_plate", AsyncMock(side_effect=error))
    response = await client.post("/api/photos/read", json=photo_payload())
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


async def test_larger_body_limit_is_scoped_to_photo_route(application, monkeypatch):
    _, client, _ = application
    monkeypatch.setattr(server, "read_plate", AsyncMock(return_value={"plate": "", "jurisdiction": "", "warnings": []}))
    data = photo_payload(image_data_url="data:image/png;base64," + "A" * 1_000_004)
    assert (await client.post("/api/photos/read", json=data)).status_code == 200
    assert (await client.post("/api/runs", json=data)).status_code == 413
    oversized = await client.post("/api/photos/read", content=b"x" * (server.MAX_PHOTO_REQUEST_BYTES + 1))
    assert oversized.status_code == 413

    async def chunks():
        yield b"x" * (server.MAX_PHOTO_REQUEST_BYTES // 2)
        yield b"x" * (server.MAX_PHOTO_REQUEST_BYTES // 2 + 1)

    chunked = await client.post("/api/photos/read", content=chunks(), headers={"Content-Type": "application/json"})
    assert chunked.status_code == 413


async def test_photo_concurrency_limit_releases_after_error(application, monkeypatch):
    _, client, _ = application
    both_started, release = asyncio.Event(), asyncio.Event()
    started = 0

    async def wait_for_release(request):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        raise server.ProviderError("Try another vision model.")

    monkeypatch.setattr(server, "read_plate", wait_for_release)
    requests = [asyncio.create_task(client.post("/api/photos/read", json=photo_payload())) for _ in range(2)]
    try:
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert (await client.post("/api/photos/read", json=photo_payload())).status_code == 429
    finally:
        release.set()
        responses = await asyncio.gather(*requests)
    assert all(response.status_code == 502 for response in responses)
    assert (await client.post("/api/photos/read", json=photo_payload())).status_code == 502


async def test_photo_upload_rejects_cross_origin_requests_before_reading(application, monkeypatch):
    _, client, _ = application
    reader = AsyncMock()
    monkeypatch.setattr(server, "read_plate", reader)
    response = await client.post("/api/photos/read", json=photo_payload(), headers={"Origin": "https://example.com"})
    assert response.status_code == 403
    reader.assert_not_awaited()


async def test_concurrent_run_limit_and_cancel_cleanup(application, monkeypatch):
    app, client, _ = application
    started = asyncio.Event()

    async def wait_for_cancel(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.providers, "complete", wait_for_cancel)
    ids = []
    for _ in range(2):
        response = await client.post("/api/runs", json=payload(provider="ollama"))
        assert response.status_code == 201
        ids.append(response.json()["id"])
    await asyncio.wait_for(started.wait(), timeout=1)
    assert (await client.post("/api/runs", json=payload())).status_code == 429
    cancelled = await client.post(f"/api/runs/{ids[0]}/cancel")
    assert cancelled.status_code == 200
    await asyncio.wait_for(app.state.store.runs[ids[0]].task, timeout=1)
    assert app.state.store.runs[ids[0]].data["status"] == "cancelled"
    assert (await client.post(f"/api/runs/{ids[0]}/cancel")).json()["status"] == "cancelled"
    agent.cancel_terminal.assert_any_await(ids[0])
    await client.post(f"/api/runs/{ids[1]}/cancel")
    await asyncio.wait_for(app.state.store.runs[ids[1]].task, timeout=1)
    agent.cancel_terminal.assert_any_await(ids[1])


async def test_server_shutdown_cancels_inflight_research_and_cleans_terminal(tmp_path, monkeypatch):
    entered = asyncio.Event()
    cleanup = AsyncMock()

    async def wait_for_shutdown(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.providers, "complete", wait_for_shutdown)
    monkeypatch.setattr(agent, "cancel_terminal", cleanup)
    app = server.create_app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client,
    ):
        response = await client.post("/api/runs", json=payload(provider="ollama"))
        run_id = response.json()["id"]
        await asyncio.wait_for(entered.wait(), timeout=1)
        run = app.state.store.runs[run_id]
        assert not run.task.done()
    assert run.task.done()
    assert run.data["status"] == "cancelled"
    assert run.data["events"][-1]["type"] == "done"
    cleanup.assert_awaited_once_with(run_id)


async def test_history_startup_recovers_interrupted_runs_and_skips_invalid_files(tmp_path):
    valid_id = "b" * 32
    good = {
        **payload(),
        "id": valid_id,
        "created_at": now(),
        "status": "running",
        "events": [],
        "sources": [],
        "report": None,
    }
    (tmp_path / f"{valid_id}.json").write_text(json.dumps(good))
    (tmp_path / "broken.json").write_text("not json")
    (tmp_path / "traversal.json").write_text(json.dumps({"id": "../bad", "status": "completed"}))
    (tmp_path / "incomplete.json").write_text(json.dumps({"id": "c" * 32, "status": "completed"}))
    app = server.create_app(tmp_path)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://localhost"
        ) as client,
    ):
        response = await client.get("/api/runs")
        assert response.status_code == 200
        runs = response.json()["runs"]
        assert len(runs) == 1
        assert runs[0]["id"] == valid_id and runs[0]["status"] == "interrupted"
        detail = (await client.get(f"/api/runs/{valid_id}")).json()
        assert "server stopped" in detail["error"]
    saved = json.loads((tmp_path / f"{valid_id}.json").read_text())
    assert saved["status"] == "interrupted"
