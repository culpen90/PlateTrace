"""Loopback-only application server and resumable event stream."""
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import issues
from .agent import research
from .memory import MemoryStore
from .models import RunRequest
from .photos import MAX_PHOTO_REQUEST_BYTES, PhotoError, PhotoRequest, read_plate
from .providers import ProviderError, list_models
from .store import Store, markdown_report, now
from .terminal import terminal_status

STATIC = Path(__file__).parent / "static"


def create_app(data_dir: Path | None = None):
    store = Store(data_dir or Path(os.getenv("PLATETRACE_DATA_DIR", ".platetrace")) / "runs")
    memory = MemoryStore(store.directory / "memory")
    photo_reads = 0

    @asynccontextmanager
    async def lifespan(app):
        store.load()
        memory.load()
        try:
            memory.bootstrap(run.data for run in store.runs.values())
        except OSError:
            # Optional memory must not stop research when local storage is unavailable.
            pass
        yield
        tasks = [run.task for run in store.runs.values() if run.task and not run.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="PlateTrace", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.memory = memory
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        body_limit = MAX_PHOTO_REQUEST_BYTES if request.url.path == "/api/photos/read" else 1_000_000
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed_origin = urlsplit(origin)
                allowed_origin = (parsed_origin.scheme in ("http", "https")
                                  and parsed_origin.netloc == request.headers.get("host"))
            except ValueError:
                allowed_origin = False
            if not allowed_origin:
                return JSONResponse({"detail": "Cross-origin access is disabled."}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site access is disabled."}, status_code=403)
        if (request.headers.get("content-length", "").isdigit()
                and int(request.headers["content-length"]) > body_limit):
            return JSONResponse({"detail": "Request body is too large."}, status_code=413)
        if request.method == "POST":
            # Bound chunked requests as well as requests declaring a length.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > body_limit:
                    return JSONResponse({"detail": "Request body is too large."}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default detail can echo secret-bearing request input.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in exc.errors()
        ]})

    def get_run(run_id):
        if run_id not in store.runs:
            raise HTTPException(404, "Research run not found.")
        return store.runs[run_id]

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/config")
    async def config():
        return {"version": "0.1.0", "providers": [
            {"id": "ollama", "label": "Ollama · local", "default_model": os.getenv("OLLAMA_MODEL", "qwen3:8b")},
            {"id": "openrouter", "label": "OpenRouter · cloud", "default_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")},
            {"id": "demo", "label": "Demo · synthetic fixture", "default_model": "demo-fixture"},
        ], "defaults": {"provider": "ollama", "model": os.getenv("OLLAMA_MODEL", "qwen3:8b")},
            "terminal": await terminal_status(),
            "issue_reporting": issues.issue_reporting_status(),
            "search": {"available": True, "provider": "Brave" if os.getenv("BRAVE_SEARCH_API_KEY") else "DuckDuckGo (best effort)"},
            "allowed_domains": [], "limits": {"max_steps": 40},
            "openrouter_key_configured": bool(os.getenv("OPENROUTER_API_KEY"))}

    @app.get("/api/models")
    async def models(provider: str):
        if provider == "demo":
            return {"models": [{"id": "demo-fixture", "name": "Synthetic demo fixture"}]}
        if provider not in ("ollama", "openrouter"):
            raise HTTPException(400, "Unknown provider.")
        try:
            return {"models": await list_models(provider)}
        except ProviderError as exc:
            return {"models": [], "error": str(exc)}

    @app.get("/api/memory")
    async def memory_detail():
        return memory.snapshot()

    @app.delete("/api/memory")
    async def clear_memory():
        count = memory.snapshot()["count"]
        try:
            memory.clear()
        except OSError:
            raise HTTPException(503, "Research memory could not be cleared. Check local data storage and retry.") from None
        return {**memory.snapshot(), "cleared": count}

    @app.post("/api/runs", status_code=201)
    async def start_run(request: RunRequest):
        if sum(run.data["status"] in ("queued", "running", "cancelling") for run in store.runs.values()) >= 2:
            raise HTTPException(429, "Two investigations are already running. Stop or finish one before starting another.")
        if request.provider == "openrouter" and not (request.api_key.strip() or os.getenv("OPENROUTER_API_KEY")):
            raise HTTPException(400, "Enter an OpenRouter API key or set OPENROUTER_API_KEY in .env.")
        if request.enable_terminal and request.provider != "demo":
            status = await terminal_status()
            if not status["available"]:
                raise HTTPException(400, status["reason"])
        run_id = uuid.uuid4().hex
        data = request.model_dump(exclude={"api_key", "records"})
        data.update({"id": run_id, "created_at": now(), "status": "queued", "events": [], "sources": [],
                     "issue_reports": [], "report": None})
        run = store.add(data)
        run.task = asyncio.create_task(
            research(run, request, memory, memory_generation=memory.generation), name=f"research-{run_id}",
        )
        return {"id": run_id}

    @app.post("/api/photos/read")
    async def read_photo(request: PhotoRequest):
        nonlocal photo_reads
        if photo_reads >= 2:
            raise HTTPException(429, "Two photos are already being read. Wait for one to finish and try again.")
        photo_reads += 1
        try:
            return await read_plate(request)
        except PhotoError as exc:
            raise HTTPException(422, str(exc)) from None
        except ProviderError as exc:
            raise HTTPException(502, str(exc)) from None
        finally:
            photo_reads -= 1

    @app.get("/api/runs")
    async def list_runs():
        fields = ("id", "plate", "jurisdiction", "provider", "model_id", "status", "created_at")
        return {"runs": [{key: run.data[key] for key in fields}
                         for run in reversed(list(store.runs.values()))]}

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        return get_run(run_id).data

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str):
        run = get_run(run_id)
        if run.task and not run.task.done() and run.data["status"] in ("queued", "running"):
            await run.status("cancelling")
            run.task.cancel()
        return {"id": run_id, "status": run.data["status"]}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        run = get_run(run_id)
        try:
            cursor = max(0, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            cursor = 0

        async def stream():
            nonlocal cursor
            while True:
                pending = []
                async with run.changed:
                    pending = [event for event in run.data["events"] if event["id"] > cursor]
                    if not pending and run.data["status"] in ("queued", "running", "cancelling"):
                        try:
                            await asyncio.wait_for(run.changed.wait(), timeout=15)
                        except TimeoutError:
                            pass
                        pending = [event for event in run.data["events"] if event["id"] > cursor]
                for event in pending:
                    cursor = event["id"]
                    yield f"id: {cursor}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
                if run.data["status"] not in ("queued", "running", "cancelling"):
                    if not any(event["type"] == "done" for event in pending):
                        yield f"event: done\ndata: {json.dumps({'type': 'done', 'data': {'status': run.data['status']}})}\n\n"
                    break
                if not pending:
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    break

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/export")
    async def export(run_id: str, format: str = "json"):
        data = get_run(run_id).data
        if format not in ("json", "markdown"):
            raise HTTPException(400, "Export format must be json or markdown.")
        if format == "json":
            content, extension, mime = json.dumps(data, indent=2, ensure_ascii=False), "json", "application/json"
        else:
            content, extension, mime = markdown_report(data), "md", "text/markdown"
        return Response(content, media_type=mime, headers={"Content-Disposition": f'attachment; filename="platetrace-{run_id}.{extension}"'})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
