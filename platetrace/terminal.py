"""A disposable, internet-enabled Docker workspace for each research run.

Commands execute as an unprivileged user in a read-only container. Only its
bounded tmpfs workspace persists between calls; host files and credentials are
never mounted. Call ``cancel_terminal`` when the research run finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field

MAX_OUTPUT_CHARS = 16_000
COMMAND_TIMEOUT = 30
CONTROL_TIMEOUT = 10
MAX_PAYLOAD_BYTES = 2_000_000
DEFAULT_IMAGE = "python:3.12-alpine"
CONTAINER_LIFETIME = 1200
logger = logging.getLogger(__name__)

# This program is fixed application code. User/model commands and evidence only
# enter the container as JSON on stdin, never as part of a host shell command.
_EXEC_WRAPPER = """
import json, os, subprocess, sys
payload = json.load(sys.stdin)
with open('/work/evidence.json', 'w', encoding='utf-8') as handle:
    json.dump(payload['evidence'], handle, ensure_ascii=False)
result = subprocess.run(
    ['/bin/sh', '-c', payload['command']], cwd='/work', stdin=subprocess.DEVNULL,
    env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp', 'LANG': 'C.UTF-8'}
)
sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
""".strip()


@dataclass
class _Session:
    docker: str
    image: str
    name: str = field(default_factory=lambda: "platetrace-" + uuid.uuid4().hex)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started: bool = False
    cancelled: bool = False
    cleanup_pending: bool = False
    process: asyncio.subprocess.Process | None = None


_sessions: dict[str, _Session] = {}


class _Capture:
    def __init__(self) -> None:
        self.data = bytearray()
        self.total = 0

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(8192):
            self.total += len(chunk)
            # Keep draining after reaching the cap so a noisy command cannot
            # block on a full pipe or grow the application's memory unchecked.
            remaining = MAX_OUTPUT_CHARS * 4 - len(self.data)
            if remaining > 0:
                self.data.extend(chunk[:remaining])

    def result(self) -> tuple[str, bool]:
        decoded = self.data.decode("utf-8", errors="replace")
        return decoded[:MAX_OUTPUT_CHARS], self.total > len(self.data) or len(decoded) > MAX_OUTPUT_CHARS


async def _execute(
    *args: str,
    payload: bytes | None = None,
    timeout: float = CONTROL_TIMEOUT,
    session: _Session | None = None,
) -> dict:
    """Run a host CLI without a shell, draining both streams with bounded storage."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if payload is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if session is not None:
        session.process = process
    stdout, stderr = _Capture(), _Capture()
    readers = [asyncio.create_task(stdout.drain(process.stdout)),
               asyncio.create_task(stderr.drain(process.stderr))]

    async def feed() -> None:
        if payload is None:
            return
        try:
            process.stdin.write(payload)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()

    writer = asyncio.create_task(feed())
    timed_out = False
    try:
        if session is not None and session.cancelled and session.started and process.returncode is None:
            process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            if process.returncode is None:
                process.kill()
            await process.wait()
        await asyncio.wait_for(asyncio.gather(writer, *readers), timeout=2)
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        for task in [writer, *readers]:
            if not task.done():
                task.cancel()
        await asyncio.gather(writer, *readers, return_exceptions=True)
        if session is not None and session.process is process:
            session.process = None
    out, out_truncated = stdout.result()
    err, err_truncated = stderr.result()
    return {
        "stdout": out,
        "stderr": err,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout_truncated": out_truncated,
        "stderr_truncated": err_truncated,
    }


def _image() -> str:
    image = os.environ.get("PLATETRACE_TERMINAL_IMAGE", DEFAULT_IMAGE)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}", image):
        raise ValueError("PLATETRACE_TERMINAL_IMAGE must be a Docker image reference.")
    return image


async def terminal_status() -> dict:
    """Check the Docker daemon and locally available image without pulling one."""
    docker = shutil.which("docker")
    if not docker:
        return {"available": False, "reason": "Docker is not installed or is not on PATH."}
    try:
        image = _image()
        daemon = await _execute(docker, "info", "--format", "{{.OSType}}")
        if daemon["exit_code"] != 0 or daemon["timed_out"]:
            return {"available": False, "reason": "Docker is not running or is not accessible."}
        if daemon["stdout"].strip() != "linux":
            return {"available": False, "reason": "The terminal requires Docker running Linux containers."}
        local_image = await _execute(docker, "image", "inspect", image, "--format", "{{.Id}}")
        if local_image["exit_code"] != 0 or local_image["timed_out"]:
            return {
                "available": False,
                "reason": f"Terminal image {image} is not available locally. Run docker pull {image} first.",
            }
    except (OSError, TimeoutError, ValueError) as exc:
        return {"available": False, "reason": f"Terminal unavailable: {exc}"}
    return {
        "available": True,
        "reason": "Internet-enabled isolated Docker terminal; /work persists only for this research run.",
        "image": image,
    }


def _container_args(session: _Session) -> list[str]:
    args = [
        session.docker, "run", "--detach", "--rm", "--pull", "never",
        "--name", session.name, "--label", "app=platetrace",
        "--network", "bridge", "--user", "65534:65534", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "1", "--ulimit", "nofile=256:256", "--init", "--no-healthcheck",
        "--tmpfs", "/work:rw,nosuid,nodev,size=64m,mode=1777",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m,mode=1777",
        "--workdir", "/work", "--entrypoint", "python",
    ]
    # Docker may otherwise inject proxy credentials from the user's Docker
    # configuration. Explicit blank values prevent that implicit propagation.
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY",
                     "http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy"):
        args.extend(["--env", f"{variable}="])
    # Runs have a 15-minute application deadline. This additional lifetime cap
    # lets an otherwise idle container expire if the application crashes.
    args.extend([session.image, "-c", f"import time; time.sleep({CONTAINER_LIFETIME})"])
    return args


async def _settle(task: asyncio.Task):
    """Finish bounded lifecycle work even if cancellation is requested again."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _start_container(session: _Session) -> dict:
    startup = asyncio.create_task(_execute(*_container_args(session), session=session))
    try:
        return await asyncio.shield(startup)
    except asyncio.CancelledError:
        # Killing the CLI immediately can race daemon-side container creation:
        # an early `rm` sees no container, then the daemon creates an orphan.
        # Let the bounded startup operation settle before the caller removes it.
        with contextlib.suppress(OSError, TimeoutError):
            await _settle(startup)
        raise


async def _remove_container(session: _Session) -> bool:
    try:
        result = await _execute(session.docker, "rm", "--force", session.name, timeout=5)
        removed = not result["timed_out"] and (
            result["exit_code"] == 0 or "No such container" in result["stderr"]
        )
    except (OSError, TimeoutError):
        removed = False
    if not removed:
        # Keep the name for the research-finally cleanup retry. A disconnected
        # daemon cannot guarantee removal; do not silently claim it succeeded.
        session.cleanup_pending = True
        session.cancelled = True
        logger.warning("Could not remove terminal container %s; cleanup needs a retry.", session.name)
        return False
    session.cleanup_pending = False
    session.started = False
    return True


def _error(message: str, *, cancelled: bool = False) -> dict:
    return {
        "stdout": "", "stderr": message[:MAX_OUTPUT_CHARS], "exit_code": None,
        "cancelled": cancelled, "stderr_truncated": len(message) > MAX_OUTPUT_CHARS,
    }


async def run_terminal(command: str, evidence: dict, run_id: str) -> dict:
    """Execute arbitrary shell text in this run's isolated, temporary workspace.

    The workspace contains ``evidence.json`` (refreshed each call), Python,
    BusyBox tools including wget, and internet access. No host shell is invoked.
    Only bounded stdout/stderr are exported. A timeout destroys the workspace.
    """
    if not isinstance(command, str) or not command.strip():
        return _error("A nonempty shell command is required.")
    if not isinstance(run_id, str) or not run_id:
        return _error("A research run ID is required.")
    try:
        payload = json.dumps({"command": command, "evidence": evidence}, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return _error("Command and evidence must be JSON-serializable UTF-8 data.")
    if len(payload) > MAX_PAYLOAD_BYTES:
        return _error("Command and evidence exceed the 2 MB terminal input limit.")
    session = _sessions.get(run_id)
    if session is None:
        docker = shutil.which("docker")
        if not docker:
            return _error("Docker is not installed or is not on PATH.")
        try:
            session = _Session(docker=docker, image=_image())
        except ValueError as exc:
            return _error(str(exc))
        _sessions[run_id] = session
    try:
        async with session.lock:
            if session.cancelled:
                return _error("Terminal session was cancelled.", cancelled=True)
            if not session.started:
                start = await _start_container(session)
                if session.cancelled:
                    await _remove_container(session)
                    return _error("Terminal session was cancelled.", cancelled=True)
                if start["exit_code"] != 0 or start["timed_out"]:
                    await _remove_container(session)
                    return _error("Could not start the terminal container: " + start["stderr"])
                session.started = True
            result = await _execute(
                session.docker, "exec", "--interactive", session.name, "python", "-c", _EXEC_WRAPPER,
                payload=payload, timeout=COMMAND_TIMEOUT, session=session,
            )
            if session.cancelled:
                return _error("Terminal session was cancelled.", cancelled=True)
            if result["timed_out"]:
                removed = await _remove_container(session)
                result["error"] = "Command exceeded 30 seconds; " + (
                    "its temporary workspace was destroyed." if removed
                    else "Docker cleanup failed and will be retried when this run ends."
                )
                if not removed:
                    session.cancelled = True
                    result["cleanup_pending"] = True
            result["workspace"] = "/work"
            result["persistent_for_run"] = session.started
            return result
    except asyncio.CancelledError:
        session.cancelled = True
        await _settle(asyncio.create_task(_remove_container(session)))
        raise
    except (OSError, TimeoutError) as exc:
        await _remove_container(session)
        return _error(f"Terminal failed: {exc}")
    finally:
        if not session.started and not session.cleanup_pending and _sessions.get(run_id) is session:
            _sessions.pop(run_id, None)


async def cancel_terminal(run_id: str) -> None:
    """Stop the current command and remove every temporary file for a run."""
    session = _sessions.get(run_id)
    if session is None:
        return
    session.cancelled = True

    async def cleanup() -> None:
        if session.started and session.process is not None and session.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                session.process.kill()
        # Do not kill startup: allow daemon-side creation to settle before rm.
        async with session.lock:
            removed = await _remove_container(session)
            if not removed:
                removed = await _remove_container(session)
            if removed and _sessions.get(run_id) is session:
                _sessions.pop(run_id, None)

    cleanup_task = asyncio.create_task(cleanup())
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        await _settle(cleanup_task)
        raise
