import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from platetrace import terminal


def output(stdout="", stderr="", exit_code=0, timed_out=False):
    return {
        "stdout": stdout, "stderr": stderr, "exit_code": exit_code,
        "timed_out": timed_out, "stdout_truncated": False, "stderr_truncated": False,
    }


@pytest.fixture(autouse=True)
def clean_sessions(monkeypatch):
    terminal._sessions.clear()
    monkeypatch.setattr(terminal.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.delenv("PLATETRACE_TERMINAL_IMAGE", raising=False)
    yield
    terminal._sessions.clear()


async def test_terminal_unavailable_without_docker(monkeypatch):
    monkeypatch.setattr(terminal.shutil, "which", lambda name: None)
    assert not (await terminal.terminal_status())["available"]
    result = await terminal.run_terminal("echo test", {}, "test")
    assert result["exit_code"] is None
    assert "not installed" in result["stderr"]


async def test_status_checks_daemon_and_local_image_without_pulling(monkeypatch):
    execute = AsyncMock(side_effect=[output("linux\n"), output("sha256:123\n")])
    monkeypatch.setattr(terminal, "_execute", execute)
    status = await terminal.terminal_status()
    assert status["available"]
    assert "Internet-enabled" in status["reason"]
    assert execute.call_args_list[0].args == ("/usr/bin/docker", "info", "--format", "{{.OSType}}")
    assert execute.call_args_list[1].args == (
        "/usr/bin/docker", "image", "inspect", "python:3.12-alpine", "--format", "{{.Id}}"
    )


@pytest.mark.parametrize("daemon, expected", [
    (output(stderr="denied", exit_code=1), "not running"),
    (output("windows\n"), "Linux containers"),
    (output("linux", timed_out=True), "not running"),
])
async def test_status_daemon_failures(monkeypatch, daemon, expected):
    monkeypatch.setattr(terminal, "_execute", AsyncMock(return_value=daemon))
    status = await terminal.terminal_status()
    assert not status["available"]
    assert expected in status["reason"]


async def test_missing_image_is_reported_without_automatic_pull(monkeypatch):
    execute = AsyncMock(side_effect=[output("linux"), output(exit_code=1)])
    monkeypatch.setattr(terminal, "_execute", execute)
    status = await terminal.terminal_status()
    assert not status["available"]
    assert "docker pull python:3.12-alpine" in status["reason"]
    assert not any("pull" in call.args for call in execute.call_args_list)


@pytest.mark.parametrize("image", ["--privileged", "image;echo bad", "image\n--mount", ""])
async def test_invalid_image_configuration_is_rejected(monkeypatch, image):
    monkeypatch.setenv("PLATETRACE_TERMINAL_IMAGE", image)
    execute = AsyncMock()
    monkeypatch.setattr(terminal, "_execute", execute)
    assert not (await terminal.terminal_status())["available"]
    assert (await terminal.run_terminal("pwd", {}, "run"))["exit_code"] is None
    execute.assert_not_called()


async def test_commands_use_persistent_isolated_workspace_and_stdin(monkeypatch):
    execute = AsyncMock(side_effect=[output("container-id"), output("first"), output("second")])
    monkeypatch.setattr(terminal, "_execute", execute)
    command = "printf '%s' '$(cat /host/secret); `whoami`' > state.txt"
    evidence = {"source": "https://example.com", "text": "quote '\" and newline\n"}
    first = await terminal.run_terminal(command, evidence, "../host-run;$(id)")
    second = await terminal.run_terminal("cat state.txt", evidence, "../host-run;$(id)")
    assert first["stdout"] == "first"
    assert second["stdout"] == "second"
    assert first["persistent_for_run"] and second["persistent_for_run"]
    start_args = execute.call_args_list[0].args
    assert start_args[:2] == ("/usr/bin/docker", "run")
    for option, value in {
        "--network": "bridge", "--pull": "never", "--user": "65534:65534",
        "--cap-drop": "ALL", "--security-opt": "no-new-privileges",
        "--pids-limit": "64", "--memory": "256m", "--memory-swap": "256m",
        "--cpus": "1", "--entrypoint": "python", "--workdir": "/work",
    }.items():
        assert start_args[start_args.index(option) + 1] == value
    assert "--read-only" in start_args and "--rm" in start_args and "--init" in start_args
    assert "/work:rw,nosuid,nodev,size=64m,mode=1777" in start_args
    assert "/tmp:rw,nosuid,nodev,size=16m,mode=1777" in start_args
    assert not {"--volume", "-v", "--mount", "--privileged", "--env-file", "--use-api-socket"} & set(start_args)
    assert "HTTP_PROXY=" in start_args and "https_proxy=" in start_args
    name = start_args[start_args.index("--name") + 1]
    assert name.startswith("platetrace-") and len(name) == 43
    assert "../host-run;$(id)" not in name
    assert command not in start_args
    for call in execute.call_args_list[1:]:
        assert call.args[:4] == ("/usr/bin/docker", "exec", "--interactive", name)
        assert command not in call.args
        assert call.kwargs["timeout"] == 30
    assert json.loads(execute.call_args_list[1].kwargs["payload"]) == {
        "command": command, "evidence": evidence,
    }
    assert execute.call_count == 3


async def test_exit_code_and_streams_are_preserved(monkeypatch):
    execute = AsyncMock(side_effect=[output("id"), output("partial", "failure", exit_code=7)])
    monkeypatch.setattr(terminal, "_execute", execute)
    result = await terminal.run_terminal("exit 7", {}, "run")
    assert result["exit_code"] == 7
    assert result["stdout"] == "partial"
    assert result["stderr"] == "failure"


async def test_timeout_removes_container_and_reports_lost_workspace(monkeypatch):
    execute = AsyncMock(side_effect=[output("id"), output("partial", exit_code=-9, timed_out=True), output()])
    monkeypatch.setattr(terminal, "_execute", execute)
    result = await terminal.run_terminal("sleep 99", {}, "run")
    assert result["timed_out"]
    assert not result["persistent_for_run"]
    assert "destroyed" in result["error"]
    assert execute.call_args_list[-1].args[1:3] == ("rm", "--force")
    assert "run" not in terminal._sessions


async def test_start_failure_cleans_up_and_does_not_exec(monkeypatch):
    execute = AsyncMock(side_effect=[output(stderr="missing image", exit_code=125), output()])
    monkeypatch.setattr(terminal, "_execute", execute)
    result = await terminal.run_terminal("pwd", {}, "run")
    assert "missing image" in result["stderr"]
    assert [call.args[1] for call in execute.call_args_list] == ["run", "rm"]
    assert not terminal._sessions


async def test_cancel_finished_command_removes_persistent_session(monkeypatch):
    execute = AsyncMock(return_value=output())
    monkeypatch.setattr(terminal, "_execute", execute)
    await terminal.run_terminal("pwd", {}, "run")
    session = terminal._sessions["run"]
    await terminal.cancel_terminal("run")
    assert session.cancelled
    assert "run" not in terminal._sessions
    assert execute.call_args_list[-1].args == ("/usr/bin/docker", "rm", "--force", session.name)
    count = execute.call_count
    await terminal.cancel_terminal("missing")
    assert execute.call_count == count


async def test_task_cancellation_always_removes_the_container(monkeypatch):
    running = asyncio.Event()
    calls = []

    async def execute(*args, **kwargs):
        calls.append(args)
        if args[1] == "exec":
            running.set()
            await asyncio.Event().wait()
        return output()

    monkeypatch.setattr(terminal, "_execute", execute)
    task = asyncio.create_task(terminal.run_terminal("sleep 999", {}, "run"))
    await running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls[-1][1:3] == ("rm", "--force")
    assert "run" not in terminal._sessions


async def test_cancel_during_startup_cleans_up_late_container(monkeypatch):
    starting = asyncio.Event()
    release_startup = asyncio.Event()
    startup_process = Mock(returncode=None)
    calls = []

    async def execute(*args, **kwargs):
        calls.append(args)
        if args[1] == "run":
            kwargs["session"].process = startup_process
            starting.set()
            await release_startup.wait()
        return output()

    monkeypatch.setattr(terminal, "_execute", execute)
    task = asyncio.create_task(terminal.run_terminal("pwd", {}, "run"))
    await starting.wait()
    cancellation = asyncio.create_task(terminal.cancel_terminal("run"))
    await asyncio.sleep(0)
    release_startup.set()
    await cancellation
    result = await task
    assert result["cancelled"]
    assert "exec" not in [call[1] for call in calls]
    assert calls[-1][1:3] == ("rm", "--force")
    assert "run" not in terminal._sessions
    startup_process.kill.assert_not_called()


async def test_task_cancellation_waits_for_startup_before_removing(monkeypatch):
    starting = asyncio.Event()
    release_startup = asyncio.Event()
    created = asyncio.Event()
    removed = asyncio.Event()

    async def execute(*args, **kwargs):
        if args[1] == "run":
            starting.set()
            await release_startup.wait()
            created.set()
        if args[1] == "rm":
            assert created.is_set(), "Cleanup must wait for daemon-side creation to finish."
            removed.set()
        return output()

    monkeypatch.setattr(terminal, "_execute", execute)
    task = asyncio.create_task(terminal.run_terminal("pwd", {}, "run"))
    await starting.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not removed.is_set()
    release_startup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert removed.is_set()
    assert not terminal._sessions


async def test_failed_removal_is_retained_for_retry_and_reported(monkeypatch, caplog):
    execute = AsyncMock(side_effect=[
        output("id"), output(timed_out=True, exit_code=-9),
        output(stderr="daemon unavailable", exit_code=1), output(),
    ])
    monkeypatch.setattr(terminal, "_execute", execute)
    result = await terminal.run_terminal("sleep 99", {}, "run")
    assert result["cleanup_pending"]
    assert "cleanup failed" in result["error"]
    assert "run" in terminal._sessions
    assert "cleanup needs a retry" in caplog.text
    await terminal.cancel_terminal("run")
    assert "run" not in terminal._sessions


async def test_already_removed_container_is_successful_cleanup(monkeypatch):
    execute = AsyncMock(side_effect=[output(), output(), output(stderr="No such container: gone", exit_code=1)])
    monkeypatch.setattr(terminal, "_execute", execute)
    await terminal.run_terminal("pwd", {}, "run")
    await terminal.cancel_terminal("run")
    assert not terminal._sessions


async def test_cancelling_cleanup_itself_still_removes_container(monkeypatch):
    removing = asyncio.Event()
    release_removal = asyncio.Event()

    async def execute(*args, **kwargs):
        if args[1] == "rm":
            removing.set()
            await release_removal.wait()
        return output()

    monkeypatch.setattr(terminal, "_execute", execute)
    await terminal.run_terminal("pwd", {}, "run")
    task = asyncio.create_task(terminal.cancel_terminal("run"))
    await removing.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_removal.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not terminal._sessions


@pytest.mark.parametrize("command,evidence,run_id,expected", [
    (" ", {}, "run", "nonempty"),
    ("pwd", {}, "", "run ID"),
    ("pwd", {"bad": object()}, "run", "JSON-serializable"),
    ("x" * 2_000_001, {}, "run", "2 MB"),
])
async def test_invalid_inputs_do_not_start_processes(monkeypatch, command, evidence, run_id, expected):
    execute = AsyncMock()
    monkeypatch.setattr(terminal, "_execute", execute)
    result = await terminal.run_terminal(command, evidence, run_id)
    assert expected in result["stderr"]
    execute.assert_not_called()


class FakeProcess:
    def __init__(self, stdout=b"", stderr=b"", exit_code=0, hang=False):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        self.stdin = Mock(drain=AsyncMock(), wait_closed=AsyncMock())
        self.returncode = None if hang else exit_code
        self.killed = False
        self.done = asyncio.Event()
        if not hang:
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.done.set()

    async def wait(self):
        await self.done.wait()
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.done.set()


async def test_process_capture_drains_excess_output_without_host_shell(monkeypatch):
    process = FakeProcess(stdout=b"A" * 100_000, stderr="\U0001f697".encode() * 20_000, exit_code=9)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", spawn)
    result = await terminal._execute("docker", "exec", "name", payload=b'{"command":"ls"}')
    assert len(result["stdout"]) == terminal.MAX_OUTPUT_CHARS
    assert len(result["stderr"]) == terminal.MAX_OUTPUT_CHARS
    assert result["stdout_truncated"] and result["stderr_truncated"]
    assert result["exit_code"] == 9
    assert not result["timed_out"]
    assert spawn.call_args.args == ("docker", "exec", "name")
    assert "shell" not in spawn.call_args.kwargs
    process.stdin.write.assert_called_once_with(b'{"command":"ls"}')
    assert process.stdout.at_eof() and process.stderr.at_eof()


async def test_process_timeout_kills_child_and_retains_partial_output(monkeypatch):
    process = FakeProcess(stdout=b"before timeout", hang=True)
    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    session = terminal._Session(docker="docker", image=terminal.DEFAULT_IMAGE)
    result = await terminal._execute("docker", "exec", "name", timeout=0.01, session=session)
    assert result["timed_out"] and process.killed
    assert result["stdout"] == "before timeout"
    assert session.process is None


async def test_process_cancellation_kills_child_and_releases_reference(monkeypatch):
    process = FakeProcess(hang=True)
    spawned = asyncio.Event()

    async def spawn(*args, **kwargs):
        spawned.set()
        return process

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", spawn)
    session = terminal._Session(docker="docker", image=terminal.DEFAULT_IMAGE)
    task = asyncio.create_task(terminal._execute("docker", "exec", "name", session=session))
    await spawned.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed
    assert session.process is None
