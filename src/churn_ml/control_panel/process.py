from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

import psutil

from src.churn_ml.control_panel.path_safety import canonical_executable
from src.churn_ml.control_panel.schemas import EnvironmentVariableSpec


BASE_ENVIRONMENT_NAMES = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)
IdentityState = Literal["matching", "exited", "mismatched", "unverifiable"]


class ProcessIdentityError(RuntimeError):
    """Raised when a process cannot be identified or signalled safely."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_time_utc: str
    executable_path: str
    argv_sha256: str
    process_group: int | None
    session_id: int | None


@dataclass(frozen=True)
class ProcessCheck:
    state: IdentityState
    diagnostic: str | None = None


@dataclass(frozen=True)
class SpawnedProcess:
    identity: ProcessIdentity

    @property
    def pid(self) -> int:
        return self.identity.pid

    @property
    def process_group(self) -> int | None:
        return self.identity.process_group


class ProcessBackend(Protocol):
    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str],
        redactions: Sequence[str],
    ) -> SpawnedProcess: ...

    def poll(self, identity: ProcessIdentity) -> int | None: ...

    def verify(self, identity: ProcessIdentity) -> ProcessCheck: ...

    def stop(self, identity: ProcessIdentity) -> ProcessCheck: ...


def argv_fingerprint(argv: Sequence[str]) -> str:
    payload = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def build_child_environment(
    declared: Mapping[str, EnvironmentVariableSpec],
    *,
    parent: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    source = os.environ if parent is None else parent
    environment = {
        name: source[name] for name in sorted(BASE_ENVIRONMENT_NAMES) if name in source
    }
    sensitive: list[str] = []
    for name, spec in declared.items():
        value = source.get(name)
        if spec.required and not value:
            raise ProcessIdentityError(
                f"Required environment variable is absent: {name}."
            )
        if value is None:
            continue
        environment[name] = value
        if spec.sensitive and value:
            sensitive.append(value)
    return environment, tuple(sensitive)


class LocalProcessBackend:
    """Cross-platform subprocess backend with stable identity verification."""

    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._log_threads: list[threading.Thread] = []

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str],
        redactions: Sequence[str],
    ) -> SpawnedProcess:
        kwargs: dict[str, object] = {
            "args": list(argv),
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "shell": False,
            "env": dict(environment),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(**kwargs)  # type: ignore[call-overload]
        self._processes[process.pid] = process
        try:
            identity = _capture_identity(process.pid, argv_fingerprint(argv))
        except BaseException:
            process.terminate()
            process.wait(timeout=5)
            raise
        assert process.stdout is not None
        assert process.stderr is not None
        self._start_log_drain(process.stdout, stdout_path, redactions)
        self._start_log_drain(process.stderr, stderr_path, redactions)
        return SpawnedProcess(identity=identity)

    def poll(self, identity: ProcessIdentity) -> int | None:
        process = self._processes.get(identity.pid)
        if process is None:
            return None
        return process.poll()

    def verify(self, identity: ProcessIdentity) -> ProcessCheck:
        return verify_process_identity(identity)

    def stop(self, identity: ProcessIdentity) -> ProcessCheck:
        check = self.verify(identity)
        if check.state != "matching":
            return check
        process = self._processes.get(identity.pid)
        try:
            if os.name == "nt":
                if process is not None:
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    except (OSError, ValueError):
                        process.terminate()
                else:
                    completed = subprocess.run(
                        ["taskkill", "/PID", str(identity.pid), "/T"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=False,
                        env={
                            name: os.environ[name]
                            for name in ("SYSTEMROOT", "WINDIR")
                            if name in os.environ
                        },
                    )
                    if completed.returncode != 0:
                        return ProcessCheck("exited", "Process exited before stop.")
            else:
                assert identity.process_group is not None
                _posix_killpg(identity.process_group, signal.SIGTERM)
        except (ProcessLookupError, psutil.NoSuchProcess):
            return ProcessCheck("exited", "Process exited before stop.")
        except (PermissionError, psutil.AccessDenied, OSError):
            return ProcessCheck("unverifiable", "Process signalling was denied.")
        return ProcessCheck("matching")

    def _start_log_drain(
        self,
        source: object,
        target: Path,
        redactions: Sequence[str],
    ) -> None:
        thread = threading.Thread(
            target=_drain_redacted,
            args=(source, target, tuple(redactions)),
            daemon=True,
        )
        thread.start()
        self._log_threads.append(thread)


def verify_process_identity(identity: ProcessIdentity) -> ProcessCheck:
    try:
        process = psutil.Process(identity.pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return ProcessCheck("exited", "Process has exited.")
        creation_time = _canonical_creation_time(process.create_time())
        if creation_time != identity.creation_time_utc:
            return ProcessCheck("mismatched", "Process creation time does not match.")
        executable = canonical_executable(process.exe())
        if executable != identity.executable_path:
            return ProcessCheck("mismatched", "Process executable does not match.")
        command_line = process.cmdline()
        if not command_line:
            return ProcessCheck("unverifiable", "Process command line is unavailable.")
        if argv_fingerprint(command_line) != identity.argv_sha256:
            return ProcessCheck(
                "mismatched", "Process command identity does not match."
            )
        if os.name != "nt":
            if identity.process_group is None or identity.session_id is None:
                return ProcessCheck(
                    "unverifiable", "Process group identity is missing."
                )
            if _posix_getpgid(identity.pid) != identity.process_group:
                return ProcessCheck("mismatched", "Process group does not match.")
            if _posix_getsid(identity.pid) != identity.session_id:
                return ProcessCheck("mismatched", "Process session does not match.")
    except psutil.NoSuchProcess:
        return ProcessCheck("exited", "Process has exited.")
    except (psutil.AccessDenied, PermissionError):
        return ProcessCheck("unverifiable", "Process identity access was denied.")
    except (OSError, ValueError):
        return ProcessCheck("unverifiable", "Process identity could not be verified.")
    return ProcessCheck("matching")


def _capture_identity(pid: int, fingerprint: str) -> ProcessIdentity:
    try:
        process = psutil.Process(pid)
        creation_time = _canonical_creation_time(process.create_time())
        executable = canonical_executable(process.exe())
        process_group = None if os.name == "nt" else _posix_getpgid(pid)
        session_id = None if os.name == "nt" else _posix_getsid(pid)
    except (psutil.Error, OSError, ValueError) as error:
        raise ProcessIdentityError(
            "Spawned process identity could not be captured."
        ) from error
    return ProcessIdentity(
        pid=pid,
        creation_time_utc=creation_time,
        executable_path=executable,
        argv_sha256=fingerprint,
        process_group=process_group,
        session_id=session_id,
    )


def _canonical_creation_time(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _posix_killpg(process_group: int, sig: signal.Signals) -> None:
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        raise ProcessIdentityError("Process group signalling is unavailable.")
    killpg(process_group, sig)


def _posix_getpgid(pid: int) -> int:
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        raise ProcessIdentityError("Process group identity is unavailable.")
    return int(getpgid(pid))


def _posix_getsid(pid: int) -> int:
    getsid = getattr(os, "getsid", None)
    if getsid is None:
        raise ProcessIdentityError("Process session identity is unavailable.")
    return int(getsid(pid))


def _drain_redacted(source: object, target: Path, redactions: Sequence[str]) -> None:
    stream = source
    secrets = sorted(
        {value.encode("utf-8") for value in redactions if value},
        key=len,
        reverse=True,
    )
    try:
        with target.open("ab", buffering=0) as handle:
            while True:
                chunk = stream.readline()  # type: ignore[attr-defined]
                if not chunk:
                    break
                for secret in secrets:
                    chunk = chunk.replace(secret, b"<redacted>")
                handle.write(chunk)
    finally:
        stream.close()  # type: ignore[attr-defined]
