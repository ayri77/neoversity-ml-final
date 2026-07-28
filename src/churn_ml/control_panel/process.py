from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, Sequence


@dataclass(frozen=True)
class SpawnedProcess:
    pid: int
    process_group: int | None


class ProcessBackend(Protocol):
    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> SpawnedProcess: ...

    def poll(self, pid: int) -> int | None: ...

    def pid_exists(self, pid: int) -> bool: ...

    def stop(self, pid: int, process_group: int | None) -> None: ...


class LocalProcessBackend:
    """Small cross-platform subprocess abstraction used by the job manager."""

    def __init__(self) -> None:
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> SpawnedProcess:
        kwargs: dict[str, object] = {
            "args": list(argv),
            "cwd": str(cwd),
            "stdout": stdout,
            "stderr": stderr,
            "stdin": subprocess.DEVNULL,
            "shell": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(**kwargs)  # type: ignore[call-overload]
        self._processes[process.pid] = process
        return SpawnedProcess(
            pid=process.pid,
            process_group=process.pid,
        )

    def poll(self, pid: int) -> int | None:
        process = self._processes.get(pid)
        return None if process is None else process.poll()

    def pid_exists(self, pid: int) -> bool:
        process = self._processes.get(pid)
        if process is not None:
            return process.poll() is None
        if pid <= 0:
            return False
        if os.name == "nt":
            return _windows_pid_exists(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def stop(self, pid: int, process_group: int | None) -> None:
        process = self._processes.get(pid)
        if os.name == "nt":
            if process is not None:
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    return
                except (OSError, ValueError):
                    process.terminate()
                    return
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            return
        target = process_group or pid
        kill_process_group = getattr(os, "killpg")
        try:
            kill_process_group(target, signal.SIGTERM)
        except ProcessLookupError:
            return


def _windows_pid_exists(pid: int) -> bool:
    try:
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except (AttributeError, OSError):
        return False
