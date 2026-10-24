"""A fresh child process for untrusted parsing work: memory limit, hard kill and crash containment (audit M2).

A parser that never returns, crashes (MuPDF is C code) or allocates without bound must not take the service with
it. ``ChildProcess`` starts ``python -m <module>`` as a new interpreter (never a fork of the service: no lock, thread,
socket or secret of the service is copied into it), writes one request to its stdin and reads one reply from its
stdout. The child gets a minimal environment and its memory is capped before the request is sent:

- Windows: the child is put in a Job Object with a per-process commit limit. Closing the job handle kills every
  process of the job (the child and anything it started), so a child never outlives the service, and a completion
  port reports when an allocation hit the limit. Processes of the job die on an unhandled exception without the
  error-reporting dialog.
- Linux: the parent sets RLIMIT_AS on the child (``prlimit``); the child is a session leader and is killed as a
  process group. An allocation past the limit fails (``MemoryError`` in Python code); a crash of C code that ran out
  of memory is reported as a crash.

``kill`` ends the child at once, so the caller's deadline is a hard one. The blocking ``exchange`` runs on a worker
thread (``app.core.concurrency.ConcurrencyRuntime.run_in_child``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from app.core.logging import log_event

logger = logging.getLogger(__name__)

# The repository root: ``python -m app...`` in the child imports the service's code from here.
APP_ROOT = Path(__file__).resolve().parents[2]
_READ_CHUNK_BYTES = 1024 * 1024
# What the interpreter, the parsers and a document converter need to run. Everything else of the service's
# environment (DATABASE_URL, keys, HMAC secrets) stays out of the child.
_INHERITED_ENVIRONMENT = frozenset({
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "PROGRAMDATA", "PROGRAMFILES", "LOCALAPPDATA", "APPDATA", "USERPROFILE",
    "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ",
})


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """How the child ended: its exit code, the reply bytes it wrote and what the parent saw."""

    returncode: int
    output: bytes
    # The reply grew past ``max_output_bytes``: the child was killed and ``output`` is incomplete.
    output_too_large: bool
    # An allocation of the child failed at the memory limit (Windows; see the module docstring for Linux).
    memory_limit_reached: bool
    # ``kill`` ran (a cancelled caller, an oversized reply or the service stopping).
    killed: bool


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if name.upper() in _INHERITED_ENVIRONMENT}
    environment.update(extra or {})
    return environment


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0000_0100
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0000_0400
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x0000_2000
    _JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount",
            "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimitInformation), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _AssociateCompletionPort(ctypes.Structure):
        _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateIoCompletionPort.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD)
    _kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.GetQueuedCompletionStatus.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD)
    _kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    def _checked(result: int | None) -> int:
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    class _MemoryCap:
        """A Job Object with a per-process commit limit that kills its processes when its handle closes."""

        def __init__(self, memory_limit_bytes: int) -> None:
            self._job = _checked(_kernel32.CreateJobObjectW(None, None))
            self._port: int | None = None
            self._limit_reached = False
            try:
                self._port = _checked(_kernel32.CreateIoCompletionPort(_INVALID_HANDLE_VALUE, None, 0, 1))
                limits = _ExtendedLimitInformation()
                limits.BasicLimitInformation.LimitFlags = (_JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                                           | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                                                           | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
                limits.ProcessMemoryLimit = memory_limit_bytes
                _checked(_kernel32.SetInformationJobObject(self._job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                                           ctypes.byref(limits), ctypes.sizeof(limits)))
                port = _AssociateCompletionPort(None, self._port)
                _checked(_kernel32.SetInformationJobObject(self._job, _JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION,
                                                           ctypes.byref(port), ctypes.sizeof(port)))
            except OSError:
                self.close()
                raise

        def assign(self, process: subprocess.Popen[bytes]) -> None:
            handle = _checked(_kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, process.pid))
            try:
                _checked(_kernel32.AssignProcessToJobObject(self._job, handle))
            finally:
                _kernel32.CloseHandle(handle)

        def kill(self) -> bool:
            """Every process of the job, the child's own children included; False when it could not."""

            return bool(self._job and _kernel32.TerminateJobObject(self._job, 1))

        def limit_reached(self) -> bool:
            if self._port:
                message, key, overlapped = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
                while _kernel32.GetQueuedCompletionStatus(self._port, ctypes.byref(message), ctypes.byref(key),
                                                          ctypes.byref(overlapped), 0):
                    self._limit_reached = self._limit_reached or message.value == _JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT
            return self._limit_reached

        def close(self) -> None:
            for handle in (self._port, self._job):
                if handle:
                    _kernel32.CloseHandle(handle)
            self._port = None
            self._job = 0

    def _kill_tree(process: subprocess.Popen[bytes]) -> None:
        """Without a job: the child and its children by process tree (a venv's python.exe is a launcher that runs
        the interpreter as its own child)."""

        if process.poll() is None:
            subprocess.run(  # noqa: S603 - fixed Windows process-tree command
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],  # noqa: S607 - OS utility
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

    _CREATION_FLAGS = subprocess.CREATE_NO_WINDOW
    _NEW_SESSION = False
else:
    import resource

    class _MemoryCap:
        """RLIMIT_AS on the child (inherited by anything it starts); the child's process group is killed."""

        def __init__(self, memory_limit_bytes: int) -> None:
            self._memory_limit_bytes = memory_limit_bytes

        def assign(self, process: subprocess.Popen[bytes]) -> None:
            resource.prlimit(process.pid, resource.RLIMIT_AS, (self._memory_limit_bytes, self._memory_limit_bytes))

        def kill(self) -> bool:
            return False  # the process group is killed by _kill_tree

        def limit_reached(self) -> bool:
            return False

        def close(self) -> None:
            return None

    def _kill_tree(process: subprocess.Popen[bytes]) -> None:
        if process.returncode is None:  # not reaped yet: its process group id is still its own
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)

    _CREATION_FLAGS = 0
    _NEW_SESSION = True


class ChildProcess:
    """``python -m module`` with a memory cap, waiting on stdin for its one request."""

    def __init__(self, module: str, *, memory_limit_bytes: int, environment: dict[str, str] | None = None) -> None:
        if not sys.executable:
            raise RuntimeError("The interpreter path is unknown: no child process can be started.")
        # -E -s: no PYTHON* variable and no user site directory; -B keeps the parent's choice not to write bytecode.
        command = [sys.executable, "-E", "-s", *(["-B"] if sys.dont_write_bytecode else []), "-m", module]
        self._lock = threading.Lock()
        self._killed = False
        # Made before the child starts so it is assigned right after: a Windows venv's python.exe is a launcher that
        # starts the interpreter as its own child, which joins the job only if the launcher is in it by then.
        try:
            self._cap: _MemoryCap | None = _MemoryCap(memory_limit_bytes)
        except OSError as error:
            self._cap = None
            self._cap_unavailable(error)
        try:
            self._process = subprocess.Popen(  # noqa: S603 - the server's own interpreter and module, no shell
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=APP_ROOT,
                env=child_environment(environment), creationflags=_CREATION_FLAGS, start_new_session=_NEW_SESSION,
            )
        except BaseException:
            if self._cap is not None:
                self._cap.close()
            raise
        if self._cap is not None:
            try:
                self._cap.assign(self._process)
            except OSError as error:
                self._cap.close()
                self._cap = None
                self._cap_unavailable(error)

    @staticmethod
    def _cap_unavailable(error: OSError) -> None:
        # The child still runs apart and killable, without the memory cap (e.g. a host that forbids nested job
        # objects). The cap is part of the hardening, not of the result: the extraction goes on.
        log_event(logger, logging.WARNING, "child_process_memory_cap_unavailable", error_type=type(error).__name__,
                  platform=sys.platform)

    @property
    def pid(self) -> int:
        return self._process.pid

    def exchange(self, request: bytes, *, max_output_bytes: int) -> ChildOutcome:
        """Send the request, read the reply until the child closes its stdout, then reap it (blocking)."""

        stdin, stdout = self._process.stdin, self._process.stdout
        assert stdin is not None and stdout is not None  # noqa: S101 - both are pipes (see __init__)
        with contextlib.suppress(OSError):  # a child that already ended: its exit code says why
            stdin.write(request)
        with contextlib.suppress(OSError):
            stdin.close()
        output, too_large = _read_bounded(stdout, max_output_bytes)
        if too_large:
            self.kill()
        returncode = self._process.wait()
        with self._lock:
            killed = self._killed
        return ChildOutcome(returncode=returncode, output=output, output_too_large=too_large,
                            memory_limit_reached=self._cap.limit_reached() if self._cap is not None else False,
                            killed=killed)

    def kill(self) -> None:
        """End the child and anything it started, now. Safe from any thread and more than once."""

        with self._lock:
            self._killed = True
            tree_killed = self._cap is not None and self._cap.kill()
        if not tree_killed:
            _kill_tree(self._process)
        with contextlib.suppress(OSError):
            self._process.kill()

    def close(self) -> None:
        """Release the pipes and the job; a child still running is killed first."""

        if self._process.poll() is None:
            self.kill()
            self._process.wait()
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        with self._lock:
            if self._cap is not None:
                self._cap.close()
            self._cap = None


def _read_bounded(stream: IO[bytes], max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    while chunk := stream.read(_READ_CHUNK_BYTES):
        size += len(chunk)
        if size > max_bytes:
            return b"", True
        chunks.append(chunk)
    return b"".join(chunks), False


__all__ = ["APP_ROOT", "ChildOutcome", "ChildProcess", "child_environment"]
