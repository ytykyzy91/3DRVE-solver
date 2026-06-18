from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class _UnbufferedStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for real-time output."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass  # Ignore flush errors on closed streams


def _windows_console_workaround() -> None:
    """Apply Windows-specific console output workarounds."""
    import os
    # Enable ANSI escape sequence support on Windows 10+
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Get STD_OUTPUT_HANDLE
            stdout_handle = kernel32.GetStdHandle(-11)
            # Get current mode
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
                # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
                # and ENABLE_PROCESSED_OUTPUT (0x0001)
                new_mode = mode.value | 0x0004 | 0x0001
                kernel32.SetConsoleMode(stdout_handle, new_mode)
        except Exception:
            pass  # Ignore if not running in a real console


@contextmanager
def timed() -> Iterator[callable]:
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def setup_logging(log_file: Path | str, level: str = "INFO") -> Path:
    _windows_console_workaround()
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root.addHandler(file_handler)

    # Use line buffering for console output on Windows
    import os
    if os.name == "nt":
        import io
        # Re-open stdout with line buffering
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding=sys.stdout.encoding,
            errors=sys.stdout.errors,
            newline=sys.stdout.newlines,
            line_buffering=True,
            write_through=True,
        )
    console_handler = _UnbufferedStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized: %s", path)
    return path
