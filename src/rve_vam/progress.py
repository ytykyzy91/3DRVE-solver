"""Simple progress reporting that works with both logging and direct print."""
from __future__ import annotations

import logging
import sys
from datetime import timedelta


class ProgressReporter:
    """Reports progress to both logging and stdout with immediate flushing."""

    def __init__(self, name: str = "progress"):
        self.logger = logging.getLogger(name)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Report info level message."""
        self.logger.info(msg, *args, **kwargs)
        self._print(f"INFO: {msg}")

    def _print(self, msg: str) -> None:
        """Print with immediate flush."""
        try:
            print(msg, flush=True)
        except Exception:
            pass  # Ignore print errors

    def start(self, operation: str) -> None:
        """Report start of an operation."""
        msg = f"▶ Starting: {operation}"
        self.logger.info(msg)
        self._print(msg)

    def complete(self, operation: str, elapsed: float | timedelta) -> None:
        """Report completion of an operation with elapsed time."""
        if isinstance(elapsed, float):
            elapsed_str = f"{elapsed:.2f}s"
        else:
            elapsed_str = str(elapsed)
        msg = f"✓ Completed: {operation} ({elapsed_str})"
        self.logger.info(msg)
        self._print(msg)

    def progress(self, operation: str, current: int, total: int, extra: str = "") -> None:
        """Report progress percentage."""
        if total > 0:
            pct = 100.0 * current / total
            msg = f"  {operation}: {current}/{total} ({pct:.1f}%) {extra}"
        else:
            msg = f"  {operation}: {current} {extra}"
        self.logger.info(msg)
        self._print(msg)

    def warning(self, msg: str) -> None:
        """Report warning."""
        self.logger.warning(msg)
        self._print(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        """Report error."""
        self.logger.error(msg)
        self._print(f"ERROR: {msg}")


# Global progress reporter instance
_progress = ProgressReporter()


def info(msg: str) -> None:
    """Global info reporter."""
    _progress.info(msg)


def start(operation: str) -> None:
    """Global start reporter."""
    _progress.start(operation)


def complete(operation: str, elapsed: float | timedelta) -> None:
    """Global complete reporter."""
    _progress.complete(operation, elapsed)


def progress(operation: str, current: int, total: int, extra: str = "") -> None:
    """Global progress reporter."""
    _progress.progress(operation, current, total, extra)
