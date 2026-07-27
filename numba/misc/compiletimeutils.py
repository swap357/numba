"""
Compile time monitoring utilities for measuring JIT compilation time.

Numba only emits events around *compilation*; the JIT'd function runs as
native code with no event hook, so runtime is wall-clock minus compile time.
Compile time is split further with the finer events ``numba:run_pass`` (the
pass pipeline) and ``numba:llvm_lock`` (LLVM codegen). The containment is
codegen <= passes <= compile (codegen happens inside the lowering pass).

Example usage:
    tracker = CompileTimeTracker("my_test")
    with tracker.monitor():
        run_test()
    # Access data: tracker.compile_time, tracker.compile_count, etc.
    # Get formatted string: tracker.get_summary()
"""
from __future__ import annotations
import contextlib
import time
from typing import Optional

from numba.core import event


class _CompileTimer(event.TimingListener):
    """TimingListener that also counts top-level compile episodes.

    Counting keys off ``TimingListener._depth`` returning to 0 on each *END*;
    keep in sync with ``numba.core.event.TimingListener`` if its nesting logic
    changes.
    """
    count = 0

    def on_end(self, evt):
        super().on_end(evt)
        if self._depth == 0:
            self.count += 1


class CompileTimeTracker:
    """
    A simple compile time monitor that tracks JIT compilation time.

    Stores monitoring data in instance attributes for later access.
    Each instance is typically used for monitoring a single test.
    """
    name: str
    start_time: float | None
    end_time: float | None
    duration: float | None
    compile_time: float | None
    compile_count: int | None
    passes_time: float | None
    codegen_time: float | None

    def __init__(self, name: str):
        """Initialize a CompileTimeTracker with empty monitoring data."""
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.compile_time = None
        self.compile_count = None
        self.passes_time = None
        self.codegen_time = None

    @contextlib.contextmanager
    def monitor(self):
        """
        Context manager to monitor compile time during test execution.

        Records compile time via Numba's ``numba:compile`` event, plus the
        ``numba:run_pass`` and ``numba:llvm_lock`` sub-phases, and stores all
        data in instance attributes.

        Yields:
            self: The CompileTimeTracker instance for accessing stored data
        """
        compile_l = _CompileTimer()
        passes_l = event.TimingListener()
        codegen_l = event.TimingListener()
        self.start_time = time.perf_counter()
        with event.install_listener("numba:compile", compile_l), \
                event.install_listener("numba:run_pass", passes_l), \
                event.install_listener("numba:llvm_lock", codegen_l):
            try:
                yield self
            finally:
                self.end_time = time.perf_counter()
                self.duration = self.end_time - self.start_time
                self.compile_time = (compile_l.duration if compile_l.done
                                     else 0.0)
                self.compile_count = compile_l.count
                self.passes_time = (passes_l.duration if passes_l.done
                                    else 0.0)
                self.codegen_time = (codegen_l.duration if codegen_l.done
                                     else 0.0)

    def get_summary(self) -> str:
        """
        Return a formatted summary of the compile time monitoring data.

        Returns:
            str: Formatted summary string with monitoring results

        Note:
            Should be called after monitor() context has completed
            to ensure all data is available.
        """
        if self.duration is None:
            raise ValueError("Compile time monitoring data not available")

        def format_time(seconds: Optional[float]) -> str:
            """Convert seconds to human readable format."""
            if seconds is None:
                return "N/A"
            if seconds >= 1.0:
                return f"{seconds:.3f}s"
            return f"{seconds * 1000:.2f}ms"

        run_time = self.duration - self.compile_time

        buf = [
            f"Name: {self.name}",
            f"Duration: {format_time(self.duration)}",
            f"Compile: {format_time(self.compile_time)} "
            f"({self.compile_count} calls; "
            f"passes {format_time(self.passes_time)}, "
            f"codegen {format_time(self.codegen_time)})",
            f"Run: {format_time(run_time)}",
        ]
        return ' | '.join(buf)
