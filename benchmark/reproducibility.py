"""
Reproducibility framework for benchmark measurements.

Captures system state and provides environment isolation for consistent results.
"""

import os
import gc
import sys
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from contextlib import contextmanager


@dataclass
class SystemState:
    """Captures system state relevant to benchmark reproducibility."""
    # CPU info
    cpu_model: str = ""
    cpu_freq_mhz: Optional[float] = None
    cpu_freq_min_mhz: Optional[float] = None
    cpu_freq_max_mhz: Optional[float] = None
    cpu_governor: str = ""
    cpu_count: int = 0
    cpu_count_physical: int = 0

    # System load
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0

    # Memory
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0

    # Software versions
    python_version: str = ""
    numpy_version: str = ""
    numba_version: str = ""
    llvm_version: str = ""

    # Git info
    git_hash: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Platform
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""

    # Warnings
    warnings: List[str] = field(default_factory=list)


def _read_file(path: str) -> str:
    """Read file contents, return empty string on error."""
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except (IOError, OSError):
        return ""


def _run_cmd(cmd: List[str]) -> str:
    """Run command and return output, empty string on error."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information from various sources."""
    info = {
        'model': '',
        'freq_mhz': None,
        'freq_min_mhz': None,
        'freq_max_mhz': None,
        'governor': '',
        'count': os.cpu_count() or 0,
        'count_physical': 0,
    }

    # Try /proc/cpuinfo
    cpuinfo = _read_file('/proc/cpuinfo')
    if cpuinfo:
        for line in cpuinfo.split('\n'):
            if line.startswith('model name'):
                info['model'] = line.split(':')[-1].strip()
                break

    # Try to get physical core count
    try:
        # Count unique physical IDs * cores per physical
        physical_ids = set()
        cores_per_physical = 1
        for line in cpuinfo.split('\n'):
            if line.startswith('physical id'):
                physical_ids.add(line.split(':')[-1].strip())
            if line.startswith('cpu cores'):
                cores_per_physical = int(line.split(':')[-1].strip())
        if physical_ids:
            info['count_physical'] = len(physical_ids) * cores_per_physical
        else:
            info['count_physical'] = info['count']
    except (ValueError, AttributeError):
        info['count_physical'] = info['count']

    # CPU frequency from scaling driver
    freq_cur = _read_file('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    freq_min = _read_file('/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq')
    freq_max = _read_file('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')
    governor = _read_file('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')

    if freq_cur:
        info['freq_mhz'] = int(freq_cur) / 1000
    if freq_min:
        info['freq_min_mhz'] = int(freq_min) / 1000
    if freq_max:
        info['freq_max_mhz'] = int(freq_max) / 1000
    if governor:
        info['governor'] = governor

    return info


def _get_memory_info() -> Dict[str, float]:
    """Get memory information."""
    info = {'total_gb': 0.0, 'available_gb': 0.0}

    meminfo = _read_file('/proc/meminfo')
    if meminfo:
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                kb = int(line.split()[1])
                info['total_gb'] = kb / 1024 / 1024
            elif line.startswith('MemAvailable:'):
                kb = int(line.split()[1])
                info['available_gb'] = kb / 1024 / 1024

    return info


def _get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get git repository information."""
    info = {'hash': '', 'branch': '', 'dirty': False}

    cwd = repo_path or os.getcwd()

    # Git hash
    hash_out = _run_cmd(['git', '-C', cwd, 'rev-parse', '--short', 'HEAD'])
    if hash_out:
        info['hash'] = hash_out

    # Git branch
    branch_out = _run_cmd(['git', '-C', cwd, 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch_out:
        info['branch'] = branch_out

    # Check for uncommitted changes
    status_out = _run_cmd(['git', '-C', cwd, 'status', '--porcelain'])
    info['dirty'] = bool(status_out)

    return info


def capture_system_state(git_repo_path: Optional[str] = None) -> SystemState:
    """
    Capture complete system state for reproducibility tracking.

    Args:
        git_repo_path: Optional path to git repo for version tracking

    Returns:
        SystemState dataclass with all captured information
    """
    state = SystemState()
    warnings = []

    # CPU info
    cpu_info = _get_cpu_info()
    state.cpu_model = cpu_info['model']
    state.cpu_freq_mhz = cpu_info['freq_mhz']
    state.cpu_freq_min_mhz = cpu_info['freq_min_mhz']
    state.cpu_freq_max_mhz = cpu_info['freq_max_mhz']
    state.cpu_governor = cpu_info['governor']
    state.cpu_count = cpu_info['count']
    state.cpu_count_physical = cpu_info['count_physical']

    # Load average
    try:
        load = os.getloadavg()
        state.load_avg_1m = load[0]
        state.load_avg_5m = load[1]
        state.load_avg_15m = load[2]
    except (OSError, AttributeError):
        pass

    # Memory
    mem_info = _get_memory_info()
    state.memory_total_gb = mem_info['total_gb']
    state.memory_available_gb = mem_info['available_gb']

    # Python version
    state.python_version = platform.python_version()

    # NumPy version
    try:
        import numpy as np
        state.numpy_version = np.__version__
    except ImportError:
        pass

    # Numba version
    try:
        import numba
        state.numba_version = numba.__version__
    except ImportError:
        pass

    # LLVM version from Numba
    try:
        from numba.core import config
        if hasattr(config, 'LLVM_VERSION'):
            state.llvm_version = str(config.LLVM_VERSION)
        else:
            from llvmlite import binding as llvm
            state.llvm_version = '.'.join(map(str, llvm.llvm_version_info))
    except (ImportError, AttributeError):
        pass

    # Git info
    git_info = _get_git_info(git_repo_path)
    state.git_hash = git_info['hash']
    state.git_branch = git_info['branch']
    state.git_dirty = git_info['dirty']

    # Platform info
    state.os_name = platform.system()
    state.os_version = platform.release()
    state.kernel_version = platform.version()

    # Generate warnings
    state.warnings = check_reproducibility_requirements(state)

    return state


def check_reproducibility_requirements(state: SystemState) -> List[str]:
    """
    Check for conditions that may affect benchmark reproducibility.

    Returns:
        List of warning messages
    """
    warnings = []

    # Check CPU governor
    if state.cpu_governor and state.cpu_governor != 'performance':
        warnings.append(
            f"CPU governor is '{state.cpu_governor}', recommend 'performance' "
            f"for stable measurements. Set with: "
            f"sudo cpupower frequency-set -g performance"
        )

    # Check for frequency scaling
    if state.cpu_freq_min_mhz and state.cpu_freq_max_mhz:
        if state.cpu_freq_min_mhz != state.cpu_freq_max_mhz:
            warnings.append(
                f"CPU frequency scaling enabled ({state.cpu_freq_min_mhz:.0f}-"
                f"{state.cpu_freq_max_mhz:.0f} MHz). Consider disabling for "
                f"consistent results."
            )

    # Check system load
    if state.load_avg_1m > state.cpu_count_physical * 0.5:
        warnings.append(
            f"High system load ({state.load_avg_1m:.1f}). Consider reducing "
            f"background processes."
        )

    # Check available memory
    if state.memory_available_gb < 1.0:
        warnings.append(
            f"Low available memory ({state.memory_available_gb:.1f} GB). "
            f"May cause swapping."
        )

    # Check for hyperthreading
    if state.cpu_count > state.cpu_count_physical and state.cpu_count_physical > 0:
        warnings.append(
            f"Hyperthreading detected ({state.cpu_count} logical, "
            f"{state.cpu_count_physical} physical cores). Pin to physical "
            f"cores for consistent results."
        )

    # Check for dirty git state
    if state.git_dirty:
        warnings.append(
            "Git repository has uncommitted changes. Results may not be "
            "reproducible."
        )

    return warnings


@contextmanager
def isolated_benchmark_environment(
    disable_gc: bool = True,
    flush_caches: bool = True,
    set_affinity: Optional[int] = None,
    nice_priority: Optional[int] = None
):
    """
    Context manager for isolated benchmark environment.

    Args:
        disable_gc: Disable garbage collection during benchmark
        flush_caches: Attempt to flush CPU caches (requires elevated privileges)
        set_affinity: Pin to specific CPU core
        nice_priority: Set process priority (negative = higher priority)

    Usage:
        with isolated_benchmark_environment(disable_gc=True, set_affinity=7):
            # Run benchmarks here
            result = time_function(...)
    """
    # Store original state
    original_gc_enabled = gc.isenabled()
    original_affinity = None
    original_nice = None

    try:
        # Disable GC
        if disable_gc:
            gc.disable()
            gc.collect()  # Clean up before disabling

        # Set CPU affinity
        if set_affinity is not None:
            try:
                original_affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, {set_affinity})
            except (AttributeError, OSError):
                pass  # Not available on this platform

        # Set process priority
        if nice_priority is not None:
            try:
                original_nice = os.nice(0)  # Get current
                os.nice(nice_priority)
            except (PermissionError, OSError):
                pass

        # Attempt cache flush (Linux only, requires privileges)
        if flush_caches:
            try:
                # This requires root but we try anyway
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')
            except (PermissionError, IOError):
                pass

        yield

    finally:
        # Restore original state
        if disable_gc and original_gc_enabled:
            gc.enable()

        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
            except (AttributeError, OSError):
                pass

        # Note: nice value cannot be decreased without privileges
        # so we don't try to restore it


def format_system_state(state: SystemState) -> str:
    """Format SystemState as human-readable string."""
    lines = [
        "=" * 60,
        "SYSTEM STATE",
        "=" * 60,
        "",
        "CPU:",
        f"  Model: {state.cpu_model}",
        f"  Cores: {state.cpu_count} logical, {state.cpu_count_physical} physical",
    ]

    if state.cpu_freq_mhz:
        lines.append(f"  Frequency: {state.cpu_freq_mhz:.0f} MHz")
        if state.cpu_freq_min_mhz and state.cpu_freq_max_mhz:
            lines.append(
                f"  Freq range: {state.cpu_freq_min_mhz:.0f}-"
                f"{state.cpu_freq_max_mhz:.0f} MHz"
            )
    if state.cpu_governor:
        lines.append(f"  Governor: {state.cpu_governor}")

    lines.extend([
        "",
        "System:",
        f"  Load: {state.load_avg_1m:.2f} / {state.load_avg_5m:.2f} / "
        f"{state.load_avg_15m:.2f}",
        f"  Memory: {state.memory_available_gb:.1f} / "
        f"{state.memory_total_gb:.1f} GB available",
        f"  OS: {state.os_name} {state.os_version}",
        "",
        "Software:",
        f"  Python: {state.python_version}",
        f"  NumPy: {state.numpy_version}",
        f"  Numba: {state.numba_version}",
        f"  LLVM: {state.llvm_version}",
    ])

    if state.git_hash:
        dirty_flag = " (dirty)" if state.git_dirty else ""
        lines.append(f"  Git: {state.git_branch}@{state.git_hash}{dirty_flag}")

    if state.warnings:
        lines.extend([
            "",
            "WARNINGS:",
        ])
        for warning in state.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Convert SystemState to dictionary for serialization."""
    return {
        'cpu': {
            'model': state.cpu_model,
            'freq_mhz': state.cpu_freq_mhz,
            'freq_range_mhz': [state.cpu_freq_min_mhz, state.cpu_freq_max_mhz],
            'governor': state.cpu_governor,
            'count_logical': state.cpu_count,
            'count_physical': state.cpu_count_physical,
        },
        'load': {
            '1m': state.load_avg_1m,
            '5m': state.load_avg_5m,
            '15m': state.load_avg_15m,
        },
        'memory': {
            'total_gb': state.memory_total_gb,
            'available_gb': state.memory_available_gb,
        },
        'software': {
            'python': state.python_version,
            'numpy': state.numpy_version,
            'numba': state.numba_version,
            'llvm': state.llvm_version,
        },
        'git': {
            'hash': state.git_hash,
            'branch': state.git_branch,
            'dirty': state.git_dirty,
        },
        'platform': {
            'os': state.os_name,
            'os_version': state.os_version,
            'kernel': state.kernel_version,
        },
        'warnings': state.warnings,
    }
