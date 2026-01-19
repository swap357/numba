#!/usr/bin/env python3
"""
NumPy vs Numba Benchmark Harness

Compares performance of NumPy functions against Numba-compiled equivalents.
Designed for low-jitter measurements with CPU pinning.

Usage:
    python harness.py --cpu 7 --sizes 1000,10000,100000,1000000
    taskset -c 7 python harness.py  # Alternative CPU pinning
"""

import os
import sys

# Add script directory to path for benchmark imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import time
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
import subprocess

import numpy as np

# Import statistical analysis modules
from stats import TimingStats, compute_timing_stats, compare_with_significance, format_stats
from reproducibility import (
    SystemState, capture_system_state, format_system_state,
    isolated_benchmark_environment, state_to_dict
)

# Try to import numba - will be used for compilation
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: Numba not available", file=sys.stderr)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    size: int
    dtype: str
    numpy_time_ns: float
    numba_time_ns: float
    speedup: float  # numpy_time / numba_time (>1 means numba faster)
    numpy_result_hash: int  # For correctness check
    numba_result_hash: int
    correct: bool
    # Statistical analysis fields
    numpy_stats: Optional[TimingStats] = None
    numba_stats: Optional[TimingStats] = None
    significant: bool = True  # Whether speedup difference is statistically significant
    p_value: float = 0.0  # P-value from significance test


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    cpu: int = 7  # CPU to pin to
    sizes: List[int] = field(default_factory=lambda: [1_000, 10_000, 100_000, 1_000_000])
    dtypes: List[str] = field(default_factory=lambda: ['float64'])
    warmup_runs: int = 3
    timed_runs: int = 50
    output_dir: str = 'results'


def pin_to_cpu(cpu: int) -> bool:
    """Pin current process to specified CPU core."""
    try:
        os.sched_setaffinity(0, {cpu})
        # Verify
        actual = os.sched_getaffinity(0)
        if actual == {cpu}:
            print(f"Pinned to CPU {cpu}")
            return True
        else:
            print(f"WARNING: Affinity set to {actual}, expected {{{cpu}}}")
            return False
    except Exception as e:
        print(f"WARNING: Could not pin to CPU {cpu}: {e}")
        return False


def set_high_priority() -> bool:
    """Try to set high process priority."""
    try:
        os.nice(-10)
        return True
    except PermissionError:
        # Not root, that's ok
        return False


def hash_result(arr: np.ndarray) -> int:
    """Quick hash of array for correctness checking."""
    if arr is None:
        return 0
    if np.isscalar(arr):
        return hash(float(arr))
    # Use a sample for large arrays
    flat = np.asarray(arr).ravel()
    if len(flat) > 1000:
        indices = np.linspace(0, len(flat)-1, 1000, dtype=int)
        sample = flat[indices]
    else:
        sample = flat
    # Handle NaN/Inf
    sample = np.nan_to_num(sample, nan=0.0, posinf=1e308, neginf=-1e308)
    return hash(sample.tobytes())


def time_function(func: Callable, args: tuple, warmup: int, runs: int,
                  return_stats: bool = False) -> tuple:
    """
    Time a function with comprehensive statistics.

    Args:
        func: Function to time
        args: Arguments to pass to function
        warmup: Number of warmup runs
        runs: Number of timed runs
        return_stats: If True, return (TimingStats, raw_times), else (median, raw_times)

    Returns:
        Tuple of (median_or_stats, raw_times_list)
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        func(*args)
        end = time.perf_counter_ns()
        times.append(end - start)

    if return_stats:
        stats = compute_timing_stats(times)
        return stats, times
    else:
        # Return median (more robust than mean)
        times_sorted = sorted(times)
        median = times_sorted[len(times_sorted) // 2]
        return median, times


def check_results_close(np_result, nb_result, rtol=1e-5, atol=1e-8) -> bool:
    """Check if numpy and numba results are close enough."""
    if np_result is None and nb_result is None:
        return True
    if np_result is None or nb_result is None:
        return False

    try:
        np_arr = np.asarray(np_result)
        nb_arr = np.asarray(nb_result)

        if np_arr.shape != nb_arr.shape:
            return False

        return np.allclose(np_arr, nb_arr, rtol=rtol, atol=atol, equal_nan=True)
    except Exception:
        return False


class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self._numba_func = None

    def setup(self, size: int, dtype: np.dtype) -> tuple:
        """Create input data. Returns tuple of args."""
        raise NotImplementedError

    def numpy_impl(self, *args) -> Any:
        """NumPy implementation."""
        raise NotImplementedError

    def numba_impl(self, *args) -> Any:
        """Numba implementation (will be JIT compiled)."""
        raise NotImplementedError

    def get_numba_func(self) -> Callable:
        """Get or create JIT-compiled function."""
        if self._numba_func is None:
            self._numba_func = njit(self.numba_impl, cache=True, fastmath=True)
        return self._numba_func

    def run(self, config: BenchmarkConfig, size: int, dtype: str) -> BenchmarkResult:
        """Run the benchmark for given size and dtype."""
        np_dtype = getattr(np, dtype)
        args = self.setup(size, np_dtype)

        # Get numba function (triggers compilation on first call)
        numba_func = self.get_numba_func()

        # Warmup numba (includes compilation)
        for _ in range(config.warmup_runs):
            numba_func(*args)

        # Time numpy with statistics
        numpy_stats, numpy_times = time_function(
            self.numpy_impl, args,
            config.warmup_runs, config.timed_runs,
            return_stats=True
        )

        # Time numba with statistics
        numba_stats, numba_times = time_function(
            numba_func, args,
            config.warmup_runs, config.timed_runs,
            return_stats=True
        )

        # Check correctness
        np_result = self.numpy_impl(*args)
        nb_result = numba_func(*args)
        correct = check_results_close(np_result, nb_result)

        # Calculate speedup using median
        speedup = numpy_stats.median / numba_stats.median if numba_stats.median > 0 else float('inf')

        # Statistical significance test
        comparison = compare_with_significance(numpy_times, numba_times)

        return BenchmarkResult(
            name=self.name,
            category=self.category,
            size=size,
            dtype=dtype,
            numpy_time_ns=numpy_stats.median,
            numba_time_ns=numba_stats.median,
            speedup=speedup,
            numpy_result_hash=hash_result(np_result),
            numba_result_hash=hash_result(nb_result),
            correct=correct,
            numpy_stats=numpy_stats,
            numba_stats=numba_stats,
            significant=comparison.significant,
            p_value=comparison.p_value
        )


# Import registry from separate module to avoid circular import issues
from registry import BENCHMARKS, register_benchmark


def run_benchmarks(config: BenchmarkConfig, categories: Optional[List[str]] = None) -> List[BenchmarkResult]:
    """Run all registered benchmarks."""
    results = []

    if categories is None:
        categories = list(BENCHMARKS.keys())

    total = sum(
        len(BENCHMARKS[cat]) * len(config.sizes) * len(config.dtypes)
        for cat in categories
    )
    current = 0

    for category in categories:
        benchmarks = BENCHMARKS[category]
        for bench in benchmarks:
            for dtype in config.dtypes:
                for size in config.sizes:
                    current += 1
                    print(f"[{current}/{total}] {bench.name} size={size} dtype={dtype}...", end=' ', flush=True)

                    try:
                        result = bench.run(config, size, dtype)
                        results.append(result)

                        status = "OK" if result.correct else "MISMATCH"
                        print(f"{status} speedup={result.speedup:.2f}x")
                    except Exception as e:
                        print(f"ERROR: {e}")

    return results


def results_to_markdown(results: List[BenchmarkResult],
                        system_state: Optional[SystemState] = None) -> str:
    """Convert results to markdown table with statistical details."""
    lines = [
        "# NumPy vs Numba Benchmark Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Numba version:** {numba.__version__ if NUMBA_AVAILABLE else 'N/A'}",
        f"**NumPy version:** {np.__version__}",
    ]

    # Add system state info if available
    if system_state:
        lines.extend([
            f"**CPU:** {system_state.cpu_model}",
            f"**Governor:** {system_state.cpu_governor}",
        ])
        if system_state.warnings:
            lines.append("")
            lines.append("### Reproducibility Warnings")
            for w in system_state.warnings:
                lines.append(f"- {w}")

    lines.extend([
        "",
        "## Results",
        "",
        "| Category | Function | Size | Dtype | NumPy (us) | Numba (us) | Speedup | CI 95% | Sig | Correct |",
        "|----------|----------|------|-------|------------|------------|---------|--------|-----|---------|",
    ])

    for r in results:
        np_us = r.numpy_time_ns / 1000
        nb_us = r.numba_time_ns / 1000
        speedup_str = f"{r.speedup:.2f}x"
        if r.speedup > 1:
            speedup_str = f"**{speedup_str}**"
        correct = "Yes" if r.correct else "**NO**"

        # CI info if available
        if r.numba_stats:
            ci_low = r.numba_stats.ci_lower / 1000
            ci_high = r.numba_stats.ci_upper / 1000
            ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]"
        else:
            ci_str = "-"

        sig_str = "Yes" if r.significant else "No"

        lines.append(
            f"| {r.category} | {r.name} | {r.size:,} | {r.dtype} | "
            f"{np_us:.1f} | {nb_us:.1f} | {speedup_str} | {ci_str} | {sig_str} | {correct} |"
        )

    # Add legend
    lines.extend([
        "",
        "---",
        "*CI 95%: 95% confidence interval for Numba timing (microseconds)*",
        "*Sig: Statistical significance of speedup difference (p < 0.05)*",
    ])

    return "\n".join(lines)


def results_to_csv(results: List[BenchmarkResult], path: str):
    """Save results to CSV with full statistical details."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'category', 'name', 'size', 'dtype',
            'numpy_time_ns', 'numba_time_ns', 'speedup', 'correct',
            'numpy_ci_lower', 'numpy_ci_upper', 'numpy_cv',
            'numba_ci_lower', 'numba_ci_upper', 'numba_cv',
            'significant', 'p_value'
        ])
        for r in results:
            # Extract stats if available
            np_ci_lower = r.numpy_stats.ci_lower if r.numpy_stats else ''
            np_ci_upper = r.numpy_stats.ci_upper if r.numpy_stats else ''
            np_cv = r.numpy_stats.cv if r.numpy_stats else ''
            nb_ci_lower = r.numba_stats.ci_lower if r.numba_stats else ''
            nb_ci_upper = r.numba_stats.ci_upper if r.numba_stats else ''
            nb_cv = r.numba_stats.cv if r.numba_stats else ''

            writer.writerow([
                r.category, r.name, r.size, r.dtype,
                r.numpy_time_ns, r.numba_time_ns, r.speedup, r.correct,
                np_ci_lower, np_ci_upper, np_cv,
                nb_ci_lower, nb_ci_upper, nb_cv,
                r.significant, r.p_value
            ])


def print_summary(results: List[BenchmarkResult]):
    """Print summary statistics with statistical significance info."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Group by category
    by_category = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    for category, cat_results in by_category.items():
        speedups = [r.speedup for r in cat_results]
        correct = sum(1 for r in cat_results if r.correct)
        significant = sum(1 for r in cat_results if r.significant)
        noisy_numpy = sum(1 for r in cat_results if r.numpy_stats and r.numpy_stats.is_noisy)
        noisy_numba = sum(1 for r in cat_results if r.numba_stats and r.numba_stats.is_noisy)

        print(f"\n{category.upper()}:")
        print(f"  Benchmarks: {len(cat_results)}")
        print(f"  Correct: {correct}/{len(cat_results)}")
        print(f"  Significant results: {significant}/{len(cat_results)}")
        print(f"  Speedup - min: {min(speedups):.2f}x, max: {max(speedups):.2f}x, median: {sorted(speedups)[len(speedups)//2]:.2f}x")

        # Noise warnings
        if noisy_numpy > 0 or noisy_numba > 0:
            print(f"  Noisy measurements: {noisy_numpy} NumPy, {noisy_numba} Numba (CV > 5%)")

        # Find best/worst
        fastest_numba = max(cat_results, key=lambda r: r.speedup)
        slowest_numba = min(cat_results, key=lambda r: r.speedup)
        print(f"  Best for Numba: {fastest_numba.name} ({fastest_numba.speedup:.2f}x)")
        print(f"  Best for NumPy: {slowest_numba.name} ({slowest_numba.speedup:.2f}x)")

        # Show statistical details for extreme cases
        if fastest_numba.numba_stats:
            print(f"    -> Numba: {format_stats(fastest_numba.numba_stats)}")
        if slowest_numba.numpy_stats:
            print(f"    -> NumPy: {format_stats(slowest_numba.numpy_stats)}")


def main():
    parser = argparse.ArgumentParser(description='NumPy vs Numba Benchmark')
    parser.add_argument('--cpu', type=int, default=7, help='CPU to pin to (default: 7)')
    parser.add_argument('--sizes', type=str, default='1000,10000,100000,1000000',
                       help='Comma-separated array sizes')
    parser.add_argument('--dtypes', type=str, default='float64',
                       help='Comma-separated dtypes')
    parser.add_argument('--runs', type=int, default=50, help='Timed runs per benchmark')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup runs')
    parser.add_argument('--categories', type=str, default=None,
                       help='Comma-separated categories to run (default: all)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--no-pin', action='store_true', help='Disable CPU pinning')
    parser.add_argument('--show-system', action='store_true',
                       help='Show detailed system state')
    parser.add_argument('--disable-gc', action='store_true',
                       help='Disable garbage collection during benchmarks')

    args = parser.parse_args()

    # Capture system state first
    system_state = capture_system_state()

    # Setup config
    config = BenchmarkConfig(
        cpu=args.cpu,
        sizes=[int(s) for s in args.sizes.split(',')],
        dtypes=args.dtypes.split(','),
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        output_dir=args.output,
    )

    print("="*60)
    print("NumPy vs Numba Benchmark Harness")
    print("="*60)
    print(f"NumPy version: {np.__version__}")
    print(f"Numba version: {numba.__version__ if NUMBA_AVAILABLE else 'N/A'}")
    print(f"Sizes: {config.sizes}")
    print(f"Dtypes: {config.dtypes}")
    print(f"Runs: {config.timed_runs} (warmup: {config.warmup_runs})")

    # Show system state if requested or if there are warnings
    if args.show_system:
        print(format_system_state(system_state))
    elif system_state.warnings:
        print("\nReproducibility warnings:")
        for w in system_state.warnings:
            print(f"  - {w}")

    # Pin to CPU
    if not args.no_pin:
        pin_to_cpu(config.cpu)

    # Try high priority
    set_high_priority()

    # Import benchmarks (registers them)
    from benchmarks import ufuncs, reductions

    # Categories to run
    categories = None
    if args.categories:
        categories = args.categories.split(',')

    print(f"\nRegistered benchmarks:")
    for cat, benchs in BENCHMARKS.items():
        if benchs:
            print(f"  {cat}: {len(benchs)}")

    print("\nStarting benchmarks...\n")

    # Run with optional isolated environment
    if args.disable_gc:
        with isolated_benchmark_environment(disable_gc=True):
            results = run_benchmarks(config, categories)
    else:
        results = run_benchmarks(config, categories)

    # Summary
    print_summary(results)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(config.output_dir, f'benchmark_{timestamp}.csv')
    results_to_csv(results, csv_path)
    print(f"\nResults saved to: {csv_path}")

    md_path = os.path.join(config.output_dir, f'benchmark_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(results_to_markdown(results, system_state))
    print(f"Markdown saved to: {md_path}")

    # Save system state as JSON
    import json
    state_path = os.path.join(config.output_dir, f'system_state_{timestamp}.json')
    with open(state_path, 'w') as f:
        json.dump(state_to_dict(system_state), f, indent=2)
    print(f"System state saved to: {state_path}")


if __name__ == '__main__':
    main()
