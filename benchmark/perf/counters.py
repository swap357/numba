#!/usr/bin/env python3
"""
Hardware counter comparison using perf stat.

Compares NumPy vs Numba implementations at the hardware level:
- Memory loads/stores (L1, LLC)
- Branch predictions
- Instructions per cycle

Usage:
    python perf/counters.py --cpu 7 --function np.sum --size 100000
"""

import subprocess
import tempfile
import os
import sys
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Events to capture
PERF_EVENTS = [
    'cycles',
    'instructions',
    'L1-dcache-loads',
    'L1-dcache-load-misses',
    'L1-dcache-stores',
    'LLC-loads',
    'LLC-load-misses',
    'branches',
    'branch-misses',
]


@dataclass
class PerfCounters:
    """Hardware performance counters."""
    cycles: int = 0
    instructions: int = 0
    l1_loads: int = 0
    l1_load_misses: int = 0
    l1_stores: int = 0
    llc_loads: int = 0
    llc_load_misses: int = 0
    branches: int = 0
    branch_misses: int = 0
    time_seconds: float = 0.0

    @property
    def ipc(self) -> float:
        """Instructions per cycle."""
        return self.instructions / self.cycles if self.cycles > 0 else 0

    @property
    def l1_miss_rate(self) -> float:
        """L1 data cache miss rate."""
        return self.l1_load_misses / self.l1_loads if self.l1_loads > 0 else 0

    @property
    def llc_miss_rate(self) -> float:
        """Last level cache miss rate."""
        return self.llc_load_misses / self.llc_loads if self.llc_loads > 0 else 0

    @property
    def branch_miss_rate(self) -> float:
        """Branch misprediction rate."""
        return self.branch_misses / self.branches if self.branches > 0 else 0


def parse_perf_output(output: str) -> PerfCounters:
    """Parse perf stat output into PerfCounters."""
    counters = PerfCounters()

    # Parse each line
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Handle "not counted" cases
        if '<not counted>' in line:
            continue

        # Extract number and event name
        # Hybrid CPU format: "1,234,567      cpu_core/cycles/"
        # Standard format: "1,234,567      cycles"
        match = re.match(r'^\s*([\d,]+)\s+(?:cpu_\w+/)?([\w-]+)/?', line)
        if match:
            value_str = match.group(1).replace(',', '')
            event = match.group(2)

            try:
                value = int(value_str)
            except ValueError:
                continue

            # Sum values from both cpu_atom and cpu_core (hybrid CPUs)
            if event == 'cycles':
                counters.cycles += value
            elif event == 'instructions':
                counters.instructions += value
            elif event == 'L1-dcache-loads':
                counters.l1_loads += value
            elif event == 'L1-dcache-load-misses':
                counters.l1_load_misses += value
            elif event == 'L1-dcache-stores':
                counters.l1_stores += value
            elif event == 'LLC-loads':
                counters.llc_loads += value
            elif event == 'LLC-load-misses':
                counters.llc_load_misses += value
            elif event == 'branches':
                counters.branches += value
            elif event == 'branch-misses':
                counters.branch_misses += value

        # Parse time - handles "0.0106049 +- 0.0000540 seconds time elapsed"
        if 'seconds time elapsed' in line:
            time_match = re.match(r'^\s*([\d.]+)', line)
            if time_match:
                counters.time_seconds = float(time_match.group(1))

    return counters


def run_with_perf(script: str, cpu: int, repeat: int = 5) -> PerfCounters:
    """Run a Python script snippet with perf stat and return counters."""

    events_str = ','.join(PERF_EVENTS)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Run with perf stat, pinned to specified CPU
        cmd = [
            'perf', 'stat',
            '-e', events_str,
            '-r', str(repeat),  # Repeat and average
            '--',
            'taskset', '-c', str(cpu),
            sys.executable, script_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # perf stat outputs to stderr
        return parse_perf_output(result.stderr)

    finally:
        os.unlink(script_path)


def generate_numpy_script(func_name: str, size: int, iterations: int = 100) -> str:
    """Generate a script that runs NumPy function repeatedly."""

    # Map function names to their setup and call
    setups = {
        'np.sin': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.sin(x)'),
        'np.cos': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.cos(x)'),
        'np.exp': ('x = np.random.uniform(-10, 10, SIZE).astype(np.float64)', 'np.exp(x)'),
        'np.log': ('x = np.random.uniform(0.1, 100, SIZE).astype(np.float64)', 'np.log(x)'),
        'np.sqrt': ('x = np.random.uniform(0, 1000, SIZE).astype(np.float64)', 'np.sqrt(x)'),
        'np.tanh': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.tanh(x)'),
        'np.abs': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.abs(x)'),
        'np.add': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'np.add(x, y)'),
        'np.multiply': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'np.multiply(x, y)'),
        'np.sum': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.sum(x)'),
        'np.mean': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.mean(x)'),
        'np.min': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.min(x)'),
        'np.max': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.max(x)'),
        'np.argmin': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.argmin(x)'),
        'np.argmax': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.argmax(x)'),
        'np.cumsum': ('x = np.random.randn(SIZE).astype(np.float64)', 'np.cumsum(x)'),
        'np.all': ('x = np.ones(SIZE, dtype=bool); x[SIZE//2] = False', 'np.all(x)'),
        'np.any': ('x = np.zeros(SIZE, dtype=bool); x[SIZE//2] = True', 'np.any(x)'),
        'np.maximum': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'np.maximum(x, y)'),
    }

    if func_name not in setups:
        raise ValueError(f"Unknown function: {func_name}. Available: {list(setups.keys())}")

    setup, call = setups[func_name]

    return f'''
import numpy as np
SIZE = {size}
ITERATIONS = {iterations}

# Setup
np.random.seed(42)
{setup}

# Warmup
for _ in range(3):
    _ = {call}

# Timed iterations
for _ in range(ITERATIONS):
    _ = {call}
'''


def generate_numba_script(func_name: str, size: int, iterations: int = 100) -> str:
    """Generate a script that runs Numba-compiled function repeatedly."""

    # Map function names to their numba implementations
    setups = {
        'np.sin': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.sin(x)'),
        'np.cos': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.cos(x)'),
        'np.exp': ('x = np.random.uniform(-10, 10, SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.exp(x)'),
        'np.log': ('x = np.random.uniform(0.1, 100, SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.log(x)'),
        'np.sqrt': ('x = np.random.uniform(0, 1000, SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.sqrt(x)'),
        'np.tanh': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.tanh(x)'),
        'np.abs': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.abs(x)'),
        'np.add': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x, y)', 'def nb_impl(x, y): return np.add(x, y)'),
        'np.multiply': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x, y)', 'def nb_impl(x, y): return np.multiply(x, y)'),
        'np.sum': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.sum(x)'),
        'np.mean': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.mean(x)'),
        'np.min': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.min(x)'),
        'np.max': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.max(x)'),
        'np.argmin': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.argmin(x)'),
        'np.argmax': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.argmax(x)'),
        'np.cumsum': ('x = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x)', 'def nb_impl(x): return np.cumsum(x)'),
        'np.all': ('x = np.ones(SIZE, dtype=bool); x[SIZE//2] = False', 'nb_func(x)', 'def nb_impl(x): return np.all(x)'),
        'np.any': ('x = np.zeros(SIZE, dtype=bool); x[SIZE//2] = True', 'nb_func(x)', 'def nb_impl(x): return np.any(x)'),
        'np.maximum': ('x = np.random.randn(SIZE).astype(np.float64); y = np.random.randn(SIZE).astype(np.float64)', 'nb_func(x, y)', 'def nb_impl(x, y): return np.maximum(x, y)'),
    }

    if func_name not in setups:
        raise ValueError(f"Unknown function: {func_name}")

    setup, call, impl = setups[func_name]

    return f'''
import numpy as np
from numba import njit

SIZE = {size}
ITERATIONS = {iterations}

# Define and compile numba function
@njit(cache=True, fastmath=True)
{impl}

nb_func = nb_impl

# Setup
np.random.seed(42)
{setup}

# Warmup (includes compilation)
for _ in range(5):
    _ = {call}

# Timed iterations
for _ in range(ITERATIONS):
    _ = {call}
'''


def compare_counters(np_counters: PerfCounters, nb_counters: PerfCounters) -> Dict:
    """Compare counters and return analysis."""
    return {
        'cycles': {
            'numpy': np_counters.cycles,
            'numba': nb_counters.cycles,
            'ratio': np_counters.cycles / nb_counters.cycles if nb_counters.cycles > 0 else 0,
        },
        'instructions': {
            'numpy': np_counters.instructions,
            'numba': nb_counters.instructions,
            'ratio': np_counters.instructions / nb_counters.instructions if nb_counters.instructions > 0 else 0,
        },
        'ipc': {
            'numpy': np_counters.ipc,
            'numba': nb_counters.ipc,
            'ratio': np_counters.ipc / nb_counters.ipc if nb_counters.ipc > 0 else 0,
        },
        'l1_loads': {
            'numpy': np_counters.l1_loads,
            'numba': nb_counters.l1_loads,
            'ratio': np_counters.l1_loads / nb_counters.l1_loads if nb_counters.l1_loads > 0 else 0,
        },
        'l1_miss_rate': {
            'numpy': np_counters.l1_miss_rate,
            'numba': nb_counters.l1_miss_rate,
            'diff': np_counters.l1_miss_rate - nb_counters.l1_miss_rate,
        },
        'l1_stores': {
            'numpy': np_counters.l1_stores,
            'numba': nb_counters.l1_stores,
            'ratio': np_counters.l1_stores / nb_counters.l1_stores if nb_counters.l1_stores > 0 else 0,
        },
        'llc_loads': {
            'numpy': np_counters.llc_loads,
            'numba': nb_counters.llc_loads,
            'ratio': np_counters.llc_loads / nb_counters.llc_loads if nb_counters.llc_loads > 0 else 0,
        },
        'llc_miss_rate': {
            'numpy': np_counters.llc_miss_rate,
            'numba': nb_counters.llc_miss_rate,
            'diff': np_counters.llc_miss_rate - nb_counters.llc_miss_rate,
        },
        'branches': {
            'numpy': np_counters.branches,
            'numba': nb_counters.branches,
            'ratio': np_counters.branches / nb_counters.branches if nb_counters.branches > 0 else 0,
        },
        'branch_miss_rate': {
            'numpy': np_counters.branch_miss_rate,
            'numba': nb_counters.branch_miss_rate,
            'diff': np_counters.branch_miss_rate - nb_counters.branch_miss_rate,
        },
        'time': {
            'numpy': np_counters.time_seconds,
            'numba': nb_counters.time_seconds,
            'speedup': np_counters.time_seconds / nb_counters.time_seconds if nb_counters.time_seconds > 0 else 0,
        },
    }


def format_comparison(func_name: str, size: int, comparison: Dict) -> str:
    """Format comparison as a readable report."""
    lines = [
        f"\n{'='*70}",
        f"Hardware Counter Comparison: {func_name} (size={size:,})",
        f"{'='*70}",
        "",
        f"{'Metric':<25} {'NumPy':>15} {'Numba':>15} {'Ratio/Diff':>12}",
        f"{'-'*70}",
    ]

    # Time
    t = comparison['time']
    speedup_str = f"{t['speedup']:.2f}x"
    if t['speedup'] > 1:
        speedup_str = f"NB {speedup_str}"
    else:
        speedup_str = f"NP {1/t['speedup']:.2f}x"
    lines.append(f"{'Time (s)':<25} {t['numpy']:>15.4f} {t['numba']:>15.4f} {speedup_str:>12}")

    # IPC
    ipc = comparison['ipc']
    lines.append(f"{'IPC':<25} {ipc['numpy']:>15.2f} {ipc['numba']:>15.2f} {ipc['ratio']:>11.2f}x")

    lines.append("")
    lines.append("Memory:")

    # L1 loads
    l1 = comparison['l1_loads']
    lines.append(f"{'  L1 Data Loads':<25} {l1['numpy']:>15,} {l1['numba']:>15,} {l1['ratio']:>11.2f}x")

    # L1 stores
    l1s = comparison['l1_stores']
    lines.append(f"{'  L1 Data Stores':<25} {l1s['numpy']:>15,} {l1s['numba']:>15,} {l1s['ratio']:>11.2f}x")

    # L1 miss rate
    l1m = comparison['l1_miss_rate']
    lines.append(f"{'  L1 Miss Rate':<25} {l1m['numpy']*100:>14.2f}% {l1m['numba']*100:>14.2f}% {l1m['diff']*100:>+10.2f}%")

    # LLC loads
    llc = comparison['llc_loads']
    lines.append(f"{'  LLC Loads':<25} {llc['numpy']:>15,} {llc['numba']:>15,} {llc['ratio']:>11.2f}x")

    # LLC miss rate
    llcm = comparison['llc_miss_rate']
    lines.append(f"{'  LLC Miss Rate':<25} {llcm['numpy']*100:>14.2f}% {llcm['numba']*100:>14.2f}% {llcm['diff']*100:>+10.2f}%")

    lines.append("")
    lines.append("Branches:")

    # Branches
    br = comparison['branches']
    lines.append(f"{'  Total Branches':<25} {br['numpy']:>15,} {br['numba']:>15,} {br['ratio']:>11.2f}x")

    # Branch miss rate
    brm = comparison['branch_miss_rate']
    lines.append(f"{'  Branch Miss Rate':<25} {brm['numpy']*100:>14.2f}% {brm['numba']*100:>14.2f}% {brm['diff']*100:>+10.2f}%")

    lines.append("")
    lines.append("Instructions/Cycles:")

    # Instructions
    ins = comparison['instructions']
    lines.append(f"{'  Instructions':<25} {ins['numpy']:>15,} {ins['numba']:>15,} {ins['ratio']:>11.2f}x")

    # Cycles
    cyc = comparison['cycles']
    lines.append(f"{'  Cycles':<25} {cyc['numpy']:>15,} {cyc['numba']:>15,} {cyc['ratio']:>11.2f}x")

    return '\n'.join(lines)


def run_comparison(func_name: str, size: int, cpu: int, iterations: int = 100, repeats: int = 3) -> Dict:
    """Run full comparison for a function."""
    print(f"Running {func_name} (size={size:,}) on CPU {cpu}...")
    print(f"  NumPy...", end=' ', flush=True)

    np_script = generate_numpy_script(func_name, size, iterations)
    np_counters = run_with_perf(np_script, cpu, repeats)
    print(f"done ({np_counters.time_seconds:.3f}s)")

    print(f"  Numba...", end=' ', flush=True)
    nb_script = generate_numba_script(func_name, size, iterations)
    nb_counters = run_with_perf(nb_script, cpu, repeats)
    print(f"done ({nb_counters.time_seconds:.3f}s)")

    comparison = compare_counters(np_counters, nb_counters)
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare hardware counters: NumPy vs Numba')
    parser.add_argument('--cpu', type=int, default=7, help='CPU to pin to')
    parser.add_argument('--function', '-f', type=str, default='np.sum',
                       help='Function to benchmark (e.g., np.sum, np.sin)')
    parser.add_argument('--size', '-s', type=int, default=100000, help='Array size')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Iterations per measurement')
    parser.add_argument('--repeats', '-r', type=int, default=3,
                       help='perf stat repeats for averaging')
    parser.add_argument('--all', action='store_true', help='Run all supported functions')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')

    args = parser.parse_args()

    all_funcs = [
        'np.sum', 'np.mean', 'np.min', 'np.max', 'np.argmin', 'np.argmax',
        'np.sin', 'np.cos', 'np.exp', 'np.sqrt', 'np.tanh',
        'np.add', 'np.multiply', 'np.cumsum', 'np.all', 'np.any', 'np.maximum'
    ]

    if args.all:
        functions = all_funcs
    else:
        functions = [args.function]

    results = {}
    for func in functions:
        try:
            comparison = run_comparison(
                func, args.size, args.cpu,
                args.iterations, args.repeats
            )
            results[func] = comparison
            print(format_comparison(func, args.size, comparison))
        except Exception as e:
            print(f"Error running {func}: {e}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
