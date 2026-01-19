#!/usr/bin/env python3
"""
Compare NumPy vs Numba with hardware counters.
Properly warms up Numba to exclude compilation time.
"""

import subprocess
import tempfile
import os
import sys
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List

PERF_EVENTS = [
    'cycles',
    'instructions',
    'L1-dcache-loads',
    'L1-dcache-load-misses',
    'branches',
    'branch-misses',
]

@dataclass
class PerfCounters:
    cycles: int = 0
    instructions: int = 0
    l1_loads: int = 0
    l1_load_misses: int = 0
    branches: int = 0
    branch_misses: int = 0
    time_seconds: float = 0.0

    @property
    def ipc(self) -> float:
        return self.instructions / self.cycles if self.cycles > 0 else 0

    @property
    def cycles_per_element(self) -> float:
        return 0  # Set externally


def parse_perf_output(output: str) -> PerfCounters:
    counters = PerfCounters()
    for line in output.split('\n'):
        if '<not counted>' in line:
            continue
        match = re.match(r'^\s*([\d,]+)\s+(?:cpu_\w+/)?([\w-]+)/?', line)
        if match:
            value = int(match.group(1).replace(',', ''))
            event = match.group(2)
            if event == 'cycles':
                counters.cycles += value
            elif event == 'instructions':
                counters.instructions += value
            elif event == 'L1-dcache-loads':
                counters.l1_loads += value
            elif event == 'L1-dcache-load-misses':
                counters.l1_load_misses += value
            elif event == 'branches':
                counters.branches += value
            elif event == 'branch-misses':
                counters.branch_misses += value
        if 'seconds time elapsed' in line:
            time_match = re.match(r'^\s*([\d.]+)', line)
            if time_match:
                counters.time_seconds = float(time_match.group(1))
    return counters


def run_perf(script_content: str, cpu: int, repeats: int = 5) -> PerfCounters:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        path = f.name

    try:
        cmd = [
            'perf', 'stat', '-e', ','.join(PERF_EVENTS),
            '-r', str(repeats), '--',
            'taskset', '-c', str(cpu),
            sys.executable, path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return parse_perf_output(result.stderr)
    finally:
        os.unlink(path)


def compare_function(func_name: str, size: int, cpu: int, iterations: int = 100):
    """Compare NumPy vs Numba for a function."""

    # Define function-specific setup
    configs = {
        'np.min': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.min(x)',
            'numba_impl': 'def nb_func(x): return np.min(x)',
        },
        'np.max': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.max(x)',
            'numba_impl': 'def nb_func(x): return np.max(x)',
        },
        'np.std': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.std(x)',
            'numba_impl': 'def nb_func(x): return np.std(x)',
        },
        'np.var': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.var(x)',
            'numba_impl': 'def nb_func(x): return np.var(x)',
        },
        'np.diff': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.diff(x)',
            'numba_impl': 'def nb_func(x): return np.diff(x)',
        },
        'np.all': {
            'setup': 'x = np.ones(SIZE, dtype=bool); x[SIZE//2] = False',
            'numpy_call': 'np.all(x)',
            'numba_impl': 'def nb_func(x): return np.all(x)',
        },
        'np.any': {
            'setup': 'x = np.zeros(SIZE, dtype=bool); x[SIZE//2] = True',
            'numpy_call': 'np.any(x)',
            'numba_impl': 'def nb_func(x): return np.any(x)',
        },
        'np.sum': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.sum(x)',
            'numba_impl': 'def nb_func(x): return np.sum(x)',
        },
        'np.argmin': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.argmin(x)',
            'numba_impl': 'def nb_func(x): return np.argmin(x)',
        },
        'np.argmax': {
            'setup': 'x = np.random.randn(SIZE)',
            'numpy_call': 'np.argmax(x)',
            'numba_impl': 'def nb_func(x): return np.argmax(x)',
        },
    }

    if func_name not in configs:
        raise ValueError(f"Unknown function: {func_name}")

    cfg = configs[func_name]

    # NumPy script
    numpy_script = f'''
import numpy as np
np.random.seed(42)
SIZE = {size}
ITERATIONS = {iterations}
{cfg['setup']}

# Warmup
for _ in range(5):
    _ = {cfg['numpy_call']}

# Timed
for _ in range(ITERATIONS):
    _ = {cfg['numpy_call']}
'''

    # Numba script - pre-compile before timing
    numba_script = f'''
import numpy as np
from numba import njit
np.random.seed(42)
SIZE = {size}
ITERATIONS = {iterations}

@njit(fastmath=True)
{cfg['numba_impl']}

{cfg['setup']}

# Warmup + compile
for _ in range(20):
    _ = nb_func(x)

# Timed
for _ in range(ITERATIONS):
    _ = nb_func(x)
'''

    print(f"  NumPy...", end=' ', flush=True)
    np_counters = run_perf(numpy_script, cpu)
    print(f"{np_counters.time_seconds:.3f}s")

    print(f"  Numba...", end=' ', flush=True)
    nb_counters = run_perf(numba_script, cpu)
    print(f"{nb_counters.time_seconds:.3f}s")

    return np_counters, nb_counters


def format_comparison(func_name: str, size: int, np_c: PerfCounters, nb_c: PerfCounters) -> str:
    """Format comparison as readable text."""

    speedup = np_c.time_seconds / nb_c.time_seconds if nb_c.time_seconds > 0 else 0
    speedup_str = f"Numba {speedup:.2f}x faster" if speedup > 1 else f"NumPy {1/speedup:.2f}x faster"

    cycles_ratio = np_c.cycles / nb_c.cycles if nb_c.cycles > 0 else 0
    instr_ratio = np_c.instructions / nb_c.instructions if nb_c.instructions > 0 else 0

    # Per-element metrics (for iterations * size elements processed)
    total_elements = size * 100  # 100 iterations
    np_cycles_per_elem = np_c.cycles / total_elements
    nb_cycles_per_elem = nb_c.cycles / total_elements
    np_instr_per_elem = np_c.instructions / total_elements
    nb_instr_per_elem = nb_c.instructions / total_elements

    lines = [
        "",
        "=" * 70,
        f"  {func_name} (size={size:,}) - {speedup_str}",
        "=" * 70,
        "",
        f"{'Metric':<25} {'NumPy':>15} {'Numba':>15} {'Ratio':>10}",
        "-" * 70,
        f"{'Time (seconds)':<25} {np_c.time_seconds:>15.4f} {nb_c.time_seconds:>15.4f} {speedup:>9.2f}x",
        f"{'Cycles (total)':<25} {np_c.cycles:>15,} {nb_c.cycles:>15,} {cycles_ratio:>9.2f}x",
        f"{'Cycles/element':<25} {np_cycles_per_elem:>15.2f} {nb_cycles_per_elem:>15.2f} {np_cycles_per_elem/nb_cycles_per_elem:>9.2f}x",
        f"{'Instructions (total)':<25} {np_c.instructions:>15,} {nb_c.instructions:>15,} {instr_ratio:>9.2f}x",
        f"{'Instructions/element':<25} {np_instr_per_elem:>15.2f} {nb_instr_per_elem:>15.2f} {np_instr_per_elem/nb_instr_per_elem:>9.2f}x",
        f"{'IPC':<25} {np_c.ipc:>15.2f} {nb_c.ipc:>15.2f} {np_c.ipc/nb_c.ipc if nb_c.ipc else 0:>9.2f}x",
        "",
        f"{'L1 Cache Loads':<25} {np_c.l1_loads:>15,} {nb_c.l1_loads:>15,} {np_c.l1_loads/nb_c.l1_loads if nb_c.l1_loads else 0:>9.2f}x",
        f"{'L1 Cache Misses':<25} {np_c.l1_load_misses:>15,} {nb_c.l1_load_misses:>15,}",
        f"{'L1 Miss Rate':<25} {100*np_c.l1_load_misses/np_c.l1_loads if np_c.l1_loads else 0:>14.2f}% {100*nb_c.l1_load_misses/nb_c.l1_loads if nb_c.l1_loads else 0:>14.2f}%",
        "",
        f"{'Branches':<25} {np_c.branches:>15,} {nb_c.branches:>15,} {np_c.branches/nb_c.branches if nb_c.branches else 0:>9.2f}x",
        f"{'Branch Misses':<25} {np_c.branch_misses:>15,} {nb_c.branch_misses:>15,}",
        f"{'Branch Miss Rate':<25} {100*np_c.branch_misses/np_c.branches if np_c.branches else 0:>14.2f}% {100*nb_c.branch_misses/nb_c.branches if nb_c.branches else 0:>14.2f}%",
    ]

    return '\n'.join(lines)


def explain_performance(func_name: str, np_c: PerfCounters, nb_c: PerfCounters, size: int) -> str:
    """Generate human-readable explanation of performance difference."""

    speedup = np_c.time_seconds / nb_c.time_seconds if nb_c.time_seconds > 0 else 0

    explanations = []

    # Cycles per element
    np_cpe = np_c.cycles / (size * 100)
    nb_cpe = nb_c.cycles / (size * 100)

    if speedup > 1:
        explanations.append(f"Numba is {speedup:.1f}x faster than NumPy.")

        if nb_c.ipc > np_c.ipc:
            explanations.append(f"- Higher IPC ({nb_c.ipc:.2f} vs {np_c.ipc:.2f}) indicates better instruction-level parallelism")

        if nb_cpe < np_cpe:
            explanations.append(f"- Fewer cycles per element ({nb_cpe:.1f} vs {np_cpe:.1f}) shows more efficient computation")

    else:
        explanations.append(f"NumPy is {1/speedup:.1f}x faster than Numba.")

        np_instr_per_elem = np_c.instructions / (size * 100)
        nb_instr_per_elem = nb_c.instructions / (size * 100)

        if nb_instr_per_elem > np_instr_per_elem * 1.5:
            explanations.append(f"- Numba executes {nb_instr_per_elem/np_instr_per_elem:.1f}x more instructions per element")
            explanations.append("  This suggests NumPy uses SIMD to process multiple elements per instruction")

        if nb_c.ipc < np_c.ipc:
            explanations.append(f"- Lower IPC ({nb_c.ipc:.2f} vs {np_c.ipc:.2f}) indicates pipeline stalls or dependencies")

    return '\n'.join(explanations)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=7)
    parser.add_argument('--size', type=int, default=1_000_000)
    parser.add_argument('--functions', '-f', nargs='+',
                        default=['np.min', 'np.max', 'np.std', 'np.var', 'np.diff', 'np.all', 'np.sum'])
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    results = {}

    for func in args.functions:
        print(f"\n{func} (size={args.size:,}):")
        np_c, nb_c = compare_function(func, args.size, args.cpu)
        print(format_comparison(func, args.size, np_c, nb_c))
        print("\nExplanation:")
        print(explain_performance(func, np_c, nb_c, args.size))

        results[func] = {
            'numpy': asdict(np_c),
            'numba': asdict(nb_c),
            'speedup': np_c.time_seconds / nb_c.time_seconds if nb_c.time_seconds else 0,
        }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
