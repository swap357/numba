# NumPy vs Numba Benchmark Suite

**Date:** 2026-01-18

Benchmark harness comparing NumPy functions against Numba-compiled equivalents.

## Usage

```bash
# Activate environment
source /root/miniconda3/bin/activate numba63

# Run benchmarks (pinned to CPU 7)
python harness.py --cpu 7 --sizes 10000,100000,1000000 --runs 30

# Quick test
python harness.py --cpu 7 --sizes 10000 --runs 10
```

## Key Findings (2026-01-18)

### Where Numba Wins

| Function | Speedup | Notes |
|----------|---------|-------|
| np.cumsum | **4.3x** | Consistent across all sizes |
| np.cumprod | **2.3x** | Consistent across all sizes |
| np.count_nonzero | **1.6x** | Consistent |
| np.add/multiply | **1.1-1.8x** | Less Python call overhead |
| np.hypot | **1.3x** | |
| np.maximum/minimum | **1.1-1.5x** | |

### Where NumPy Wins

| Function | NumPy Factor | Why |
|----------|--------------|-----|
| np.all/any | **3-30x** | Early-exit + SIMD |
| np.min/max | **3-5x** | SIMD optimized |
| np.argmin/argmax | **3-5x** | SIMD optimized |
| np.tanh | **2x** | Likely SVML library |
| np.diff (large) | **9x** | Memory allocation |

### Roughly Even (within 10%)

- Transcendentals: sin, cos, exp, log, arcsin, arccos
- Simple ops: sqrt, floor, ceil, abs
- Binary ops: divide, power

## Architecture

```
benchmark/
├── harness.py          # Main benchmark runner
├── registry.py         # Benchmark registry (singleton)
├── benchmarks/
│   ├── ufuncs.py      # Element-wise operations (27 tests)
│   └── reductions.py  # Reduction operations (20 tests)
├── perf/
│   └── counters.py    # Hardware counter comparison (perf stat)
└── results/           # Output CSV, markdown, and JSON
```

## Hardware Counter Analysis

Use `perf/counters.py` to compare hardware-level behavior:

```bash
# Single function
python perf/counters.py --cpu 7 --function np.sum --size 1000000

# All supported functions
python perf/counters.py --cpu 7 --all --size 1000000 --output results/counters.json
```

### Key Hardware Insights

| Function | Winner | Why (Hardware Perspective) |
|----------|--------|----------------------------|
| np.sum/mean | NumPy 3x | 3x fewer instructions (SIMD), better IPC |
| np.min/max | NumPy 5x | 10x fewer instructions via SIMD |
| np.all/any | NumPy 6x | Early exit: 7x fewer instructions |
| np.cumsum | Numba 1.1x | Better IPC (1.47 vs 1.02) despite more instructions |
| np.tanh | NumPy 2.4x | 2.4x fewer branches, likely SVML |

**Pattern**: NumPy wins when SIMD can reduce instruction count significantly. Numba wins when it can improve IPC through better code generation for sequential operations.

## Notes

1. **CPU Pinning**: Use `--cpu N` to pin to a specific core. Check `mpstat -P ALL 1 1` for idle cores.

2. **Correctness**: Some nan* functions show MISMATCH due to strict hash comparison. Results are still numerically correct.

3. **SVML**: NumPy may use Intel SVML for transcendentals, giving it an edge on those functions.

4. **Early Exit**: NumPy's all/any implementations short-circuit, while Numba's JIT code processes all elements.

5. **Memory**: For large arrays, NumPy's in-place SIMD operations can be faster than Numba's allocate-and-fill pattern.

## Next Steps

- [x] Add perf stat integration for hardware counters
- [ ] Add compound expression benchmarks
- [ ] Test with float32
- [ ] Add parallel=True variants
