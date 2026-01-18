"""
Reduction benchmarks: NumPy vs Numba

Tests operations that reduce arrays to scalars or smaller arrays.
"""

import numpy as np
from numba import njit
from harness import Benchmark
from registry import register_benchmark


# =============================================================================
# JIT FUNCTIONS FOR REDUCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def _nb_sum(x):
    return np.sum(x)

@njit(cache=True, fastmath=True)
def _nb_prod(x):
    return np.prod(x)

@njit(cache=True, fastmath=True)
def _nb_mean(x):
    return np.mean(x)

@njit(cache=True, fastmath=True)
def _nb_std(x):
    return np.std(x)

@njit(cache=True, fastmath=True)
def _nb_var(x):
    return np.var(x)

@njit(cache=True, fastmath=True)
def _nb_min(x):
    return np.min(x)

@njit(cache=True, fastmath=True)
def _nb_max(x):
    return np.max(x)

@njit(cache=True, fastmath=True)
def _nb_argmin(x):
    return np.argmin(x)

@njit(cache=True, fastmath=True)
def _nb_argmax(x):
    return np.argmax(x)

@njit(cache=True, fastmath=True)
def _nb_all(x):
    return np.all(x)

@njit(cache=True, fastmath=True)
def _nb_any(x):
    return np.any(x)

@njit(cache=True, fastmath=True)
def _nb_cumsum(x):
    return np.cumsum(x)

@njit(cache=True, fastmath=True)
def _nb_cumprod(x):
    return np.cumprod(x)

@njit(cache=True, fastmath=True)
def _nb_nansum(x):
    return np.nansum(x)

@njit(cache=True, fastmath=True)
def _nb_nanmean(x):
    return np.nanmean(x)

@njit(cache=True, fastmath=True)
def _nb_nanstd(x):
    return np.nanstd(x)

@njit(cache=True, fastmath=True)
def _nb_nanmin(x):
    return np.nanmin(x)

@njit(cache=True, fastmath=True)
def _nb_nanmax(x):
    return np.nanmax(x)

@njit(cache=True, fastmath=True)
def _nb_diff(x):
    return np.diff(x)

@njit(cache=True, fastmath=True)
def _nb_count_nonzero(x):
    return np.count_nonzero(x)


# =============================================================================
# BASIC REDUCTIONS
# =============================================================================

@register_benchmark('reduction')
class SumBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.sum', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.sum(x)
    def get_numba_func(self):
        return _nb_sum

@register_benchmark('reduction')
class ProdBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.prod', 'reduction')
    def setup(self, size, dtype):
        return (np.random.uniform(0.999, 1.001, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.prod(x)
    def get_numba_func(self):
        return _nb_prod

@register_benchmark('reduction')
class MeanBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.mean', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.mean(x)
    def get_numba_func(self):
        return _nb_mean

@register_benchmark('reduction')
class StdBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.std', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.std(x)
    def get_numba_func(self):
        return _nb_std

@register_benchmark('reduction')
class VarBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.var', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.var(x)
    def get_numba_func(self):
        return _nb_var

@register_benchmark('reduction')
class MinBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.min', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.min(x)
    def get_numba_func(self):
        return _nb_min

@register_benchmark('reduction')
class MaxBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.max', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.max(x)
    def get_numba_func(self):
        return _nb_max

@register_benchmark('reduction')
class ArgminBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.argmin', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.argmin(x)
    def get_numba_func(self):
        return _nb_argmin

@register_benchmark('reduction')
class ArgmaxBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.argmax', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.argmax(x)
    def get_numba_func(self):
        return _nb_argmax


# =============================================================================
# LOGICAL REDUCTIONS
# =============================================================================

@register_benchmark('reduction')
class AllBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.all', 'reduction')
    def setup(self, size, dtype):
        x = np.ones(size, dtype=bool)
        x[size // 2] = False
        return (x,)
    def numpy_impl(self, x):
        return np.all(x)
    def get_numba_func(self):
        return _nb_all

@register_benchmark('reduction')
class AnyBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.any', 'reduction')
    def setup(self, size, dtype):
        x = np.zeros(size, dtype=bool)
        x[size // 2] = True
        return (x,)
    def numpy_impl(self, x):
        return np.any(x)
    def get_numba_func(self):
        return _nb_any


# =============================================================================
# CUMULATIVE OPERATIONS
# =============================================================================

@register_benchmark('reduction')
class CumsumBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.cumsum', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.cumsum(x)
    def get_numba_func(self):
        return _nb_cumsum

@register_benchmark('reduction')
class CumprodBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.cumprod', 'reduction')
    def setup(self, size, dtype):
        return (np.random.uniform(0.9999, 1.0001, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.cumprod(x)
    def get_numba_func(self):
        return _nb_cumprod


# =============================================================================
# NAN-AWARE REDUCTIONS
# =============================================================================

@register_benchmark('reduction')
class NansumBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.nansum', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.nansum(x)
    def get_numba_func(self):
        return _nb_nansum

@register_benchmark('reduction')
class NanmeanBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.nanmean', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.nanmean(x)
    def get_numba_func(self):
        return _nb_nanmean

@register_benchmark('reduction')
class NanstdBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.nanstd', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.nanstd(x)
    def get_numba_func(self):
        return _nb_nanstd

@register_benchmark('reduction')
class NanminBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.nanmin', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.nanmin(x)
    def get_numba_func(self):
        return _nb_nanmin

@register_benchmark('reduction')
class NanmaxBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.nanmax', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.nanmax(x)
    def get_numba_func(self):
        return _nb_nanmax


# =============================================================================
# OTHER
# =============================================================================

@register_benchmark('reduction')
class DiffBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.diff', 'reduction')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.diff(x)
    def get_numba_func(self):
        return _nb_diff

@register_benchmark('reduction')
class CountNonzeroBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.count_nonzero', 'reduction')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[x < 0] = 0
        return (x,)
    def numpy_impl(self, x):
        return np.count_nonzero(x)
    def get_numba_func(self):
        return _nb_count_nonzero
