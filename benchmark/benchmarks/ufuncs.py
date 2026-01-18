"""
Element-wise (ufunc) benchmarks: NumPy vs Numba

Tests unary and binary operations on arrays.
"""

import numpy as np
from numba import njit
from harness import Benchmark
from registry import register_benchmark


# =============================================================================
# Helper to create benchmarks for simple ufuncs
# =============================================================================

def make_unary_benchmark(name: str, np_func, setup_range=(-10, 10)):
    """Factory for unary ufunc benchmarks."""

    # Create the JIT function at module load time
    @njit(cache=True, fastmath=True)
    def numba_func(x):
        return np_func(x)

    class UnaryBenchmark(Benchmark):
        def __init__(self):
            super().__init__(f'np.{name}', 'ufunc')
            self._numba_func = numba_func

        def setup(self, size, dtype):
            low, high = setup_range
            if low >= 0:
                return (np.random.uniform(low, high, size).astype(dtype),)
            return (np.random.uniform(low, high, size).astype(dtype),)

        def numpy_impl(self, x):
            return np_func(x)

        def numba_impl(self, x):
            return np_func(x)

        def get_numba_func(self):
            return self._numba_func

    return UnaryBenchmark


def make_binary_benchmark(name: str, np_func, setup_x=(-10, 10), setup_y=(-10, 10)):
    """Factory for binary ufunc benchmarks."""

    @njit(cache=True, fastmath=True)
    def numba_func(x, y):
        return np_func(x, y)

    class BinaryBenchmark(Benchmark):
        def __init__(self):
            super().__init__(f'np.{name}', 'ufunc')
            self._numba_func = numba_func

        def setup(self, size, dtype):
            x_low, x_high = setup_x
            y_low, y_high = setup_y
            return (
                np.random.uniform(x_low, x_high, size).astype(dtype),
                np.random.uniform(y_low, y_high, size).astype(dtype)
            )

        def numpy_impl(self, x, y):
            return np_func(x, y)

        def numba_impl(self, x, y):
            return np_func(x, y)

        def get_numba_func(self):
            return self._numba_func

    return BinaryBenchmark


# =============================================================================
# UNARY MATH OPERATIONS
# =============================================================================

# Create JIT functions for each operation
@njit(cache=True, fastmath=True)
def _nb_sin(x):
    return np.sin(x)

@njit(cache=True, fastmath=True)
def _nb_cos(x):
    return np.cos(x)

@njit(cache=True, fastmath=True)
def _nb_exp(x):
    return np.exp(x)

@njit(cache=True, fastmath=True)
def _nb_log(x):
    return np.log(x)

@njit(cache=True, fastmath=True)
def _nb_sqrt(x):
    return np.sqrt(x)

@njit(cache=True, fastmath=True)
def _nb_abs(x):
    return np.abs(x)

@njit(cache=True, fastmath=True)
def _nb_tanh(x):
    return np.tanh(x)

@njit(cache=True, fastmath=True)
def _nb_floor(x):
    return np.floor(x)

@njit(cache=True, fastmath=True)
def _nb_ceil(x):
    return np.ceil(x)

@njit(cache=True, fastmath=True)
def _nb_expm1(x):
    return np.expm1(x)

@njit(cache=True, fastmath=True)
def _nb_log1p(x):
    return np.log1p(x)

@njit(cache=True, fastmath=True)
def _nb_arcsin(x):
    return np.arcsin(x)

@njit(cache=True, fastmath=True)
def _nb_arccos(x):
    return np.arccos(x)

@njit(cache=True, fastmath=True)
def _nb_arctan(x):
    return np.arctan(x)

@njit(cache=True, fastmath=True)
def _nb_sinh(x):
    return np.sinh(x)

@njit(cache=True, fastmath=True)
def _nb_cosh(x):
    return np.cosh(x)


# Register benchmarks
@register_benchmark('ufunc')
class SinBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.sin', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.sin(x)
    def get_numba_func(self):
        return _nb_sin

@register_benchmark('ufunc')
class CosBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.cos', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.cos(x)
    def get_numba_func(self):
        return _nb_cos

@register_benchmark('ufunc')
class ExpBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.exp', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-10, 10, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.exp(x)
    def get_numba_func(self):
        return _nb_exp

@register_benchmark('ufunc')
class LogBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.log', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(0.1, 100, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.log(x)
    def get_numba_func(self):
        return _nb_log

@register_benchmark('ufunc')
class SqrtBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.sqrt', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(0, 1000, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.sqrt(x)
    def get_numba_func(self):
        return _nb_sqrt

@register_benchmark('ufunc')
class AbsBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.abs', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.abs(x)
    def get_numba_func(self):
        return _nb_abs

@register_benchmark('ufunc')
class TanhBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.tanh', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.tanh(x)
    def get_numba_func(self):
        return _nb_tanh

@register_benchmark('ufunc')
class FloorBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.floor', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-100, 100, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.floor(x)
    def get_numba_func(self):
        return _nb_floor

@register_benchmark('ufunc')
class CeilBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.ceil', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-100, 100, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.ceil(x)
    def get_numba_func(self):
        return _nb_ceil

@register_benchmark('ufunc')
class Expm1Benchmark(Benchmark):
    def __init__(self):
        super().__init__('np.expm1', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-5, 5, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.expm1(x)
    def get_numba_func(self):
        return _nb_expm1

@register_benchmark('ufunc')
class Log1pBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.log1p', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(0, 100, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.log1p(x)
    def get_numba_func(self):
        return _nb_log1p

@register_benchmark('ufunc')
class ArcsinBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.arcsin', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-0.99, 0.99, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.arcsin(x)
    def get_numba_func(self):
        return _nb_arcsin

@register_benchmark('ufunc')
class ArccosBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.arccos', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-0.99, 0.99, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.arccos(x)
    def get_numba_func(self):
        return _nb_arccos

@register_benchmark('ufunc')
class ArctanBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.arctan', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype),)
    def numpy_impl(self, x):
        return np.arctan(x)
    def get_numba_func(self):
        return _nb_arctan

@register_benchmark('ufunc')
class SinhBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.sinh', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-5, 5, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.sinh(x)
    def get_numba_func(self):
        return _nb_sinh

@register_benchmark('ufunc')
class CoshBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.cosh', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(-5, 5, size).astype(dtype),)
    def numpy_impl(self, x):
        return np.cosh(x)
    def get_numba_func(self):
        return _nb_cosh


# =============================================================================
# BINARY MATH OPERATIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def _nb_add(x, y):
    return np.add(x, y)

@njit(cache=True, fastmath=True)
def _nb_multiply(x, y):
    return np.multiply(x, y)

@njit(cache=True, fastmath=True)
def _nb_divide(x, y):
    return np.divide(x, y)

@njit(cache=True, fastmath=True)
def _nb_power(x, y):
    return np.power(x, y)

@njit(cache=True, fastmath=True)
def _nb_maximum(x, y):
    return np.maximum(x, y)

@njit(cache=True, fastmath=True)
def _nb_minimum(x, y):
    return np.minimum(x, y)

@njit(cache=True, fastmath=True)
def _nb_arctan2(y, x):
    return np.arctan2(y, x)

@njit(cache=True, fastmath=True)
def _nb_hypot(x, y):
    return np.hypot(x, y)


@register_benchmark('ufunc')
class AddBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.add', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.add(x, y)
    def get_numba_func(self):
        return _nb_add

@register_benchmark('ufunc')
class MultiplyBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.multiply', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.multiply(x, y)
    def get_numba_func(self):
        return _nb_multiply

@register_benchmark('ufunc')
class DivideBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.divide', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.uniform(0.1, 10, size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.divide(x, y)
    def get_numba_func(self):
        return _nb_divide

@register_benchmark('ufunc')
class PowerBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.power', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.uniform(0.1, 10, size).astype(dtype), np.random.uniform(0.5, 3, size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.power(x, y)
    def get_numba_func(self):
        return _nb_power

@register_benchmark('ufunc')
class MaximumBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.maximum', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.maximum(x, y)
    def get_numba_func(self):
        return _nb_maximum

@register_benchmark('ufunc')
class MinimumBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.minimum', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.minimum(x, y)
    def get_numba_func(self):
        return _nb_minimum

@register_benchmark('ufunc')
class Arctan2Benchmark(Benchmark):
    def __init__(self):
        super().__init__('np.arctan2', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, y, x):
        return np.arctan2(y, x)
    def get_numba_func(self):
        return _nb_arctan2

@register_benchmark('ufunc')
class HypotBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.hypot', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.hypot(x, y)
    def get_numba_func(self):
        return _nb_hypot


# =============================================================================
# COMPARISON / FLOATING POINT OPERATIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def _nb_greater(x, y):
    return np.greater(x, y)

@njit(cache=True, fastmath=True)
def _nb_isnan(x):
    return np.isnan(x)

@njit(cache=True, fastmath=True)
def _nb_isfinite(x):
    return np.isfinite(x)


@register_benchmark('ufunc')
class GreaterBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.greater', 'ufunc')
    def setup(self, size, dtype):
        return (np.random.randn(size).astype(dtype), np.random.randn(size).astype(dtype))
    def numpy_impl(self, x, y):
        return np.greater(x, y)
    def get_numba_func(self):
        return _nb_greater

@register_benchmark('ufunc')
class IsnanBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.isnan', 'ufunc')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.nan
        return (x,)
    def numpy_impl(self, x):
        return np.isnan(x)
    def get_numba_func(self):
        return _nb_isnan

@register_benchmark('ufunc')
class IsfiniteBenchmark(Benchmark):
    def __init__(self):
        super().__init__('np.isfinite', 'ufunc')
    def setup(self, size, dtype):
        x = np.random.randn(size).astype(dtype)
        x[::100] = np.inf
        return (x,)
    def numpy_impl(self, x):
        return np.isfinite(x)
    def get_numba_func(self):
        return _nb_isfinite
