"""Discover runtimes that provide LLVM 22's vector-library ABI exports.

The export requirements must be reviewed when LLVM's mappings change.
"""

import ctypes
from ctypes.util import find_library
import functools
import os
import platform
import sys

from llvmlite import binding as llvm


_LIBRARIES = {
    'darwin_libsystem_m': 'System',
    'accelerate': 'Accelerate',
    'svml': 'svml',
}


def _candidates():
    machine = platform.machine().lower()
    if sys.platform == 'darwin' and machine in ('arm64', 'x86_64'):
        return ('darwin_libsystem_m', 'accelerate') + (
            ('svml',) if machine == 'x86_64' else ())
    if (sys.platform.startswith(('linux', 'win')) and
            machine in ('x86_64', 'amd64')):
        return ('svml',)
    return ()


def _symbols(provider):
    if provider == 'svml':
        functions = 'sin cos tan pow exp exp2 log log2 log10 sqrt'.split()
        return tuple(f'__svml_{name}{suffix}' for name in functions
                     for suffix in ('2', '4', '8', 'f4', 'f8', 'f16'))
    if provider == 'accelerate':
        functions = ('ceil fabs floor sqrt exp expm1 log log1p log10 logb '
                     'sin cos tan asin acos atan atan2 sinh cosh tanh '
                     'asinh acosh atanh').split()
        return tuple(f'v{name}f' for name in functions)
    functions = ('exp acos asin atan atan2 cos sin tan cbrt erf pow '
                 'sinh cosh tanh asinh acosh atanh').split()
    return tuple(f'_simd_{name}_{suffix}' for name in functions
                 for suffix in ('d2', 'f4'))


@functools.lru_cache(maxsize=None)
def _load(provider, path):
    try:
        library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        for symbol in _symbols(provider):
            getattr(library, symbol)
        llvm.load_library_permanently(path)
    except (OSError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot load vector library {provider!r} from {path!r}: {exc}"
        ) from exc


def resolve(requested='auto', path=None):
    """Load a compatible runtime and return ``(provider, runtime_path)``."""
    choices = ('auto', 'none', *_LIBRARIES)
    if not isinstance(requested, str) or requested not in choices:
        raise ValueError(
            f"Invalid NUMBA_VECTOR_LIB value {requested!r}; "
            f"expected one of {choices}"
        )
    if path is not None:
        if requested in ('auto', 'none'):
            raise ValueError(
                "NUMBA_VECTOR_LIB_PATH requires an explicit vector library"
            )
        path = os.fsdecode(os.fspath(path))
        if not path or '\0' in path:
            raise ValueError("NUMBA_VECTOR_LIB_PATH must be a nonempty path")
        path = os.path.abspath(path)
    if requested == 'none':
        return ('none', None)
    candidates = _candidates()
    if requested != 'auto' and requested not in candidates:
        raise RuntimeError(
            f"Vector library {requested!r} is unsupported on "
            f"{sys.platform}/{platform.machine()}"
        )
    for provider in candidates if requested == 'auto' else (requested,):
        if provider == 'svml' and not llvm.targets.has_svml():
            if requested == 'auto':
                continue
            raise RuntimeError(
                "SVML requires LLVM support for its calling convention and "
                "vector-width legalization; this llvmlite build does not "
                "provide it")
        name = _LIBRARIES[provider]
        if provider == 'svml' and sys.platform == 'win32':
            name = 'svml_dispmd'
        runtime = path or find_library(name)
        if runtime is None:
            if requested == 'auto':
                continue
            raise RuntimeError(
                f"Vector library {provider!r} could not be found"
            )
        try:
            _load(provider, runtime)
        except RuntimeError:
            if requested == 'auto':
                continue
            raise
        return (provider, runtime)
    return ('none', None)
