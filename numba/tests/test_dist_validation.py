import glob
import os
import re
import platform
import unittest
from typing import Any, Callable, Optional
from numba.tests.support import TestCase

EXPECTED_IMPORTS_BY_PLATFORM = {
  "win-64": {
    "imports_by_ext": {
      "_dynfunc.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_helperlib.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "python310.dll"
      ],
      "_devicearray.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "mviewbuf.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_dispatcher.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_internal.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-string-l1-1-0.dll",
        "python310.dll"
      ],
      "omppool.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCOMP140.DLL",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "python310.dll"
      ],
      "workqueue.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "python310.dll"
      ],
      "tbbpool.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "python310.dll",
        "tbb12.dll"
      ],
      "_num_threads.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_extras.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_typeconv.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ],
      "_nrt_python.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "python310.dll"
      ],
      "_box.cp310-win_amd64.pyd": [
        "KERNEL32.dll",
        "VCRUNTIME140.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "python310.dll"
      ]
    },
    "delay_imports_by_ext": {}
  },
  "linux-64": {
    "imports_by_ext": {
      "_devicearray.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_dynfunc.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_helperlib.cpython-310-x86_64-linux-gnu.so": [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0"
      ],
      "mviewbuf.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_dispatcher.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_internal.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0"
      ],
      "omppool.cpython-310-x86_64-linux-gnu.so": [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libgcc_s.so.1",
        "libgomp.so.1.0.0",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "tbbpool.cpython-310-x86_64-linux-gnu.so": [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6",
        "libtbb.so.12"
      ],
      "_num_threads.cpython-310-x86_64-linux-gnu.so": [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libpthread.so.0"
      ],
      "workqueue.cpython-310-x86_64-linux-gnu.so": [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_extras.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_typeconv.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_nrt_python.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_box.cpython-310-x86_64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ]
    },
    "delay_imports_by_ext": {}
  },
  "linux-aarch64": {
    "imports_by_ext": {
      "_dynfunc.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_dispatcher.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_devicearray.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_helperlib.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0"
      ],
      "mviewbuf.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "omppool.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libgomp.so.1.0.0",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_num_threads.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_internal.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libm.so.6",
        "libpthread.so.0"
      ],
      "workqueue.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_extras.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ],
      "_typeconv.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_nrt_python.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6"
      ],
      "_box.cpython-310-aarch64-linux-gnu.so": [
        "libc.so.6",
        "libpthread.so.0"
      ]
    },
    "delay_imports_by_ext": {}
  },
  "osx-arm64": {
    "imports_by_ext": {
      "_devicearray.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib"
      ],
      "_dispatcher.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib"
      ],
      "_dynfunc.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "mviewbuf.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "_helperlib.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "workqueue.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib"
      ],
      "_num_threads.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "_internal.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "omppool.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib",
        "libomp.dylib"
      ],
      "_extras.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ],
      "_typeconv.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib"
      ],
      "_nrt_python.cpython-310-darwin.so": [
        "libSystem.B.dylib",
        "libc++.1.dylib"
      ],
      "_box.cpython-310-darwin.so": [
        "libSystem.B.dylib"
      ]
    },
    "delay_imports_by_ext": {}
  }
}

# Gate these tests behind an environment variable to ensure they only run for
# distribution validation by maintainers.
_RUN_DIST_TESTS = os.environ.get("NUMBA_DIST_TEST")

try:
    import lief  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    _HAS_LIEF = False
    _lief_parse: Optional[Callable[[str], Any]] = None
else:
    _HAS_LIEF = True
    try:
        _lief_parse = getattr(lief, "parse")  # type: ignore[attr-defined]
    except AttributeError:
        _lief_parse = None


def _lief_parse_safe(path: str):
    parser = _lief_parse
    if parser is None:
        return None
    try:
        return parser(path)
    except (RuntimeError, ValueError, OSError):
        return None


def _numba_package_dir() -> str:
    import numba

    return os.path.dirname(numba.__file__)


def _iter_extension_binaries() -> list[str]:
    base = _numba_package_dir()
    patterns = ["**/*.so", "**/*.pyd", "**/*.dll"]
    binaries: list[str] = []
    for pat in patterns:
        binaries.extend(glob.glob(os.path.join(base, pat), recursive=True))
    return [p for p in binaries if os.path.isfile(p)]


def _extract_library_names(parsed) -> set[str]:
    names: set[str] = set()
    if hasattr(parsed, "libraries"):
        for lib in getattr(parsed, "libraries"):
            if isinstance(lib, str):
                names.add(lib)
            else:
                name = getattr(lib, "name", None)
                if name:
                    names.add(str(name))
    if hasattr(parsed, "imported_libraries"):
        for lib in getattr(parsed, "imported_libraries"):
            names.add(str(lib))
    return names


def _platform_key() -> str | None:
    os_name = platform.system().lower()
    machine = platform.machine().lower()

    platform_map = {
        ("linux", "x86_64"): "linux-64",
        ("linux", "aarch64"): "linux-aarch64",
        ("darwin", "arm64"): "osx-arm64",
        ("darwin", "x86_64"): "osx-64",
        ("windows", "amd64"): "win-64",
        ("windows", "x86_64"): "win-64",
    }

    return platform_map.get((os_name, machine))


def _load_expected_imports() -> dict[str, list[str]] | None:
    key = _platform_key()
    if key is None:
        return None
    entry = EXPECTED_IMPORTS_BY_PLATFORM.get(key, {})
    return entry.get("imports_by_ext", {})


_PY_DLL_RE = re.compile(r"python3\d{2}\.dll", re.IGNORECASE)


def _normalize_lib_name(name: str) -> str:
    base = os.path.basename(name).lower()
    if _PY_DLL_RE.fullmatch(base):
        return "python3xx.dll"
    return base


def _canonicalize(names: set[str]) -> set[str]:
    return {_normalize_lib_name(n) for n in names}


_CPYTHON_TAG_RE = re.compile(r"cpython-3\d+")
_CP_TAG_RE = re.compile(r"cp3\d{2}")


def _normalize_ext_name(filename: str) -> str:
    name = os.path.basename(filename)
    name = _CPYTHON_TAG_RE.sub("cpython-3xx", name)
    name = _CP_TAG_RE.sub("cp3xx", name)
    return name


needs_lief = unittest.skipUnless(_HAS_LIEF, "lief is required for distribution validation")


@unittest.skipUnless(_RUN_DIST_TESTS, "Distribution-specific test")
@needs_lief
class TestNumbaDistValidation(TestCase):

    def test_expected_extensions(self):
        expected = _load_expected_imports()
        if not expected:
            self.skipTest("No expected import map for this platform")

        expected_exts = {_normalize_ext_name(k) for k in expected.keys()}
        found_exts = {_normalize_ext_name(p) for p in _iter_extension_binaries()}
        missing = sorted(expected_exts - found_exts)
        if missing:
            msg = (
                "Missing expected Numba extension binaries:\n"
                f"Missing: {missing}\n"
                f"Found: {sorted(found_exts)}\n"
            )
            raise AssertionError(msg)

    def test_expected_imports_by_extension(self):
        expected = _load_expected_imports()
        if not expected:  # None or empty dict
            self.skipTest("No expected import map for this platform")

        # Build a mapping from normalized extension filename to its imported libraries
        found: dict[str, set[str]] = {}
        for path in _iter_extension_binaries():
            parsed = _lief_parse_safe(path)
            if parsed is None:
                found[_normalize_ext_name(path)] = set()
                continue
            libs = _extract_library_names(parsed)
            found[_normalize_ext_name(path)] = _canonicalize(libs)

        mismatches: list[str] = []
        for ext_name, exp_libs in expected.items():
            ext_name = _normalize_ext_name(ext_name)
            if ext_name not in found:
                continue
            got = found[ext_name]
            exp = _canonicalize(set(exp_libs))
            if exp != got:
                mismatches.append(
                    (
                        f"Unexpected imports for {ext_name}:\n"
                        f"Expected: {sorted(exp)}\n"
                        f"     Got: {sorted(got)}\n"
                        f"Difference: {sorted((exp ^ got))}\n"
                        f"Only in Expected: {sorted(exp - got)}\n"
                        f"Only in Got: {sorted(got - exp)}\n"
                    )
                )

        if mismatches:
            raise AssertionError("\n\n".join(mismatches))
