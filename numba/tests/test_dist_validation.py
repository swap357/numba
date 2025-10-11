import os
import platform
import re
import unittest
from numba.tests.support import TestCase
from numba.core import config


_HAVE_LIEF = False
try:
    import lief  # noqa: F401
    _HAVE_LIEF = True
except ImportError:
    pass


needs_lief = unittest.skipUnless(_HAVE_LIEF, "test needs py-lief package")


@unittest.skipUnless(os.environ.get('NUMBA_DIST_TEST'),
                     "Distribution-specific test")
@needs_lief
class TestBuild(TestCase):
    """Test distribution linkage validation for wheels and conda packages"""

    wheel_expected_imports = {
        "windows": {
            "amd64": {
                "_dynfunc.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_helperlib.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-math-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-stdio-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_devicearray.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "mviewbuf.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_dispatcher.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "MSVCP140.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_internal.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-string-l1-1-0.dll",
                    "python310.dll",
                ]),
                "omppool.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "MSVCP140.dll",
                    "VCOMP140.DLL",
                    "VCRUNTIME140.dll",
                    "VCRUNTIME140_1.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-math-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-stdio-l1-1-0.dll",
                    "python310.dll",
                ]),
                "workqueue.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "MSVCP140.dll",
                    "VCRUNTIME140.dll",
                    "VCRUNTIME140_1.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-math-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-stdio-l1-1-0.dll",
                    "python310.dll",
                ]),
                "tbbpool.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "MSVCP140.dll",
                    "VCRUNTIME140.dll",
                    "VCRUNTIME140_1.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-math-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-stdio-l1-1-0.dll",
                    "python310.dll",
                    "tbb12.dll",
                ]),
                "_num_threads.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_extras.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_typeconv.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "MSVCP140.dll",
                    "VCRUNTIME140.dll",
                    "VCRUNTIME140_1.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_nrt_python.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-heap-l1-1-0.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "api-ms-win-crt-stdio-l1-1-0.dll",
                    "python310.dll",
                ]),
                "_box.cp310-win_amd64.pyd": set([
                    "KERNEL32.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "python310.dll",
                ]),
            },
        },
        "linux": {
            "x86_64": {
                "_devicearray.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_dynfunc.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_helperlib.cpython-310-x86_64-linux-gnu.so": set([
                    "ld-linux-x86-64.so.2",
                    "libc.so.6",
                    "libm.so.6",
                    "libpthread.so.0",
                ]),
                "mviewbuf.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_dispatcher.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_internal.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libm.so.6",
                    "libpthread.so.0",
                ]),
                "omppool.cpython-310-x86_64-linux-gnu.so": set([
                    "ld-linux-x86-64.so.2",
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libgomp.so.1.0.0",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "tbbpool.cpython-310-x86_64-linux-gnu.so": set([
                    "ld-linux-x86-64.so.2",
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                    "libtbb.so.12",
                ]),
                "_num_threads.cpython-310-x86_64-linux-gnu.so": set([
                    "ld-linux-x86-64.so.2",
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "workqueue.cpython-310-x86_64-linux-gnu.so": set([
                    "ld-linux-x86-64.so.2",
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_extras.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_typeconv.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_nrt_python.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_box.cpython-310-x86_64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
            },
            "aarch64": {
                "_dynfunc.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_dispatcher.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_devicearray.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_helperlib.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libm.so.6",
                    "libpthread.so.0",
                ]),
                "mviewbuf.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "omppool.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libgomp.so.1.0.0",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_num_threads.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_internal.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libm.so.6",
                    "libpthread.so.0",
                ]),
                "workqueue.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_extras.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
                "_typeconv.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_nrt_python.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libpthread.so.0",
                    "libstdc++.so.6",
                ]),
                "_box.cpython-310-aarch64-linux-gnu.so": set([
                    "libc.so.6",
                    "libpthread.so.0",
                ]),
            },
        },
        "darwin": {
            "arm64": {
                "_devicearray.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                ]),
                "_dispatcher.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                ]),
                "_dynfunc.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "mviewbuf.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "_helperlib.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "workqueue.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                ]),
                "_num_threads.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "_internal.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "omppool.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                    "libomp.dylib",
                ]),
                "_extras.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
                "_typeconv.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                ]),
                "_nrt_python.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                    "libc++.1.dylib",
                ]),
                "_box.cpython-310-darwin.so": set([
                    "libSystem.B.dylib",
                ]),
            },
        },
    }

    def check_linkage(self, info, package_type):
        machine = platform.machine().lower()
        os_name = platform.system().lower()

        if package_type == "wheel":
            expected = self.wheel_expected_imports.get(os_name, {}).get(machine, {})
        else:
            raise ValueError(f"Unexpected package type: {package_type}")

        if not expected:
            self.skipTest(f"No expected data for {os_name}/{machine}/{package_type}")

        # Process each extension module
        canonicalised_libs = info.get("canonicalised_linked_libraries", {})

        for ext_path, libs in canonicalised_libs.items():
            ext_name = os.path.basename(ext_path)

            # Make extension name version-agnostic by replacing cpython-XXX with cpython-310
            # e.g. _dispatcher.cpython-313-darwin.so -> _dispatcher.cpython-310-darwin.so
            normalized_name = re.sub(r'\.cpython-\d+', '.cpython-310', ext_name)
            normalized_name = re.sub(r'\.cp\d+', '.cp310', normalized_name)

            # Every module must have expected data - fail if missing
            if normalized_name not in expected:
                raise AssertionError(
                    f"Extension module '{ext_name}' (normalized: '{normalized_name}') "
                    f"not found in expected data for {os_name}/{machine}. "
                    f"Available modules: {sorted(expected.keys())}"
                )

            expected_libs = expected[normalized_name]

            # Normalize version-specific library names to be version-agnostic
            normalized_libs = []
            for lib in libs:
                # Normalize Python DLL version (e.g. python313.dll -> python310.dll)
                lib = re.sub(r'python\d+', 'python310', lib)
                # Normalize delvewheel-bundled MSVCP hashed name (e.g. msvcp140-abc123 -> msvcp140)
                lib = re.sub(r'msvcp140-[a-f0-9]+', 'msvcp140', lib)
                normalized_libs.append(lib)

            got = set(normalized_libs)

            print(f"Checking {ext_name}: Expected {sorted(expected_libs)}, Got {sorted(got)}", flush=True)

            if expected_libs != got:
                msg = (f"Unexpected linkage for {ext_name}:\n"
                       f"Expected: {sorted(expected_libs)}\n"
                       f"     Got: {sorted(got)}\n\n"
                       f"Difference: {set.symmetric_difference(expected_libs, got)}\n"
                       f"Only in Expected: {set.difference(expected_libs, got)}\n"
                       f"Only in Got: {set.difference(got, expected_libs)}\n")
                raise AssertionError(msg)

    @needs_lief
    def test_wheel_linkage(self):
        info = config.get_sysinfo()
        self.check_linkage(info, "wheel")
