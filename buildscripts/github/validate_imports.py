#!/usr/bin/env python3
"""
Validate combined imported libraries for platform-specific wheels.

Usage:
  python3 buildscripts/github/validate_imports.py \
    --platform {linux-64, linux-aarch64, osx-arm64, win-64} \
    --path /path/to/extracted_wheel

Behavior:
  - Recursively scans the given directory for platform binaries and aggregates their imported libraries.
  - Compares the aggregated set against in-script expectations and prints both observed and expected.
  - Raises AssertionError on any mismatch (missing or unexpected entries).

Platform notes:
  - win-64: python*.dll is excluded from comparison as it varies by Python version.
  - osx-arm64: uses `otool -L`; requires `otool` to be available on PATH.

Exit status:
  - 0 on success, non-zero on mismatch or error.
"""

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Set

import lief


EXPECTED_IMPORTS_BY_PLATFORM = {
    "linux-64": {
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libgcc_s.so.1",
        "libgomp.so.1.0.0",
        "libm.so.6",
        "libpthread.so.0",
        "libstdc++.so.6",
        "libtbb.so.12",
    },
    "linux-aarch64": {
        "libc.so.6",
        "libpthread.so.0",
        "libm.so.6",
        "libgcc_s.so.1",
        "libstdc++.so.6",
        "libgomp.so.1.0.0",
    },
    "osx-arm64": {
        "libSystem.B.dylib",
        "libc++.1.dylib",
        "libomp.dylib",
    },
    "win-64": {
        "KERNEL32.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "api-ms-win-crt-string-l1-1-0.dll",
        "tbb12.dll",
        "VCOMP140.DLL",
    },
}


def collect_imports_linux(root: pathlib.Path) -> Set[str]:
    observed: Set[str] = set()
    for path in root.rglob("**/*.so"):
        so = lief.ELF.parse(str(path))
        libs = set(getattr(so, "libraries", []) or [])
        observed.update(libs)
    return observed


def collect_imports_windows(root: pathlib.Path) -> Set[str]:
    observed: Set[str] = set()
    for path in root.rglob("**/*"):
        if path.suffix.lower() not in {".dll", ".pyd"}:
            continue
        dll = lief.PE.parse(str(path))
        imports = {x.name for x in (dll.imports or [])}
        observed.update(imports)
        if hasattr(dll, "delay_imports") and dll.delay_imports:
            delay_imports = {x.name for x in dll.delay_imports}
            observed.update(delay_imports)
    # Exclude python*.dll as it varies by Python version
    python_dll_pattern = re.compile(r"^python\d+\.dll$", re.IGNORECASE)
    observed -= {name for name in list(observed) if python_dll_pattern.match(name)}
    return observed


def collect_imports_macos(root: pathlib.Path) -> Set[str]:
    observed: Set[str] = set()
    for path in root.rglob("**/*"):
        if path.suffix.lower() not in {".so", ".dylib"}:
            continue
        result = subprocess.run(
            ["otool", "-L", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            continue
        lines = result.stdout.splitlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            dep = line.split()[0]
            base = dep.split("/")[-1]
            if base.startswith("@rpath/"):
                base = base.split("/")[-1]
            observed.add(base)
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate combined imported libraries for platform wheels"
    )
    parser.add_argument(
        "--platform", required=True, choices=EXPECTED_IMPORTS_BY_PLATFORM.keys()
    )
    parser.add_argument(
        "--path", default=".", help="Path to extracted wheel directory (defaults to .)"
    )
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    platform_key = args.platform
    expected = set(EXPECTED_IMPORTS_BY_PLATFORM[platform_key])

    if platform_key.startswith("linux-"):
        observed = collect_imports_linux(root)
    elif platform_key == "win-64":
        observed = collect_imports_windows(root)
    elif platform_key == "osx-arm64":
        observed = collect_imports_macos(root)
    else:
        raise ValueError(f"Unsupported platform: {platform_key}")

    print("observed imports:", sorted(observed))
    print("expected imports:", sorted(expected))

    missing = expected - observed
    unexpected = observed - expected

    if missing or unexpected:
        raise AssertionError(
            f"Import set mismatch for {platform_key} artifacts\n"
            f"Missing (expected-not-observed): {sorted(missing)}\n"
            f"Unexpected (observed-not-expected): {sorted(unexpected)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
