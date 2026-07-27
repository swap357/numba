#!/usr/bin/env python3
"""Load per-test compile-time artifacts (from the wheel-builder CI) and rank
tests by compile time across platforms, for a single python/numpy config.

Artifacts are directories named ``compile-times-<platform>-py<X>[-np<Y>]``,
each holding a ``compile_times.txt`` of lines:

    Name: <test id> | Duration: <t> | Compile: <t> (<n> calls; \
passes <t>, codegen <t>) | Run: <t>

Usage:
    python analyze_compile_times.py [--data-dir DIR] [--py 3.12] [--np 2.0]
                                    [--top 25] [--csv pivot.csv]
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import statistics as st
from collections import defaultdict

# Name: <id> | Duration: t | Compile: t (n calls; passes t, codegen t) | Run: t
_LINE = re.compile(
    r"^Name: (?P<name>.+?) \| Duration: (?P<dur>[\d.]+(?:us|ms|s)) \| "
    r"Compile: (?P<comp>[\d.]+(?:us|ms|s)) \((?P<cnt>\d+) calls; "
    r"passes (?P<passes>[\d.]+(?:us|ms|s)), "
    r"codegen (?P<codegen>[\d.]+(?:us|ms|s))\) \| "
    r"Run: (?P<run>[\d.]+(?:us|ms|s))$")

_DIR = re.compile(
    r"^compile-times-(?P<plat>.+?)-py(?P<py>[\w.]+?)(?:-np(?P<np>[\d.]+))?$")


def to_sec(tok: str) -> float:
    if tok.endswith("us"):
        return float(tok[:-2]) / 1e6
    if tok.endswith("ms"):
        return float(tok[:-2]) / 1e3
    return float(tok[:-1])  # plain seconds


def fmt(sec: float) -> str:
    if sec >= 1.0:
        return f"{sec:.3f}s"
    return f"{sec * 1e3:.2f}ms"


def load(data_dir: str, want_py: str, want_np: str | None):
    """Return {platform: {test_name: compile_seconds}} for the chosen config,
    plus a per-platform totals dict."""
    per_platform: dict[str, dict[str, float]] = defaultdict(dict)
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"compile": 0.0, "run": 0.0, "passes": 0.0,
                 "codegen": 0.0, "n": 0, "n_compiled": 0})
    skipped_dirs = []

    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name, "compile_times.txt")
        m = _DIR.match(name)
        if not m or not os.path.isfile(path):
            continue
        plat, py, np = m["plat"], m["py"], m["np"]
        # want_np is None means "only artifacts with no numpy tag"
        # (e.g. win-arm64); otherwise require an exact numpy match.
        np_ok = (np is None) if want_np is None else (np == want_np)
        if py != want_py or not np_ok:
            skipped_dirs.append(name)
            continue

        for line in open(path):
            lm = _LINE.match(line.rstrip("\n"))
            if not lm:
                continue
            comp = to_sec(lm["comp"])
            t = totals[plat]
            t["compile"] += comp
            t["run"] += to_sec(lm["run"])
            t["passes"] += to_sec(lm["passes"])
            t["codegen"] += to_sec(lm["codegen"])
            t["n"] += 1
            if int(lm["cnt"]) > 0:
                t["n_compiled"] += 1
            # keep the max if a test id repeats (retry / parametrization)
            prev = per_platform[plat].get(lm["name"], 0.0)
            if comp > prev:
                per_platform[plat][lm["name"]] = comp
    return per_platform, totals, skipped_dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/tmp/compile_times_pr110")
    ap.add_argument("--py", default="3.12")
    ap.add_argument("--np", default="2.0",
                    help="numpy version, or 'none' for untagged "
                         "single-config builds like win-arm64")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--csv", help="write the cross-platform pivot to this CSV")
    args = ap.parse_args()

    want_np = None if args.np.lower() in ("none", "-", "") else args.np
    per_platform, totals, _ = load(args.data_dir, args.py, want_np)
    platforms = sorted(per_platform)
    if not platforms:
        raise SystemExit(
            f"no data for py{args.py}/np{args.np} in {args.data_dir}")

    print(f"config: py{args.py} / np{args.np}   "
          f"platforms: {', '.join(platforms)}")
    print("=" * 78)

    # --- per-platform aggregate -------------------------------------------
    print("per-platform totals (this config)")
    hdr = (f"{'platform':14} {'tests':>6} {'compiled':>8} "
           f"{'Σcompile':>10} {'Σpasses':>10} {'Σcodegen':>10} {'Σrun':>10}")
    print(hdr)
    print("-" * len(hdr))
    for p in platforms:
        t = totals[p]
        print(f"{p:14} {t['n']:>6} {t['n_compiled']:>8} "
              f"{fmt(t['compile']):>10} {fmt(t['passes']):>10} "
              f"{fmt(t['codegen']):>10} {fmt(t['run']):>10}")

    # --- per-platform top-N slowest compiles ------------------------------
    for p in platforms:
        ranked = sorted(per_platform[p].items(), key=lambda kv: -kv[1])
        print(f"\n{'=' * 78}\ntop {args.top} slowest compiles on {p}")
        print(f"{'#':>3} {'compile':>10}  test")
        for i, (name, comp) in enumerate(ranked[:args.top], 1):
            print(f"{i:>3} {fmt(comp):>10}  {name}")

    # --- cross-platform pivot: tests ranked by compile time --------------
    # Rank by the mean over platforms where the test actually compiled (>0).
    # "—" marks platforms where it didn't run / compiled nothing -- note that
    # Windows CI runs with _NUMBA_REDUCED_TESTING, so heavy cases are skipped
    # there; counting those as 0 would distort the ranking.
    all_names = set().union(*(d.keys() for d in per_platform.values()))
    rows = []
    for name in all_names:
        vals = [per_platform[p].get(name, 0.0) for p in platforms]
        present = [v for v in vals if v > 0]
        key = st.mean(present) if present else 0.0
        rows.append((key, name, vals))
    rows.sort(key=lambda r: -r[0])

    colw = {p: max(9, len(p)) for p in platforms}
    print(f"\n{'=' * 78}")
    print(f"top {args.top} by mean compile over platforms where it ran "
          f"(seconds; '—' = did not compile / skipped)")
    head = (f"{'mean':>9} "
            + " ".join(f"{p:>{colw[p]}}" for p in platforms) + "  test")
    print(head)
    print("-" * len(head))
    for key, name, vals in rows[:args.top]:
        cells = " ".join(
            (f"{v:>{colw[p]}.2f}" if v > 0 else f"{'—':>{colw[p]}}")
            for p, v in zip(platforms, vals))
        print(f"{key:>9.2f} {cells}  {name}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["test", "mean_compile_s"]
                       + [f"compile_s[{p}]" for p in platforms])
            for mean, name, vals in rows:
                w.writerow([name, f"{mean:.6f}"]
                           + [f"{v:.6f}" for v in vals])
        print(f"\nwrote pivot -> {args.csv} ({len(rows)} tests)")


if __name__ == "__main__":
    main()
