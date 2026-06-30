#!/usr/bin/env python3
"""Chart per-test compile times from the wheel-builder CI artifacts.

Produces, for one python/numpy config:
  1. <out>/top10_per_platform_*.png  -- top-10 slowest compiles per platform,
     each bar split into codegen (solid) vs rest-of-compile (translucent).
  2. <out>/cross_platform_*.png      -- the worst offenders compared across
     platforms (grouped bars).

Reuses the parser from analyze_compile_times.py (run from anywhere; Python puts
this script's dir on sys.path so the sibling import resolves).

Usage:
    python chart_compile_times.py [--data-dir DIR] [--py 3.12] [--np 2.0]
                                  [--top 10] [--outdir ci_debug]
"""
from __future__ import annotations
import argparse
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from analyze_compile_times import _LINE, _DIR, to_sec  # noqa: E402

PLAT_COLOR = {
    "linux-64": "#4C72B0",
    "linux-aarch64": "#55A868",
    "osx-arm64": "#C44E52",
    "win-64": "#8172B3",
    "win-arm64": "#CCB974",
}


def short(name: str) -> str:
    """Trim the common numba test prefixes for readable axis labels."""
    for p in ("numba.cuda.tests.", "numba.tests.", "numba."):
        if name.startswith(p):
            return name[len(p):]
    return name


def load_recs(data_dir, want_py, want_np):
    """plat -> {test_name: (compile_s, codegen_s)} for the chosen config."""
    per = defaultdict(dict)
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name, "compile_times.txt")
        m = _DIR.match(name)
        if not m or not os.path.isfile(path):
            continue
        plat, py, np = m["plat"], m["py"], m["np"]
        np_ok = (np is None) if want_np is None else (np == want_np)
        if py != want_py or not np_ok:
            continue
        for line in open(path):
            lm = _LINE.match(line.rstrip("\n"))
            if not lm:
                continue
            c, g = to_sec(lm["comp"]), to_sec(lm["codegen"])
            prev = per[plat].get(lm["name"])
            if prev is None or c > prev[0]:
                per[plat][lm["name"]] = (c, g)
    return per


def chart_per_platform(per, platforms, top, cfg, path):
    """Grid of horizontal bars: top-N slowest compiles on each platform."""
    n = len(platforms)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 + 3.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, plat in zip(axes, platforms):
        ranked = sorted(per[plat].items(), key=lambda kv: -kv[1][0])[:top]
        ranked.reverse()  # largest at top of the barh
        labels = [short(nm) for nm, _ in ranked]
        compile_s = [v[0] for _, v in ranked]
        codegen_s = [v[1] for _, v in ranked]
        rest = [c - g for c, g in zip(compile_s, codegen_s)]
        color = PLAT_COLOR.get(plat, "#777777")
        y = range(len(labels))
        ax.barh(y, codegen_s, color=color, label="codegen")
        ax.barh(y, rest, left=codegen_s, color=color, alpha=0.35,
                label="compile − codegen")
        for i, c in enumerate(compile_s):
            ax.text(c, i, f" {c:.0f}s", va="center", ha="left", fontsize=8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{plat}  ·  top {top} by compile", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("compile time (s)", fontsize=9)
        ax.margins(x=0.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Slowest-compiling tests per platform  ({cfg})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_cross_platform(per, platforms, top, cfg, path):
    """Grouped bars: the worst offenders compared across platforms."""
    names = set().union(*(d.keys() for d in per.values()))
    ranked = []
    for nm in names:
        vals = {p: per[p].get(nm, (0.0, 0.0))[0] for p in platforms}
        present = [v for v in vals.values() if v > 0]
        ranked.append((st.mean(present) if present else 0.0, nm, vals))
    ranked.sort(key=lambda r: -r[0])
    ranked = ranked[:top]
    ranked.reverse()

    labels = [short(nm) for _, nm, _ in ranked]
    y = range(len(labels))
    h = 0.8 / len(platforms)
    fig, ax = plt.subplots(figsize=(13, 1.5 + 0.55 * top))
    for j, plat in enumerate(platforms):
        offs = [i + (j - (len(platforms) - 1) / 2) * h for i in y]
        vals = [v[plat] for _, _, v in ranked]
        ax.barh(offs, vals, height=h, color=PLAT_COLOR.get(plat, "#777"),
                label=plat)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("compile time (s)", fontsize=9)
    ax.set_title(f"Worst compilers across platforms  ({cfg})",
                 fontsize=13, fontweight="bold")
    ax.margins(x=0.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/tmp/compile_times_pr110")
    ap.add_argument("--py", default="3.12")
    ap.add_argument("--np", default="2.0")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--outdir", default="ci_debug")
    args = ap.parse_args()

    want_np = None if args.np.lower() in ("none", "-", "") else args.np
    per = load_recs(args.data_dir, args.py, want_np)
    platforms = sorted(per)
    if not platforms:
        raise SystemExit(f"no data for py{args.py}/np{args.np}")

    cfg = f"py{args.py} / np{args.np}"
    tag = f"py{args.py}_np{args.np}"
    os.makedirs(args.outdir, exist_ok=True)
    p1 = os.path.join(args.outdir, f"top{args.top}_per_platform_{tag}.png")
    p2 = os.path.join(args.outdir, f"cross_platform_{tag}.png")
    chart_per_platform(per, platforms, args.top, cfg, p1)
    chart_cross_platform(per, platforms, args.top, cfg, p2)
    print(f"platforms: {', '.join(platforms)}")
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
