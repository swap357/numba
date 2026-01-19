"""
Report generation for benchmark analysis.

Generates comprehensive markdown reports with optimization recommendations.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import from sibling modules - handle both direct and package imports
try:
    from .annotate import PerformancePattern, PatternSeverity, format_patterns
    from .llvm_compare import LLVMAnalysis, format_analysis
except ImportError:
    from annotate import PerformancePattern, PatternSeverity, format_patterns
    from llvm_compare import LLVMAnalysis, format_analysis

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from stats import TimingStats, format_stats
    from reproducibility import SystemState, state_to_dict
except ImportError:
    TimingStats = None
    SystemState = None
    format_stats = lambda x: str(x)
    state_to_dict = lambda x: {}


@dataclass
class OptimizationOpportunity:
    """A ranked optimization opportunity."""
    function_name: str
    category: str
    current_speedup: float  # NumPy/Numba ratio (<1 means Numba slower)
    estimated_improvement: float  # Expected factor improvement
    patterns: List[PerformancePattern]
    file_location: str
    line_numbers: str
    priority: int  # 1=highest


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    title: str
    generated_at: datetime
    system_state: Optional[Any]
    results: List[Any]  # BenchmarkResult objects
    opportunities: List[OptimizationOpportunity]
    patterns: Dict[str, List[PerformancePattern]]
    llvm_analyses: Dict[str, LLVMAnalysis]


def _calculate_priority(speedup: float, patterns: List[PerformancePattern]) -> int:
    """Calculate optimization priority (1=highest)."""
    # Base priority on speedup (lower speedup = higher priority)
    if speedup < 0.2:  # 5x slower
        base = 1
    elif speedup < 0.5:  # 2x slower
        base = 2
    elif speedup < 1.0:
        base = 3
    else:
        base = 4

    # Adjust for critical patterns
    has_critical = any(p.severity == PatternSeverity.CRITICAL for p in patterns)
    if has_critical and base > 1:
        base -= 1

    return base


def identify_optimization_opportunities(
    results: List[Any],
    patterns: Dict[str, List[PerformancePattern]],
    numba_source_path: str = ""
) -> List[OptimizationOpportunity]:
    """
    Identify and rank optimization opportunities from benchmark results.

    Args:
        results: List of BenchmarkResult objects
        patterns: Dictionary of function name -> patterns
        numba_source_path: Path to Numba source for location info

    Returns:
        List of OptimizationOpportunity sorted by priority
    """
    opportunities = []

    # Group results by function name
    by_function = {}
    for r in results:
        by_function.setdefault(r.name, []).append(r)

    for func_name, func_results in by_function.items():
        # Get worst speedup for this function
        worst_result = min(func_results, key=lambda r: r.speedup)

        # Skip if Numba is already faster
        if worst_result.speedup >= 1.0:
            continue

        func_patterns = patterns.get(func_name, [])

        # Estimate improvement based on patterns
        estimated_improvement = 1.0
        for p in func_patterns:
            if p.name == 'nditer_overhead':
                estimated_improvement *= 3.0
            elif p.name == 'item_method_overhead':
                estimated_improvement *= 1.5
            elif p.name == 'unnecessary_copy':
                estimated_improvement *= 2.0

        # Extract location info
        if func_patterns:
            lines = sorted(set(p.line_number for p in func_patterns if p.line_number))
            line_str = ", ".join(str(l) for l in lines[:3])
            if len(lines) > 3:
                line_str += f" (+{len(lines)-3} more)"
            file_loc = func_patterns[0].file_location
        else:
            line_str = ""
            file_loc = numba_source_path

        priority = _calculate_priority(worst_result.speedup, func_patterns)

        opportunities.append(OptimizationOpportunity(
            function_name=func_name,
            category=worst_result.category,
            current_speedup=worst_result.speedup,
            estimated_improvement=estimated_improvement,
            patterns=func_patterns,
            file_location=file_loc,
            line_numbers=line_str,
            priority=priority,
        ))

    # Sort by priority (ascending) then by speedup (ascending = slowest first)
    opportunities.sort(key=lambda o: (o.priority, o.current_speedup))

    return opportunities


def generate_optimization_summary(opportunities: List[OptimizationOpportunity]) -> str:
    """Generate markdown summary of optimization opportunities."""
    if not opportunities:
        return "No optimization opportunities identified. All benchmarks show Numba >= NumPy performance."

    lines = [
        "## Optimization Opportunities",
        "",
        "Ranked by impact and feasibility:",
        "",
        "| Priority | Function | Current | Est. Gain | Root Cause | Location |",
        "|----------|----------|---------|-----------|------------|----------|",
    ]

    for opp in opportunities:
        slowdown = 1.0 / opp.current_speedup if opp.current_speedup > 0 else float('inf')
        gain_str = f"{opp.estimated_improvement:.1f}x" if opp.estimated_improvement > 1 else "-"

        # Root cause summary
        if opp.patterns:
            causes = list(set(p.name for p in opp.patterns))
            cause_str = ", ".join(causes[:2])
        else:
            cause_str = "Unknown"

        # Location
        if opp.line_numbers:
            loc_str = f"L{opp.line_numbers}"
        else:
            loc_str = "-"

        lines.append(
            f"| P{opp.priority} | {opp.function_name} | "
            f"{slowdown:.1f}x slower | {gain_str} | {cause_str} | {loc_str} |"
        )

    return "\n".join(lines)


def generate_executive_summary(results: List[Any],
                               opportunities: List[OptimizationOpportunity]) -> str:
    """Generate executive summary section."""
    total = len(results)
    numba_faster = sum(1 for r in results if r.speedup > 1.0)
    numpy_faster = sum(1 for r in results if r.speedup < 1.0)
    equal = total - numba_faster - numpy_faster

    # Calculate aggregate stats
    speedups = [r.speedup for r in results]
    if speedups:
        median_speedup = sorted(speedups)[len(speedups) // 2]
        max_speedup = max(speedups)
        min_speedup = min(speedups)
    else:
        median_speedup = max_speedup = min_speedup = 1.0

    lines = [
        "## Executive Summary",
        "",
        f"**Total benchmarks:** {total}",
        f"**Numba faster:** {numba_faster} ({100*numba_faster/total:.0f}%)" if total > 0 else "",
        f"**NumPy faster:** {numpy_faster} ({100*numpy_faster/total:.0f}%)" if total > 0 else "",
        "",
        "### Performance Range",
        f"- Best Numba speedup: **{max_speedup:.2f}x**",
        f"- Worst Numba speedup: **{min_speedup:.2f}x** ({1/min_speedup:.2f}x slower)" if min_speedup < 1 else f"- Worst case: {min_speedup:.2f}x",
        f"- Median speedup: **{median_speedup:.2f}x**",
        "",
    ]

    if opportunities:
        critical = sum(1 for o in opportunities if o.priority == 1)
        lines.extend([
            "### Optimization Summary",
            f"- **{len(opportunities)}** functions need optimization",
            f"- **{critical}** are critical priority",
        ])

        # Top 3 opportunities
        lines.append("")
        lines.append("**Top opportunities:**")
        for opp in opportunities[:3]:
            slowdown = 1/opp.current_speedup if opp.current_speedup > 0 else float('inf')
            lines.append(f"1. `{opp.function_name}` - {slowdown:.1f}x slower, est. {opp.estimated_improvement:.1f}x gain")

    return "\n".join(lines)


def generate_per_function_analysis(results: List[Any],
                                   patterns: Dict[str, List[PerformancePattern]],
                                   llvm_analyses: Dict[str, LLVMAnalysis]) -> str:
    """Generate detailed per-function analysis section."""
    lines = [
        "## Per-Function Analysis",
        "",
    ]

    # Group by function
    by_function = {}
    for r in results:
        by_function.setdefault(r.name, []).append(r)

    # Sort by worst speedup
    sorted_funcs = sorted(by_function.items(),
                          key=lambda x: min(r.speedup for r in x[1]))

    for func_name, func_results in sorted_funcs:
        worst = min(func_results, key=lambda r: r.speedup)
        best = max(func_results, key=lambda r: r.speedup)

        # Header
        if worst.speedup < 1:
            status = "NEEDS OPTIMIZATION"
        elif worst.speedup > 2:
            status = "EXCELLENT"
        else:
            status = "OK"

        lines.extend([
            f"### {func_name} [{status}]",
            "",
            f"**Category:** {worst.category}",
            f"**Speedup range:** {worst.speedup:.2f}x - {best.speedup:.2f}x",
            "",
        ])

        # Patterns
        func_patterns = patterns.get(func_name, [])
        if func_patterns:
            lines.append("**Detected patterns:**")
            for p in func_patterns:
                lines.append(f"- {p.severity.value.upper()}: {p.description}")
            lines.append("")

        # LLVM analysis if available
        if func_name in llvm_analyses:
            analysis = llvm_analyses[func_name]
            lines.append("**LLVM Analysis:**")
            if analysis.is_vectorized:
                lines.append(f"- Vectorized: Yes ({analysis.vector_width} elements)")
            else:
                lines.append("- Vectorized: No")
            lines.append(f"- Instructions: ~{analysis.total_instructions}")
            if analysis.called_functions:
                lines.append(f"- External calls: {', '.join(analysis.called_functions[:3])}")
            lines.append("")

        # Size breakdown
        lines.append("**Results by size:**")
        lines.append("| Size | NumPy (us) | Numba (us) | Speedup |")
        lines.append("|------|------------|------------|---------|")
        for r in sorted(func_results, key=lambda r: r.size):
            np_us = r.numpy_time_ns / 1000
            nb_us = r.numba_time_ns / 1000
            lines.append(f"| {r.size:,} | {np_us:.1f} | {nb_us:.1f} | {r.speedup:.2f}x |")
        lines.append("")

    return "\n".join(lines)


def generate_report(results: List[Any],
                    patterns: Dict[str, List[PerformancePattern]] = None,
                    llvm_analyses: Dict[str, LLVMAnalysis] = None,
                    system_state: Any = None,
                    title: str = "NumPy vs Numba Benchmark Report") -> str:
    """
    Generate comprehensive markdown benchmark report.

    Args:
        results: List of BenchmarkResult objects
        patterns: Dictionary of function name -> detected patterns
        llvm_analyses: Dictionary of function name -> LLVM analysis
        system_state: SystemState object
        title: Report title

    Returns:
        Complete markdown report string
    """
    patterns = patterns or {}
    llvm_analyses = llvm_analyses or {}

    # Identify optimization opportunities
    opportunities = identify_optimization_opportunities(results, patterns)

    # Build report sections
    sections = [
        f"# {title}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # System info
    if system_state:
        sections.extend([
            "## System Information",
            "",
            f"- **CPU:** {system_state.cpu_model}",
            f"- **Governor:** {system_state.cpu_governor}",
            f"- **Python:** {system_state.python_version}",
            f"- **NumPy:** {system_state.numpy_version}",
            f"- **Numba:** {system_state.numba_version}",
            f"- **LLVM:** {system_state.llvm_version}",
            "",
        ])

        if system_state.warnings:
            sections.append("### Reproducibility Warnings")
            for w in system_state.warnings:
                sections.append(f"- {w}")
            sections.append("")

    # Executive summary
    sections.append(generate_executive_summary(results, opportunities))
    sections.append("")

    # Optimization opportunities
    sections.append(generate_optimization_summary(opportunities))
    sections.append("")

    # Results table
    sections.extend([
        "## Full Results",
        "",
        "| Function | Size | NumPy (us) | Numba (us) | Speedup | Sig |",
        "|----------|------|------------|------------|---------|-----|",
    ])

    for r in sorted(results, key=lambda r: (r.category, r.name, r.size)):
        np_us = r.numpy_time_ns / 1000
        nb_us = r.numba_time_ns / 1000
        speedup_str = f"{r.speedup:.2f}x"
        if r.speedup < 1:
            speedup_str = f"**{speedup_str}**"
        sig = "Yes" if getattr(r, 'significant', True) else "No"
        sections.append(f"| {r.name} | {r.size:,} | {np_us:.1f} | {nb_us:.1f} | {speedup_str} | {sig} |")

    sections.append("")

    # Per-function analysis
    sections.append(generate_per_function_analysis(results, patterns, llvm_analyses))

    # Appendix: Detected patterns
    all_patterns = []
    for func_patterns in patterns.values():
        all_patterns.extend(func_patterns)

    if all_patterns:
        sections.extend([
            "## Appendix: All Detected Patterns",
            "",
            format_patterns(all_patterns),
        ])

    return "\n".join(sections)


def save_report(report: str, output_path: str):
    """Save report to file."""
    with open(output_path, 'w') as f:
        f.write(report)
