"""
Performance pattern detection and annotation.

Identifies known performance anti-patterns and provides recommendations.
"""

import re
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum


class PatternSeverity(Enum):
    """Severity level of detected patterns."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformancePattern:
    """A detected performance pattern."""
    name: str
    severity: PatternSeverity
    description: str
    recommendation: str
    estimated_impact: str  # e.g., "2-5x slowdown"
    file_location: str = ""
    line_number: int = 0
    code_snippet: str = ""


# Pattern database with detection functions and recommendations
PATTERN_DATABASE = {
    'nditer_overhead': {
        'severity': PatternSeverity.CRITICAL,
        'description': "Using np.nditer() for iteration causes significant overhead",
        'recommendation': "Replace np.nditer(a) with a.flat for faster iteration",
        'impact': "3-6x slowdown",
        'patterns': [
            r'np\.nditer\s*\(',
            r'numpy\.nditer\s*\(',
            r'for\s+\w+\s+in\s+np\.nditer',
        ],
    },
    'item_method_overhead': {
        'severity': PatternSeverity.WARNING,
        'description': "Using .item() to extract scalars adds overhead",
        'recommendation': "Use direct indexing or iteration with .flat",
        'impact': "1.5-2x slowdown per call",
        'patterns': [
            r'\.item\s*\(\s*\)',
            r'view\.item\s*\(',
        ],
    },
    'no_simd_reduction': {
        'severity': PatternSeverity.WARNING,
        'description': "Scalar reduction loop not vectorized",
        'recommendation': "Ensure array is contiguous and loop is simple",
        'impact': "2-4x slowdown on large arrays",
        'patterns': [
            # Look for scalar accumulator patterns
            r'for\s+\w+\s+in\s+.*:\s*\n\s+\w+\s*[+\-*/]=',
        ],
    },
    'no_early_exit': {
        'severity': PatternSeverity.INFO,
        'description': "Loop without early exit opportunity",
        'recommendation': "Consider adding early return for all/any style operations",
        'impact': "Variable, depends on data",
        'patterns': [
            # Difficult to detect reliably, use heuristics
        ],
    },
    'unnecessary_copy': {
        'severity': PatternSeverity.WARNING,
        'description': "Creating unnecessary array copy",
        'recommendation': "Use np.asarray() instead of np.array() when copy not needed",
        'impact': "1.5-3x slowdown for large arrays",
        'patterns': [
            r'np\.array\s*\(\s*a\s*\)',
            r'np\.copy\s*\(',
        ],
    },
    'non_contiguous_access': {
        'severity': PatternSeverity.WARNING,
        'description': "Potentially non-contiguous memory access",
        'recommendation': "Ensure array is C-contiguous with np.ascontiguousarray()",
        'impact': "1.5-2x slowdown",
        'patterns': [
            r'\[\s*:\s*,\s*\d+\s*\]',  # Column slicing
            r'\.T\s*[\.\[]',  # Transpose then access
        ],
    },
    'branch_in_loop': {
        'severity': PatternSeverity.INFO,
        'description': "Conditional branch inside hot loop",
        'recommendation': "Move invariant conditions outside loop or use np.where",
        'impact': "Variable, depends on branch predictability",
        'patterns': [
            # Heuristic patterns
        ],
    },
    'python_object_in_loop': {
        'severity': PatternSeverity.CRITICAL,
        'description': "Python object operations in loop prevent optimization",
        'recommendation': "Use typed arrays and avoid Python objects in hot paths",
        'impact': "10-100x slowdown",
        'patterns': [
            r'isinstance\s*\(',
            r'type\s*\(\s*\w+\s*\)',
            r'getattr\s*\(',
        ],
    },
}


def _find_pattern_in_source(source: str, patterns: List[str]) -> List[tuple]:
    """Find all pattern matches in source code."""
    matches = []
    lines = source.split('\n')

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line):
                matches.append((i + 1, line.strip(), pattern))

    return matches


def detect_patterns_in_source(source: str, file_path: str = "") -> List[PerformancePattern]:
    """
    Detect performance patterns in source code.

    Args:
        source: Source code string
        file_path: Optional file path for reporting

    Returns:
        List of detected PerformancePattern objects
    """
    detected = []

    for pattern_name, pattern_info in PATTERN_DATABASE.items():
        patterns = pattern_info.get('patterns', [])
        if not patterns:
            continue

        matches = _find_pattern_in_source(source, patterns)
        for line_num, code, _ in matches:
            detected.append(PerformancePattern(
                name=pattern_name,
                severity=pattern_info['severity'],
                description=pattern_info['description'],
                recommendation=pattern_info['recommendation'],
                estimated_impact=pattern_info['impact'],
                file_location=file_path,
                line_number=line_num,
                code_snippet=code,
            ))

    return detected


def detect_patterns(func: Callable, file_path: str = "") -> List[PerformancePattern]:
    """
    Detect performance patterns in a function.

    Args:
        func: Function to analyze
        file_path: Optional file path override

    Returns:
        List of detected PerformancePattern objects
    """
    try:
        source = inspect.getsource(func)
        if not file_path:
            try:
                file_path = inspect.getfile(func)
            except (TypeError, OSError):
                file_path = "<unknown>"
    except (TypeError, OSError):
        return []

    return detect_patterns_in_source(source, file_path)


def analyze_numba_source(numba_module_path: str,
                         function_names: Optional[List[str]] = None) -> Dict[str, List[PerformancePattern]]:
    """
    Analyze Numba source module for performance patterns.

    Args:
        numba_module_path: Path to Numba source file
        function_names: Optional list of function names to analyze

    Returns:
        Dictionary mapping function names to detected patterns
    """
    results = {}

    try:
        with open(numba_module_path, 'r') as f:
            source = f.read()
    except IOError:
        return results

    # Split into functions (rough heuristic)
    lines = source.split('\n')
    current_func = None
    current_lines = []
    indent_level = 0

    for i, line in enumerate(lines):
        # Detect function definition
        func_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
        if func_match:
            # Save previous function
            if current_func and current_lines:
                func_source = '\n'.join(current_lines)
                patterns = detect_patterns_in_source(func_source, numba_module_path)
                # Adjust line numbers
                for p in patterns:
                    p.line_number += current_lines[0] if current_lines else 0
                if patterns and (function_names is None or current_func in function_names):
                    results[current_func] = patterns

            current_func = func_match.group(2)
            current_lines = [(i + 1, line)]
            indent_level = len(func_match.group(1))
        elif current_func and line.strip():
            # Check if still in function (by indentation)
            if line and not line[0].isspace():
                # End of function
                if current_lines:
                    func_source = '\n'.join(l for _, l in current_lines)
                    patterns = detect_patterns_in_source(func_source, numba_module_path)
                    start_line = current_lines[0][0] if current_lines else 0
                    for p in patterns:
                        p.line_number += start_line - 1
                    if patterns and (function_names is None or current_func in function_names):
                        results[current_func] = patterns
                current_func = None
                current_lines = []
            else:
                current_lines.append((i + 1, line))

    # Handle last function
    if current_func and current_lines:
        func_source = '\n'.join(l for _, l in current_lines)
        patterns = detect_patterns_in_source(func_source, numba_module_path)
        start_line = current_lines[0][0] if current_lines else 0
        for p in patterns:
            p.line_number += start_line - 1
        if patterns and (function_names is None or current_func in function_names):
            results[current_func] = patterns

    return results


def get_recommendations(patterns: List[PerformancePattern],
                        sort_by_severity: bool = True) -> List[Dict[str, Any]]:
    """
    Get prioritized recommendations from detected patterns.

    Args:
        patterns: List of detected patterns
        sort_by_severity: Whether to sort by severity

    Returns:
        List of recommendation dictionaries
    """
    severity_order = {
        PatternSeverity.CRITICAL: 0,
        PatternSeverity.WARNING: 1,
        PatternSeverity.INFO: 2,
    }

    if sort_by_severity:
        patterns = sorted(patterns, key=lambda p: severity_order[p.severity])

    recommendations = []
    for p in patterns:
        recommendations.append({
            'pattern': p.name,
            'severity': p.severity.value,
            'location': f"{p.file_location}:{p.line_number}" if p.file_location else f"line {p.line_number}",
            'issue': p.description,
            'fix': p.recommendation,
            'impact': p.estimated_impact,
            'code': p.code_snippet,
        })

    return recommendations


def format_patterns(patterns: List[PerformancePattern]) -> str:
    """Format detected patterns as human-readable string."""
    if not patterns:
        return "No performance patterns detected."

    lines = [
        "Detected Performance Patterns",
        "=" * 40,
        "",
    ]

    # Group by severity
    by_severity = {}
    for p in patterns:
        by_severity.setdefault(p.severity, []).append(p)

    for severity in [PatternSeverity.CRITICAL, PatternSeverity.WARNING, PatternSeverity.INFO]:
        if severity not in by_severity:
            continue

        lines.append(f"[{severity.value.upper()}]")
        for p in by_severity[severity]:
            loc = f"{p.file_location}:{p.line_number}" if p.file_location else f"line {p.line_number}"
            lines.extend([
                f"  {p.name}",
                f"    Location: {loc}",
                f"    Issue: {p.description}",
                f"    Impact: {p.estimated_impact}",
                f"    Fix: {p.recommendation}",
            ])
            if p.code_snippet:
                lines.append(f"    Code: {p.code_snippet[:60]}...")
            lines.append("")

    return "\n".join(lines)


# Specific analysis for known Numba implementation patterns
NUMBA_FUNCTION_PATTERNS = {
    'np_all': {
        'expected_patterns': ['nditer_overhead', 'item_method_overhead'],
        'optimal_implementation': """
def flat_all(a):
    for v in a.flat:
        if not v:
            return False
    return True
""",
    },
    'np_any': {
        'expected_patterns': ['nditer_overhead', 'item_method_overhead'],
        'optimal_implementation': """
def flat_any(a):
    for v in a.flat:
        if v:
            return True
    return False
""",
    },
    'np_min': {
        'expected_patterns': ['nditer_overhead', 'item_method_overhead'],
        'optimal_implementation': """
def impl_min(a):
    min_value = a.flat[0]
    for v in a.flat:
        if v < min_value:
            min_value = v
    return min_value
""",
    },
    'np_max': {
        'expected_patterns': ['nditer_overhead', 'item_method_overhead'],
        'optimal_implementation': """
def impl_max(a):
    max_value = a.flat[0]
    for v in a.flat:
        if v > max_value:
            max_value = v
    return max_value
""",
    },
}


def analyze_numba_function_impl(func_name: str, source: str) -> Dict[str, Any]:
    """
    Analyze a known Numba function implementation.

    Args:
        func_name: Name of the function (e.g., 'np_all')
        source: Source code of the function

    Returns:
        Analysis dictionary with patterns and optimal implementation
    """
    result = {
        'function': func_name,
        'patterns': detect_patterns_in_source(source),
        'has_known_issues': False,
        'optimal_implementation': None,
    }

    if func_name in NUMBA_FUNCTION_PATTERNS:
        info = NUMBA_FUNCTION_PATTERNS[func_name]
        expected = set(info['expected_patterns'])
        detected = {p.name for p in result['patterns']}

        if expected & detected:
            result['has_known_issues'] = True
            result['optimal_implementation'] = info['optimal_implementation']

    return result
