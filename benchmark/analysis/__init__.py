"""
Analysis tools for Numba performance investigation.

This package provides tools for:
- LLVM IR and assembly analysis
- Performance pattern detection
- Report generation
"""

from .llvm_compare import (
    LLVMAnalysis,
    analyze_numba_function,
    compare_llvm_ir,
    detect_vectorization,
)
from .annotate import (
    PerformancePattern,
    detect_patterns,
    get_recommendations,
)
from .report import (
    generate_report,
    generate_optimization_summary,
)

__all__ = [
    'LLVMAnalysis',
    'analyze_numba_function',
    'compare_llvm_ir',
    'detect_vectorization',
    'PerformancePattern',
    'detect_patterns',
    'get_recommendations',
    'generate_report',
    'generate_optimization_summary',
]
