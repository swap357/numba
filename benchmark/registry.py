"""
Benchmark registry - singleton module to avoid circular import issues.
"""

from typing import Dict, List, Any

# Global benchmark registry
BENCHMARKS: Dict[str, List[Any]] = {
    'ufunc': [],
    'reduction': [],
    'linalg': [],
    'compound': [],
}


def register_benchmark(category: str):
    """Decorator to register a benchmark class."""
    def decorator(cls):
        BENCHMARKS[category].append(cls())
        return cls
    return decorator
