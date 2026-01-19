"""
Statistical analysis module for benchmark measurements.

Provides robust statistical primitives for timing analysis including:
- Bootstrap confidence intervals
- Outlier detection via Tukey's fences
- Coefficient of variation for noise detection
- Welch's t-test for significance testing (no scipy dependency)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random


@dataclass
class TimingStats:
    """Statistical summary of timing measurements."""
    median: float           # Median time in nanoseconds
    mean: float             # Mean time in nanoseconds
    std: float              # Standard deviation
    iqr: float              # Interquartile range (Q3 - Q1)
    ci_lower: float         # 95% CI lower bound
    ci_upper: float         # 95% CI upper bound
    cv: float               # Coefficient of variation (std/mean)
    outlier_count: int      # Number of outliers removed
    n_samples: int          # Number of samples after outlier removal
    raw_samples: int        # Original number of samples
    is_noisy: bool          # True if CV > 5%

    @property
    def ci_width(self) -> float:
        """Width of 95% confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def ci_percent(self) -> float:
        """CI width as percentage of median."""
        if self.median == 0:
            return float('inf')
        return (self.ci_width / self.median) * 100


def _percentile(sorted_data: List[float], p: float) -> float:
    """Calculate percentile from sorted data (0 <= p <= 100)."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def _tukey_fences(data: List[float], k: float = 1.5) -> Tuple[float, float]:
    """
    Calculate Tukey's fences for outlier detection.

    Args:
        data: List of measurements
        k: Fence multiplier (1.5 for outliers, 3.0 for extreme outliers)

    Returns:
        Tuple of (lower_fence, upper_fence)
    """
    sorted_data = sorted(data)
    q1 = _percentile(sorted_data, 25)
    q3 = _percentile(sorted_data, 75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def remove_outliers(data: List[float], k: float = 1.5) -> Tuple[List[float], int]:
    """
    Remove outliers using Tukey's fences.

    Args:
        data: List of measurements
        k: Fence multiplier

    Returns:
        Tuple of (filtered_data, outlier_count)
    """
    if len(data) < 4:
        return data, 0

    lower, upper = _tukey_fences(data, k)
    filtered = [x for x in data if lower <= x <= upper]
    return filtered, len(data) - len(filtered)


def _bootstrap_ci(data: List[float], n_bootstrap: int = 1000,
                  ci_level: float = 0.95, stat_func=None) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: List of measurements
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (default 0.95 for 95% CI)
        stat_func: Function to calculate statistic (default: median)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if stat_func is None:
        stat_func = lambda x: sorted(x)[len(x) // 2]

    if len(data) < 2:
        val = stat_func(data) if data else 0.0
        return val, val

    # Bootstrap resampling
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = [data[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(resample))

    bootstrap_stats.sort()

    # Calculate percentile bounds
    alpha = 1 - ci_level
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]


def compute_timing_stats(times_ns: List[float],
                         remove_outliers_k: float = 1.5,
                         n_bootstrap: int = 1000,
                         noise_threshold: float = 0.05) -> TimingStats:
    """
    Compute comprehensive timing statistics with outlier removal and CIs.

    Args:
        times_ns: List of timing measurements in nanoseconds
        remove_outliers_k: Tukey fence multiplier for outlier removal
        n_bootstrap: Number of bootstrap iterations for CI
        noise_threshold: CV threshold for noisy measurements (default 5%)

    Returns:
        TimingStats with all computed statistics
    """
    raw_samples = len(times_ns)

    if raw_samples == 0:
        return TimingStats(
            median=0, mean=0, std=0, iqr=0,
            ci_lower=0, ci_upper=0, cv=0,
            outlier_count=0, n_samples=0, raw_samples=0,
            is_noisy=True
        )

    # Remove outliers
    filtered, outlier_count = remove_outliers(times_ns, remove_outliers_k)

    if len(filtered) == 0:
        filtered = times_ns  # Fallback to original if all removed
        outlier_count = 0

    n = len(filtered)
    sorted_data = sorted(filtered)

    # Basic statistics
    median = _percentile(sorted_data, 50)
    mean = sum(filtered) / n

    # Standard deviation
    if n > 1:
        variance = sum((x - mean) ** 2 for x in filtered) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    # IQR
    q1 = _percentile(sorted_data, 25)
    q3 = _percentile(sorted_data, 75)
    iqr = q3 - q1

    # Coefficient of variation
    cv = std / mean if mean > 0 else 0.0

    # Bootstrap CI for median
    ci_lower, ci_upper = _bootstrap_ci(filtered, n_bootstrap)

    return TimingStats(
        median=median,
        mean=mean,
        std=std,
        iqr=iqr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cv=cv,
        outlier_count=outlier_count,
        n_samples=n,
        raw_samples=raw_samples,
        is_noisy=cv > noise_threshold
    )


def detect_noise(stats: TimingStats, cv_threshold: float = 0.05) -> List[str]:
    """
    Detect potential noise issues in measurements.

    Args:
        stats: TimingStats from compute_timing_stats
        cv_threshold: Coefficient of variation threshold

    Returns:
        List of warning messages
    """
    warnings = []

    if stats.cv > cv_threshold:
        warnings.append(
            f"High variance: CV={stats.cv:.1%} exceeds {cv_threshold:.0%} threshold"
        )

    if stats.ci_percent > 10:
        warnings.append(
            f"Wide confidence interval: {stats.ci_percent:.1f}% of median"
        )

    if stats.outlier_count > stats.raw_samples * 0.1:
        warnings.append(
            f"Many outliers: {stats.outlier_count}/{stats.raw_samples} "
            f"({stats.outlier_count/stats.raw_samples:.0%})"
        )

    return warnings


@dataclass
class ComparisonResult:
    """Result of comparing two sets of measurements."""
    faster: str                    # 'a', 'b', or 'equal'
    ratio: float                   # a_median / b_median
    significant: bool              # True if statistically significant
    p_value: float                 # Approximate p-value from Welch's t-test
    a_stats: TimingStats
    b_stats: TimingStats
    ci_overlaps: bool              # True if confidence intervals overlap

    @property
    def speedup(self) -> float:
        """Speedup factor (>1 means 'b' is faster than 'a')."""
        return self.ratio

    def summary(self) -> str:
        """Human-readable summary."""
        if self.faster == 'equal':
            return f"No significant difference (p={self.p_value:.3f})"

        winner = 'A' if self.faster == 'a' else 'B'
        factor = self.ratio if self.faster == 'b' else 1/self.ratio
        sig = "significant" if self.significant else "not significant"
        return f"{winner} is {factor:.2f}x faster ({sig}, p={self.p_value:.3f})"


def _welch_t_test(mean1: float, std1: float, n1: int,
                  mean2: float, std2: float, n2: int) -> Tuple[float, float]:
    """
    Perform Welch's t-test without scipy.

    Returns:
        Tuple of (t_statistic, approximate_p_value)
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    # Welch's t-statistic
    se1 = std1 ** 2 / n1
    se2 = std2 ** 2 / n2
    se_diff = math.sqrt(se1 + se2)

    if se_diff == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se_diff

    # Welch-Satterthwaite degrees of freedom
    if se1 + se2 == 0:
        df = n1 + n2 - 2
    else:
        df = (se1 + se2) ** 2 / (
            se1 ** 2 / (n1 - 1) + se2 ** 2 / (n2 - 1)
        )

    # Approximate p-value using normal approximation for large df
    # For small df, this is conservative
    abs_t = abs(t_stat)

    # Simple approximation based on normal distribution
    # p-value for two-tailed test
    if abs_t > 4:
        p_value = 0.0001  # Very significant
    elif abs_t > 3:
        p_value = 0.003
    elif abs_t > 2.576:
        p_value = 0.01
    elif abs_t > 1.96:
        p_value = 0.05
    elif abs_t > 1.645:
        p_value = 0.10
    else:
        # Linear interpolation for smaller t values
        p_value = min(1.0, 2 * (1 - 0.5 * (1 + math.erf(abs_t / math.sqrt(2)))))

    return t_stat, p_value


def compare_with_significance(times_a: List[float], times_b: List[float],
                              alpha: float = 0.05,
                              remove_outliers_k: float = 1.5) -> ComparisonResult:
    """
    Compare two sets of timing measurements with statistical significance.

    Args:
        times_a: First set of timing measurements (nanoseconds)
        times_b: Second set of timing measurements (nanoseconds)
        alpha: Significance level (default 0.05)
        remove_outliers_k: Tukey fence multiplier

    Returns:
        ComparisonResult with comparison details
    """
    stats_a = compute_timing_stats(times_a, remove_outliers_k)
    stats_b = compute_timing_stats(times_b, remove_outliers_k)

    # Calculate ratio (a/b, so >1 means b is faster)
    if stats_b.median > 0:
        ratio = stats_a.median / stats_b.median
    else:
        ratio = float('inf') if stats_a.median > 0 else 1.0

    # Welch's t-test
    t_stat, p_value = _welch_t_test(
        stats_a.mean, stats_a.std, stats_a.n_samples,
        stats_b.mean, stats_b.std, stats_b.n_samples
    )

    significant = p_value < alpha

    # Check CI overlap
    ci_overlaps = not (stats_a.ci_upper < stats_b.ci_lower or
                       stats_b.ci_upper < stats_a.ci_lower)

    # Determine winner
    if not significant or ci_overlaps:
        faster = 'equal'
    elif stats_a.median < stats_b.median:
        faster = 'a'
    else:
        faster = 'b'

    return ComparisonResult(
        faster=faster,
        ratio=ratio,
        significant=significant,
        p_value=p_value,
        a_stats=stats_a,
        b_stats=stats_b,
        ci_overlaps=ci_overlaps
    )


def format_time(ns: float) -> str:
    """Format nanoseconds in human-readable form."""
    if ns >= 1e9:
        return f"{ns/1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f}us"
    else:
        return f"{ns:.0f}ns"


def format_stats(stats: TimingStats) -> str:
    """Format TimingStats as a concise string."""
    time_str = format_time(stats.median)
    ci_str = f"[{format_time(stats.ci_lower)}, {format_time(stats.ci_upper)}]"
    cv_str = f"CV={stats.cv:.1%}"
    noise_flag = " [NOISY]" if stats.is_noisy else ""
    return f"{time_str} {ci_str} {cv_str}{noise_flag}"
