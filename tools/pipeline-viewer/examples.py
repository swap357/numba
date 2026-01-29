"""
Example functions that demonstrate Numba IR pass transformations.
Load these in the Pipeline Viewer to see CFG changes in the "IR Passes" tab.

Use arrow keys or click timeline dots to step through passes and watch
blocks get added, removed, or modified.
"""

from numba import njit, literal_unroll
import numpy as np


# =============================================================================
# 1. DEAD BRANCH PRUNING
# =============================================================================
# The branch on `debug` (a constant False) gets eliminated, removing the
# print block entirely. Watch block count decrease after DeadBranchPrune.

@njit
def dead_branch_example(x):
    debug = False  # Constant - branch will be pruned

    if debug:
        # This entire block gets removed
        y = x * 100
        z = y + 1
        return z
    else:
        return x * 2


# =============================================================================
# 2. DEAD CODE ELIMINATION
# =============================================================================
# Variables `unused1` and `unused2` are computed but never used.
# DeadCodeElimination removes them.

@njit
def dead_code_example(x):
    unused1 = x * 10      # Dead - result never used
    unused2 = unused1 + 5  # Dead - result never used
    result = x + 1         # Live - this is returned
    return result


# =============================================================================
# 3. LITERAL PROPAGATION / CONSTANT FOLDING
# =============================================================================
# Constants propagate through the computation. Watch `size` become a literal.

@njit
def literal_prop_example():
    size = 10          # Literal
    half = size // 2   # Becomes literal 5
    total = 0
    for i in range(half):  # range(5) - literal unrolling candidate
        total += i
    return total


# =============================================================================
# 4. BRANCH PRUNING WITH TYPE-BASED DISPATCH
# =============================================================================
# isinstance checks on typed values become constant after type inference,
# allowing branch pruning.

@njit
def type_dispatch_example(x):
    # After type inference, only one branch survives
    if isinstance(x, int):
        return x * 2
    elif isinstance(x, float):
        return x * 0.5
    else:
        return x


# =============================================================================
# 5. LOOP CANONICALIZATION
# =============================================================================
# While loops get transformed. Watch the CFG structure change.

@njit
def loop_canon_example(n):
    total = 0
    i = 0
    while i < n:
        total += i
        i += 1
    return total


# =============================================================================
# 6. LOOP WITH EARLY EXIT (Multiple exit points)
# =============================================================================
# Loop with break creates interesting CFG with multiple exit edges.

@njit
def early_exit_example(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            return i  # Early exit - creates extra block
    return -1


# =============================================================================
# 7. NESTED BRANCHES
# =============================================================================
# Multiple levels of branching creates complex CFG that gets simplified.

@njit
def nested_branch_example(x, y):
    result = 0

    if x > 0:
        if y > 0:
            result = 1  # quadrant I
        else:
            result = 4  # quadrant IV
    else:
        if y > 0:
            result = 2  # quadrant II
        else:
            result = 3  # quadrant III

    return result


# =============================================================================
# 8. PHI NODE INTRODUCTION (SSA)
# =============================================================================
# Variable `result` assigned in multiple branches needs phi node in SSA form.

@njit
def phi_node_example(cond, a, b):
    if cond:
        result = a + 1
    else:
        result = b * 2
    # After SSA: result = phi(result.1, result.2)
    return result


# =============================================================================
# 9. LOOP INVARIANT CODE MOTION (if enabled)
# =============================================================================
# `scale` computation could be hoisted out of the loop.

@njit
def loop_invariant_example(arr, factor):
    scale = factor * 2.0  # Loop invariant
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] * scale  # scale doesn't change
    return total


# =============================================================================
# 10. LITERAL UNROLL (explicit)
# =============================================================================
# Tuple iteration with literal_unroll creates separate blocks per element.

@njit
def literal_unroll_example(x):
    result = 0
    for i in literal_unroll((1, 2, 3, 4)):
        result += x * i
    return result


# =============================================================================
# 11. EXCEPTION HANDLING SIMPLIFICATION
# =============================================================================
# Try/except blocks create complex CFG that may get simplified.

@njit
def exception_example(arr, idx):
    try:
        return arr[idx]
    except IndexError:
        return -1


# =============================================================================
# 12. REDUCTION PATTERN
# =============================================================================
# Common pattern that shows loop structure clearly.

@njit
def reduction_example(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


# =============================================================================
# TRIGGER COMPILATION
# =============================================================================
if __name__ == "__main__":
    # Compile all functions
    dead_branch_example(5)
    dead_code_example(10)
    literal_prop_example()
    type_dispatch_example(42)      # int version
    type_dispatch_example(3.14)    # float version
    loop_canon_example(10)
    early_exit_example(np.array([1, 2, -3, 4]))
    nested_branch_example(1, 1)
    phi_node_example(True, 10, 20)
    loop_invariant_example(np.array([1.0, 2.0, 3.0]), 2.0)
    literal_unroll_example(5)
    # exception_example(np.array([1, 2, 3]), 0)  # May not work in all modes
    reduction_example(np.array([1.0, 2.0, 3.0]))

    print("All examples compiled successfully!")
    print("Copy any function to the Pipeline Viewer to explore its passes.")
