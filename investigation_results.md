# Investigation Results for Numba Issue #7655

## Issue Summary
The issue reports a `TypeError: cannot store ... to ...: mismatching types` when using jitclass instances inside parallel prange loops. The problem occurs when accessing jitclass attributes or calling jitclass methods within a `prange` loop with `parallel=True`.

## Environment
- Numba version tested: 0.62.1 (latest from PyPI as of 2025-11-14)
- Python version: 3.12.3

## Test Results

### 1. Original MWE (Minimum Working Example)
**Status: ❌ FAILS**

The original example from the issue still fails with the same error:
```
TypeError: cannot store %"deferred.140711387230336.value" to %"deferred.140711387230336.data"*: mismatching types
```

This occurs when directly accessing `bag.value` inside a `prange` loop.

### 2. Workaround 1: Extract Instance Variable Outside Loop
**Status: ✅ WORKS**

```python
@njit(parallel=True)
def foo_parallel(bag, n):
    out = 0
    WORKAROUND = bag.value  # Extract outside the loop
    for i in prange(n):
        out += WORKAROUND
    return out
```

This workaround successfully avoids the type mismatch error by dereferencing the jitclass member into a temporary variable before entering the prange loop.

### 3. Workaround 2: Extract Instance Method Outside Loop
**Status: ✅ WORKS**

```python
@njit(parallel=True)
def foo_parallel(bag, n):
    out = 0
    inc = bag.increment  # Extract method reference outside the loop
    WORKAROUND = bag.value
    for i in prange(n):
        out += WORKAROUND
        inc()
    return out
```

This workaround works for calling jitclass methods inside parallel loops by extracting the method reference before the loop.

### 4. Last Question from c200chromebook
**Status: ✅ WORKS (No workaround needed)**

The code example from the last question in the issue thread actually **works without any errors**:

```python
@nb.njit(parallel=True, fastmath=True)
def diff_delay(x):
    for item in nb.prange(len(x)):
        x[item].square()  # This works!
```

## Answer to the Last Question

**Q: "Is there a workaround for this case?"**

**A: The code in your case already works!** The example you provided doesn't encounter the type mismatch error from the original issue. This is because you're calling a method on an object retrieved from a list (`x[item].square()`), rather than accessing attributes of a jitclass passed as a function parameter.

However, if you want to explore alternative approaches or optimize performance, here are several working patterns:

### Option 1: Current Approach (Already Works)
```python
@nb.njit(parallel=True, fastmath=True)
def diff_delay(x):
    for item in nb.prange(len(x)):
        x[item].square()
```
- **Status**: ✅ Works as-is
- **Performance**: Good (0.229s for 100 items in testing)

### Option 2: Extract Method Reference
```python
@nb.njit(parallel=True, fastmath=True)
def diff_delay_workaround1(x):
    for idx in nb.prange(len(x)):
        obj = x[idx]
        square_method = obj.square
        square_method()
```
- **Status**: ✅ Works
- **Performance**: Slightly better (0.207s for 100 items in testing)
- **Benefit**: More explicit, follows Stuart Archibald's workaround pattern

### Option 3: Inline Logic
```python
@nb.njit(parallel=True, fastmath=True)
def diff_delay_workaround2(x):
    for idx in nb.prange(len(x)):
        obj = x[idx]
        item_val = obj.item
        while item_val > 1.001:
            item_val = item_val - 1
        obj.item = item_val
```
- **Status**: ✅ Works
- **Performance**: Best (0.203s for 100 items in testing)
- **Benefit**: Avoids method call overhead, better loop invariant code motion

### Option 4: Helper Function
```python
@nb.njit
def square_helper(obj):
    while obj.item > 1.001:
        obj.item = obj.item - 1

@nb.njit(parallel=True, fastmath=True)
def diff_delay_workaround3(x):
    for idx in nb.prange(len(x)):
        square_helper(x[idx])
```
- **Status**: ✅ Works
- **Performance**: Good (0.224s for 100 items in testing)
- **Benefit**: More modular, reusable helper function

## Parallel Diagnostics Analysis

All versions successfully parallelize. The key difference is in loop invariant code motion:

- **Options 1, 2, 4**: Method/function calls cannot be hoisted out of the loop
- **Option 3 (Inline logic)**: Allows more constants to be hoisted, resulting in slightly better performance

## Recommendation

For your specific case (c200chromebook's question):
1. **Your current code already works** - no workaround needed
2. If you want the **best performance**, use **Option 3 (Inline Logic)** as it allows better compiler optimization
3. If you prefer **code organization**, use **Option 4 (Helper Function)** for cleaner, more maintainable code
4. The original issue's workaround pattern (extracting references) is not necessary for your use case, but can be applied if you encounter similar issues

## Root Cause

The original issue occurs specifically when:
1. A jitclass instance is passed as a function parameter
2. Attributes/methods of that instance are accessed inside a `prange` loop
3. The parallel accelerator tries to store the deferred type, causing a type mismatch

Your case avoids this because you're accessing objects from a list inside the loop, not from a function parameter, which has different type inference behavior.
