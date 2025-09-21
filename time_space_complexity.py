"""
Time and Space Complexity in Python (DSA Primer)

Overview
- Time complexity estimates how running time grows with input size n.
- Space complexity estimates how extra memory (auxiliary space) grows with n.
- We use asymptotic notation: O(upper bound), Θ(tight bound), Ω(lower bound).
- Analyze worst, average, and best case when relevant; default to worst-case unless stated.

Core Rules of Thumb
- Drop constants: O(2n + 10) → O(n); O(3n^2) → O(n^2).
- Dominant term rules: O(n^2 + n) → O(n^2).
- Sequential steps add: O(f(n)) + O(g(n)) → O(f(n) + g(n)).
- Nested loops multiply: O(f(n) * g(n)).
- Divide-and-conquer recurrences often yield O(n log n).

Time vs. Auxiliary Space
- Space complexity counts extra space your algorithm allocates (not the input itself),
  including recursion stack frames, temporary arrays, maps, etc.
- In-place algorithms attempt O(1) auxiliary space.

Below are canonical examples with short, runnable demos.
"""
from __future__ import annotations
from typing import List, Optional, Tuple


# O(1) time, O(1) space example ------------------------------------------------

def get_last_item(a: List[int]) -> Optional[int]:
    """Return the last element of a list.
    Time: O(1) — direct index access
    Space: O(1)
    """
    if not a:
        return None
    return a[-1]


# O(n) time, O(1) space example -------------------------------------------------

def sum_array(a: List[int]) -> int:
    """Sum elements by a single pass.
    Time: O(n)
    Space: O(1) auxiliary
    """
    s = 0
    for x in a:
        s += x
    return s


# O(log n) time, O(1) space example --------------------------------------------

def binary_search(a: List[int], target: int) -> int:
    """Binary search on a sorted list.
    Returns index of target or -1 if not found.
    Time: O(log n)
    Space: O(1) auxiliary (iterative)
    """
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# O(n log n) time, O(n) space example ------------------------------------------

def merge_sort(a: List[int]) -> List[int]:
    """Classic merge sort.
    Time: O(n log n)
    Space: O(n) due to merging arrays (can be optimized but generally linear)
    """
    if len(a) <= 1:
        return a[:]

    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])

    # Merge
    i = j = 0
    out: List[int] = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


# O(n^2) time, O(1) space example ----------------------------------------------

def has_duplicate_quadratic(a: List[int]) -> bool:
    """Check duplicates using double loop.
    Time: O(n^2) — compares each pair
    Space: O(1)
    """
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] == a[j]:
                return True
    return False


# O(n) time, O(n) space example (hash set) -------------------------------------

def has_duplicate_linear(a: List[int]) -> bool:
    """Check duplicates using a set.
    Time: O(n)
    Space: O(n) — stores seen elements
    Trade-off: faster time, higher memory
    """
    seen = set()
    for x in a:
        if x in seen:
            return True
        seen.add(x)
    return False


# Exponential time example O(2^n) -----------------------------------------------

def fib_exponential(n: int) -> int:
    """Naive recursive Fibonacci.
    Time: O(2^n) — each call branches into two (approx)
    Space: O(n) recursion depth
    """
    if n <= 1:
        return n
    return fib_exponential(n - 1) + fib_exponential(n - 2)


# Linear time, O(1) space Fibonacci (iterative) --------------------------------

def fib_linear(n: int) -> int:
    """Iterative Fibonacci.
    Time: O(n)
    Space: O(1)
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# Logarithmic time with fast-doubling (optional) --------------------------------

def fib_fast_doubling(n: int) -> int:
    """Fast-doubling Fibonacci.
    Time: O(log n)
    Space: O(log n) recursion depth
    """
    def _fd(k: int) -> Tuple[int, int]:
        if k == 0:
            return (0, 1)
        a, b = _fd(k // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

    return _fd(n)[0]


# Space complexity demonstration ------------------------------------------------

def reverse_in_place(a: List[int]) -> None:
    """Reverse a list in-place using two pointers.
    Time: O(n)
    Space: O(1) auxiliary — in-place, mutates input
    """
    i, j = 0, len(a) - 1
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1; j -= 1


def reversed_copy(a: List[int]) -> List[int]:
    """Return a new reversed list.
    Time: O(n)
    Space: O(n) — extra array
    """
    return list(reversed(a))


def recursion_depth_example(n: int) -> int:
    """Compute sum 1..n recursively to illustrate recursion stack space.
    Time: O(n)
    Space: O(n) recursion depth
    """
    if n == 0:
        return 0
    return n + recursion_depth_example(n - 1)


def _demo() -> None:
    arr = [3, 1, 4, 1, 5, 9]
    print("O(1): last item of", arr, "=", get_last_item(arr))

    print("O(n): sum", sum_array(arr))

    sorted_arr = sorted(arr)
    print("O(log n): binary search for 4 in", sorted_arr, "= index", binary_search(sorted_arr, 4))

    print("O(n log n): merge sort", merge_sort(arr))

    print("O(n^2) vs O(n): duplicates?", has_duplicate_quadratic(arr), has_duplicate_linear(arr))

    print("Fib exponential (n=10)", fib_exponential(10), "| linear", fib_linear(10), "| fast-doubling", fib_fast_doubling(10))

    a2 = arr[:]
    reverse_in_place(a2)
    print("Reverse in-place:", a2)
    print("Reversed copy:", reversed_copy(arr))

    print("Recursion depth example (n=5):", recursion_depth_example(5))


if __name__ == "__main__":
    _demo()
