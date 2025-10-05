"""Dynamic Programming (DP) – Classic Problems and Patterns
===========================================================
This module implements essential DP problems with clean, well-documented
solutions showcasing memoization, tabulation, and optimization tricks.

Included:
- Fibonacci: memoization and tabulation variants
- 0/1 Knapsack: value maximization with item recovery
- LIS (Longest Increasing Subsequence): O(n log n) patience-sorting approach
- Matrix Chain Multiplication: optimal parenthesization via DP

Notes:
- For educational clarity, we favor readability; some implementations also
  include a space-optimized variant where it clearly adds value.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple


class DynamicProgramming:
    """A collection of classic DP problems and solutions."""

    # ----------------------------- Fibonacci -----------------------------
    def fib_memo(self, n: int, _memo: Optional[Dict[int, int]] = None) -> int:
        """Fibonacci with memoization (top-down).

        Time: O(n)
        Space: O(n) for recursion + memo
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if _memo is None:
            _memo = {0: 0, 1: 1}
        if n in _memo:
            return _memo[n]
        _memo[n] = self.fib_memo(n - 1, _memo) + self.fib_memo(n - 2, _memo)
        return _memo[n]

    def fib_tab(self, n: int) -> int:
        """Fibonacci with tabulation (bottom-up, O(1) memory).

        Time: O(n)
        Space: O(1)
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    # --------------------------- 0/1 Knapsack ----------------------------
    def knapsack_01(self, weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
        """0/1 Knapsack maximizing value under capacity.

        Returns (max_value, picked_indices)
        Time: O(n * W), Space: O(n * W)
        """
        n = len(weights)
        if n != len(values):
            raise ValueError("weights and values must have the same length")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")

        # dp[i][w] = best value using first i items with capacity w
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            wt, val = weights[i - 1], values[i - 1]
            for w in range(capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if wt <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - wt] + val)

        # Recover items
        res_value = dp[n][capacity]
        picked: List[int] = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                picked.append(i - 1)
                w -= weights[i - 1]
        picked.reverse()
        return res_value, picked

    # ------------------------------- LIS --------------------------------
    def lis_length(self, nums: List[int]) -> int:
        """Length of Longest Increasing Subsequence in O(n log n).

        Uses patience sorting: tails[k] = minimum possible tail of an
        increasing subsequence of length k+1.
        """
        import bisect

        tails: List[int] = []
        for x in nums:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)

    # -------------------- Matrix Chain Multiplication --------------------
    def matrix_chain_multiplication(self, dims: List[int]) -> Tuple[int, str]:
        """Matrix Chain Multiplication optimal parenthesization.

        dims describes matrices A1..An where Ai has dimensions dims[i-1] x dims[i].
        Returns: (min_scalar_mults, parenthesization_string)

        Time: O(n^3) | Space: O(n^2)
        """
        n = len(dims) - 1
        if n <= 0:
            return 0, ""

        # m[i][j] = minimal cost to multiply Ai..Aj
        # s[i][j] = split index producing optimal cost
        INF = 10**18
        m = [[0 if i == j else INF for j in range(n)] for i in range(n)]
        s = [[-1] * n for _ in range(n)]

        for L in range(2, n + 1):  # chain length
            for i in range(0, n - L + 1):
                j = i + L - 1
                for k in range(i, j):
                    cost = (
                        m[i][k]
                        + m[k + 1][j]
                        + dims[i] * dims[k + 1] * dims[j + 1]
                    )
                    if cost < m[i][j]:
                        m[i][j] = cost
                        s[i][j] = k

        def build(i: int, j: int) -> str:
            if i == j:
                return f"A{i+1}"
            k = s[i][j]
            return f"({build(i, k)}×{build(k + 1, j)})"

        return m[0][n - 1], build(0, n - 1)


if __name__ == "__main__":
    dp = DynamicProgramming()
    print("fib_memo(10)=", dp.fib_memo(10))
    print("fib_tab(10)=", dp.fib_tab(10))
    w, v, C = [2, 3, 4, 5], [3, 4, 5, 6], 5
    print("knapsack_01:", dp.knapsack_01(w, v, C))
    print("lis_length:", dp.lis_length([10,9,2,5,3,7,101,18]))
    cost, paren = dp.matrix_chain_multiplication([30,35,15,5,10,20,25])
    print("mcm cost=", cost, "paren=", paren)
