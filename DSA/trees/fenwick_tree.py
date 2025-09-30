"""Fenwick Tree (Binary Indexed Tree)
=====================================
A Fenwick Tree maintains prefix sums (or any invertible cumulative frequency
structure) in O(log n) time for both point updates and prefix queries using a
clever implicit encoding of partial sums in an array ``_bit``.

Lowbit Function
---------------
The expression ``i & -i`` (two's complement trick) isolates the least significant
set bit of ``i``. This value represents the size of the range that index ``i``
is responsible for aggregating.

Visualization (1-indexed indices):
Index Binary  Lowbit  Covers (inclusive indices)
1     0001    1       [1]
2     0010    2       [1..2]
3     0011    1       [3]
4     0100    4       [1..4]
5     0101    1       [5]
6     0110    2       [5..6]
8     1000    8       [1..8]

Update(index, delta)
--------------------
We add ``delta`` at position ``index`` (0-based public API). Internally we move
forward: ``i += lowbit(i)``. Each visited node covers a range that includes the
updated position, so we update its aggregated sum.

Prefix Sum(index)
-----------------
We accumulate while moving backward: ``i -= lowbit(i)``. This effectively jumps
over disjoint ranges whose union is [1..index].

Derivation of Complexity
------------------------
Number of set bits transitions until zero is O(log n); both update and prefix
sum iterate once per set bit cleared or added, giving O(log n) time.

Space: O(n) with very small constant (single int array of length n+1).

Range Sum
---------
range_sum(l, r) = prefix_sum(r) - prefix_sum(l-1)

Comparison with Segment Tree
----------------------------
Pros:
    - Simpler & generally faster in practice for prefix-sum like tasks
    - Lower memory overhead
Cons:
    - Harder to adapt for operations lacking invertibility (cannot do min/max easily)
    - Basic form handles point updates + prefix queries; range updates / range queries
      require augmented variants or switch to segment tree with lazy propagation.

Typical Applications
--------------------
- Frequency tables / order statistics with offline coordinate compression
- Inversion count in arrays
- Cumulative frequency queries in competitive programming
- 2D BIT extensions for matrix prefix sums (higher complexity)

Limitations / Pitfalls
----------------------
- Off-by-one errors: internal structure is 1-indexed while API might be 0-indexed.
- Using on non-invertible operations (like min) is invalid; structure relies on subtracting prefix sums.
- Large indices require coordinate compression when values exceed memory limits.

Possible Extensions
-------------------
- Range update + point query: store difference array semantics.
- Range update + range query: maintain two BITs for prefix weighted sums.
- 2D BIT for grid sums (O(log^2 n)).
"""
from __future__ import annotations
from typing import List

class FenwickTree:
    def __init__(self,n:int)->None:
        self.n=n; self._bit=[0]*(n+1)
    @classmethod
    def build(cls,data:List[int])->'FenwickTree':
        ft=cls(len(data))
        for i,v in enumerate(data): ft.update(i,v)
        return ft
    def update(self,index:int,delta:int)->None:
        i=index+1
        while i<=self.n:
            self._bit[i]+=delta; i+= i & -i
    def prefix_sum(self,index:int)->int:
        res=0; i=index+1
        while i>0:
            res+=self._bit[i]; i-= i & -i
        return res
    def range_sum(self,left:int,right:int)->int:
        if right<left: return 0
        return self.prefix_sum(right)-(self.prefix_sum(left-1) if left>0 else 0)
