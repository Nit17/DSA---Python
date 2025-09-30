"""Segment Tree
================
This implementation provides point update and range sum query over an array in
O(log n) time with O(n) memory. Segment trees generalize to any associative
operation (min, max, gcd, xor, custom structs) as long as the operation is:
    - associative: op(a, op(b, c)) == op(op(a, b), c)
    - has an identity element (for easy neutral returns on non-overlapping ranges)

Concept
-------
We recursively partition the array into halves forming a full binary tree where
each internal node stores the aggregated value of its segment. The root covers
the entire range [0, n-1]. A query decomposes the requested interval into a
union of O(log n) disjoint node segments.

Key Routines
------------
Build: O(n)
    Recursively compute children then combine: tree[idx] = tree[idx*2] + tree[idx*2+1]
Query (range_sum): O(log n)
    Three relation cases for node segment [l,r] and query [ql, qr]:
        1. Disjoint: return identity (0 for sum)
        2. Fully contained: return stored value
        3. Partial overlap: recurse into both children and combine
Update (point): O(log n)
    Descend to leaf, update its value, then recompute ancestors on the path back.

Complexities
------------
Operation   Time    Space
---------   ------  -----
build       O(n)    O(n)
update      O(log n)
range query O(log n)

Why Build is O(n)
-----------------
Each node is computed exactly once; a tree with n leaves has < 2n nodes (for a
binary tree that's full/complete at the leaf level). Recurrence T(n) = 2T(n/2)+O(1).

Memory Layout
-------------
We use a 1-indexed implicit array (root at 1) sized 4*n to safely accommodate
any recursion pattern without worrying about exact tree shape; for n being a
power of two this is ~2n. More compact dynamic representations are possible.

Extensions & Variants
---------------------
1. Lazy Propagation (Range Updates): store pending updates at internal nodes;
   push them to children only when needed by overlapping queries.
2. Iterative Segment Tree: bottom-up build in arrays for better constant factors.
3. Dynamic Segment Tree: allocate nodes on demand for large / sparse indices.
4. Segment Tree Beats: advanced technique for complex range operations.

Comparisons
-----------
Fenwick Tree (Binary Indexed Tree): also O(log n) for point update + prefix sum
but simpler and lower constant factors; segment tree is more flexible for
arbitrary associative range queries and easier to extend for range updates.

Pitfalls
--------
- Off-by-one errors in mid calculation boundaries.
- Forgetting identity value in disjoint segment case corrupts answers.
- Using Python recursion: deep recursion for very large n can hit recursion limit.

Possible Improvements
---------------------
- Add generic operation parameter (function + identity) for reusability.
- Implement lazy propagation for range add / assign operations.
"""
from __future__ import annotations
from typing import List

class SegmentTree:
    def __init__(self,data: List[int]) -> None:
        self.n=len(data)
        self._tree=[0]*(4*self.n)
        if self.n: self._build(data,1,0,self.n-1)
    def _build(self,data:List[int],idx:int,l:int,r:int)->None:
        if l==r: self._tree[idx]=data[l]; return
        m=(l+r)//2
        self._build(data,idx*2,l,m); self._build(data,idx*2+1,m+1,r)
        self._tree[idx]=self._tree[idx*2]+self._tree[idx*2+1]
    def update(self,pos:int,value:int)->None:
        def _u(idx:int,l:int,r:int):
            if l==r: self._tree[idx]=value; return
            m=(l+r)//2
            if pos<=m: _u(idx*2,l,m)
            else: _u(idx*2+1,m+1,r)
            self._tree[idx]=self._tree[idx*2]+self._tree[idx*2+1]
        _u(1,0,self.n-1)
    def range_sum(self,ql:int,qr:int)->int:
        def _q(idx:int,l:int,r:int)->int:
            if qr<l or ql>r: return 0
            if ql<=l and r<=qr: return self._tree[idx]
            m=(l+r)//2
            return _q(idx*2,l,m)+_q(idx*2+1,m+1,r)
        return _q(1,0,self.n-1)
