"""Heaps (Priority Queues)
=================================
This module provides array‑based implementations of:

1. MinHeap – supports retrieving/removing the smallest key in O(log n).
2. MaxHeap – implemented as a light wrapper over ``MinHeap`` by storing negated values.

Motivation / When to Use
------------------------
Heaps maintain a *partial* order: the root is the minimum (or maximum) element,
while no ordering is guaranteed among siblings or across subtrees beyond the
heap property. This makes heaps ideal for scenarios where you repeatedly need
access to the current extreme (min or max) while performing interleaved insert
and delete operations, e.g.:

- Dijkstra / Prim algorithms (priority queue of frontier edges / vertices)
- A* pathfinding (open set ordered by f-score)
- Event simulation (process next event by timestamp)
- Merge k sorted lists / external sorting (pick next smallest head)
- Stream problems (maintaining running top-k elements)

Array Representation & Indexing
--------------------------------
We store the tree implicitly in a 0‑indexed Python list ``_data``.

For an index ``i``:
        parent(i) = (i - 1) // 2
        left(i)   = 2*i + 1
        right(i)  = 2*i + 2

This leverages the *complete binary tree* property: levels are filled left to
right with no gaps, ensuring array compactness (no wasted slots).

Heap Property
-------------
Min-heap: ``_data[i] <= _data[left(i)]`` and ``_data[i] <= _data[right(i)]`` whenever children exist.
Max-heap mirrors this with ``>=``.

Core Operations (MinHeap)
-------------------------
push(x):
        - Append x to end (preserves completeness)
        - Sift Up: while x < parent, swap upward.
        - Cost: O(log n) swaps along a root‑to‑leaf path.

peek():
        - Return ``_data[0]`` (min). O(1).

pop():
        - Swap root with last element, pop last (removes min while preserving completeness)
        - Sift Down: compare with smaller child, swap until heap property restored.
        - Cost: O(log n).

Heapify (Building from Iterable)
--------------------------------
Strategy: copy all elements, then perform ``sift_down`` from the last internal
node down to index 0. This yields O(n) build time (NOT n log n). Sketch:

        Number of nodes at height h ≈ n / 2^{h+1}; each can move at most h levels.
        Cost ≈ Σ (h * n / 2^{h}) over h=0..log n = O(n).

Complexities Summary
--------------------
Operation   Average / Worst
----------  ---------------
push        O(log n)
pop         O(log n)
peek        O(1)
heapify     O(n)
search(*)   O(n)   (no ordering across siblings, can't binary search)

(*) Finding an arbitrary element by value requires a scan unless an auxiliary
index (hash map value -> indices) is maintained.

Space Complexity: O(n) contiguous array. Very cache friendly vs pointer-based trees.

Design Choices & Pythonic Notes
--------------------------------
- We keep implementation minimal: no decrease-key or update-key operations; for
    those you typically pair the heap with an index map or use a specialized
    priority queue (e.g. ``heapq`` plus entry invalidation pattern).
- ``MaxHeap`` reuses ``MinHeap`` by negating values: push(v) -> push(-v), etc.
    This is a common trick; it avoids code duplication, though it assumes values
    are orderable and negation is defined. For non-numeric types you'd implement
    a separate comparison or wrap items with a key.

Edge Cases & Pitfalls
---------------------
- Popping/peeking from an empty heap raises IndexError (we do explicit checks).
- Mixing 1‑based and 0‑based index formulas is a classic off-by-one bug.
- Forgetting to re-sift after modifying the root leads to a *silent* logical corruption.
- Negation trick for ``MaxHeap`` fails for types where ``-value`` is invalid (e.g., strings).

Alternatives & Comparisons
--------------------------
- Balanced BST (like AVL/Red-Black): Provides ordered iteration & logarithmic
    predecessor/successor queries; higher constant factors than heap for pure
    priority queue usage.
- Unsorted array/list: O(1) insertion but O(n) extract-min.
- Sorted array/list: O(n) insertion, O(1) extract-min.
- Fibonacci / pairing heaps: Amortized improvements for meld/decrease-key at
    cost of complexity; seldom worth in Python for typical problem sizes.

Possible Extensions
-------------------
- Add ``replace`` operation (pop then push in one O(log n)).
- Support key updates via index map (value -> position) + sift direction decision.
- Implement a d-ary heap (reduce height -> fewer comparisons but more children checks).

Implementation Below
--------------------
Minimal, focused, and intentionally explicit (instead of using Python's
``heapq``) for educational clarity.
"""
from __future__ import annotations
from typing import List, Iterable, Any, Optional

class MinHeap:
    def __init__(self,data:Optional[Iterable[Any]]=None)->None:
        self._data: List[Any]=[]
        if data:
            for x in data: self._data.append(x)
            self._heapify()
    def __len__(self)->int: return len(self._data)
    def _parent(self,i:int)->int: return (i-1)//2
    def _left(self,i:int)->int: return 2*i+1
    def _right(self,i:int)->int: return 2*i+2
    def _swap(self,i,j): self._data[i],self._data[j]=self._data[j],self._data[i]
    def _sift_up(self,i:int)->None:
        while i>0 and self._data[i]<self._data[self._parent(i)]:
            self._swap(i,self._parent(i)); i=self._parent(i)
    def _sift_down(self,i:int)->None:
        n=len(self._data)
        while True:
            l=self._left(i); r=self._right(i); s=i
            if l<n and self._data[l]<self._data[s]: s=l
            if r<n and self._data[r]<self._data[s]: s=r
            if s==i: break
            self._swap(i,s); i=s
    def _heapify(self):
        for i in range(len(self._data)//2-1,-1,-1): self._sift_down(i)
    def push(self,v:Any)->None:
        self._data.append(v); self._sift_up(len(self._data)-1)
    def peek(self)->Any:
        if not self._data: raise IndexError("peek from empty heap")
        return self._data[0]
    def pop(self)->Any:
        if not self._data: raise IndexError("pop from empty heap")
        self._swap(0,len(self._data)-1); val=self._data.pop()
        if self._data: self._sift_down(0)
        return val

class MaxHeap:
    def __init__(self,data:Optional[Iterable[Any]]=None)->None:
        self._heap=MinHeap([-x for x in data]) if data else MinHeap()
    def __len__(self)->int: return len(self._heap)
    def push(self,v:Any)->None: self._heap.push(-v)
    def peek(self)->Any: return -self._heap.peek()
    def pop(self)->Any: return -self._heap.pop()
