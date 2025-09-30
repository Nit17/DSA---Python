"""Hierarchical & Indexed Data Structures (trees package)
=========================================================
This package groups several foundational tree / tree-like structures frequently
used in algorithm design. Each module contains both implementation and an
extended theoretical overview focusing on invariants, operations, complexities,
design trade-offs, and practical pitfalls.

Included Structures
-------------------
Binary Tree Utilities (``binary_tree``)
	Generic node definition + traversal algorithms (pre/in/post/level) and
	example structural computations (diameter, invert). Useful for reasoning
	about recursion, shape properties, and traversal orders.

Binary Search Tree (``bst``)
	Ordered dictionary / set abstraction providing logarithmic average-case
	search/insert/delete when reasonably balanced. Highlights deletion cases
	and validation techniques. (Not self-balancing.)

Heaps (``heaps``)
	MinHeap / MaxHeap priority queues with O(log n) push/pop and O(1) peek;
	min-heap implemented directly, max-heap via value negation wrapper.

Trie (``trie``)
	Prefix tree enabling O(L) word/prefix queries (L = key length) and efficient
	enumeration of words sharing a prefix; demonstrates memory/time trade-offs.

Segment Tree (``segment_tree``)
	Range query + point update structure (sum variant here) with O(log n)
	queries/updates and O(n) build; extensible to other associative operations
	and lazy propagation for range updates.

Fenwick Tree / BIT (``fenwick_tree``)
	Compact structure supporting prefix sums & point updates in O(log n) with
	small memory footprint; ideal for cumulative frequency tasks and inversion counting.

Choosing the Right Structure
----------------------------
- Need repeated global min/max with interleaved inserts? -> Heap
- Need ordered iteration + predecessor/successor? -> (Self-balancing) BST or skip list (BST here is illustrative)
- Need prefix lookups / autocomplete? -> Trie
- Need point updates + prefix sums (fast & memory light)? -> Fenwick Tree
- Need more general associative range queries or range updates? -> Segment Tree (with lazy propagation)
- Exploring traversal patterns / structural metrics? -> Binary tree utilities

Imports below re-export the public classes/functions for convenience.
"""
from .binary_tree import (
	BinaryTreeNode,
	preorder,
	inorder,
	postorder,
	level_order,
	diameter,
	invert_tree,
)
from .bst import BinarySearchTree
from .heaps import MinHeap, MaxHeap
from .trie import Trie
from .segment_tree import SegmentTree
from .fenwick_tree import FenwickTree
