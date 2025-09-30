"""Trees package aggregating hierarchical data structures.

Exports:
- BinaryTree utilities
- BinarySearchTree
- MinHeap / MaxHeap
- Trie
- SegmentTree
- FenwickTree
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
