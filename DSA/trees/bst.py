"""
BINARY SEARCH TREE (BST) THEORY
===============================

Definition:
A BST is a binary tree where for every node:
        all keys in left subtree  < node.value
        all keys in right subtree > node.value
Duplicates are typically either disallowed or routed consistently (e.g., to the right).

Core Operations:
- Insert: Recursively (or iteratively) follow left/right based on comparison until None
- Search: Same traversal logic as insert; O(h)
- Delete: Three cases
    1. Leaf node: remove directly
    2. One child: replace node with child
    3. Two children: replace value with inorder successor (min of right subtree) or
         inorder predecessor (max of left subtree), then delete that successor node

Traversal Properties:
- Inorder traversal yields keys in sorted ascending order.
- Preorder can help reconstruct tree with additional structural info (e.g., with inorder or using BST insert semantics).

Complexities (h = height):
Average (balanced): Insert/Search/Delete O(log n)
Worst (skewed: strictly increasing inserts): O(n)

Balancing (NOT implemented here):
Self-balancing variants (AVL, Red-Black, Treap, Splay) maintain O(log n) by rotations.

Validation Techniques:
1. Inorder traversal must produce strictly increasing sequence.
2. Recurse with value range constraints (low < node.value < high).

Space Complexity:
O(h) recursion stack; O(n) storage for n nodes.

Typical Pitfalls:
- Forget to handle duplicates (policy must be explicit)
- Incorrectly handling delete when successor has right child
- Mixing up returning node vs value in recursive delete

Extensions:
- Implement iterative inorder with stack
- Add kth_smallest / kth_largest via augmented counts
- Add range queries (collect keys in [L, R])
- Augment nodes with subtree size or height for order statistics / balancing heuristics
"""
from __future__ import annotations
from typing import Any, Optional, List

class BSTNode:
    __slots__ = ("value","left","right")
    def __init__(self, value: Any) -> None:
        self.value = value; self.left: Optional['BSTNode']=None; self.right: Optional['BSTNode']=None

class BinarySearchTree:
    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
    def insert(self, value: Any) -> None:
        def _ins(node: Optional[BSTNode], v: Any) -> BSTNode:
            if not node: return BSTNode(v)
            if v < node.value: node.left = _ins(node.left, v)
            elif v > node.value: node.right = _ins(node.right, v)
            return node
        self.root = _ins(self.root, value)
    def search(self, value: Any) -> bool:
        n = self.root
        while n:
            if value == n.value: return True
            n = n.left if value < n.value else n.right
        return False
    def inorder(self) -> List[Any]:
        res: List[Any] = []
        def _in(n: Optional[BSTNode]):
            if not n: return
            _in(n.left); res.append(n.value); _in(n.right)
        _in(self.root); return res
    def min_value(self) -> Optional[Any]:
        n=self.root
        if not n: return None
        while n.left: n=n.left
        return n.value
    def max_value(self) -> Optional[Any]:
        n=self.root
        if not n: return None
        while n.right: n=n.right
        return n.value
    def delete(self, value: Any) -> None:
        def _del(node: Optional[BSTNode], v: Any) -> Optional[BSTNode]:
            if not node: return None
            if v < node.value:
                node.left = _del(node.left, v)
            elif v > node.value:
                node.right = _del(node.right, v)
            else:
                if not node.left: return node.right
                if not node.right: return node.left
                succ_parent = node; succ = node.right
                while succ.left:
                    succ_parent = succ; succ = succ.left
                node.value = succ.value
                if succ_parent.left is succ:
                    succ_parent.left = succ.right
                else:
                    succ_parent.right = succ.right
            return node
        self.root = _del(self.root, value)
    def is_valid(self) -> bool:
        def _v(node: Optional[BSTNode], low: Any, high: Any) -> bool:
            if not node: return True
            if (low is not None and node.value <= low) or (high is not None and node.value >= high): return False
            return _v(node.left, low, node.value) and _v(node.right, node.value, high)
        return _v(self.root, None, None)
    def height(self) -> int:
        def _h(n: Optional[BSTNode]) -> int:
            if not n: return 0
            return 1 + max(_h(n.left), _h(n.right))
        return _h(self.root)
