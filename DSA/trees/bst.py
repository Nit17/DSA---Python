"""Full Binary Search Tree implementation (relocated under trees/)."""
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
