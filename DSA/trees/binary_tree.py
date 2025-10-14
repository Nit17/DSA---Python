"""
BINARY TREE THEORY & UTILITIES
==============================

Definition:
A Binary Tree is a hierarchical structure where each node has at most two children:
left and right. It is the foundation for many specialized trees (BST, AVL, Heap, etc.).

Key Concepts:
- Depth: Number of edges from root to the node.
- Height (of node): Longest downward path to a leaf. Height of tree = height(root).
- Full Binary Tree: Every node has 0 or 2 children.
- Complete Binary Tree: All levels filled except possibly last, filled left to right.
- Perfect Binary Tree: All internal nodes have 2 children and all leaves at same depth.
- Balanced (loosely): Left/right subtree heights differ by at most 1 (strict definitions vary).

Traversals (depth-first):
- Preorder (Root, Left, Right): Used to clone/serialize a tree.
- Inorder (Left, Root, Right): Yields sorted order in a Binary Search Tree.
- Postorder (Left, Right, Root): Useful for deleting/freeing nodes bottom-up.

Breadth-first:
- Level Order: Processes nodes level by level using a queue. Useful for shortest-path style logic in unweighted tree, also for building complete trees.

Diameter:
Number of edges (or nodes) on the longest path between any two nodes. We compute in O(n)
by tracking the sum of left/right depths at each node while computing depths.

Inversion (Mirror):
Swap left/right children recursively. Often asked to test tree recursion fundamentals.

Complexities:
Let n = number of nodes, h = height.
- Traversals: O(n) time, O(h) space (recursion stack) average; worst-case O(n) space for skewed tree.
- Height / Diameter: O(n)
- Insert/Search in plain binary tree (no ordering): O(n) worst; structure-dependent.

Common Pitfalls:
- Forgetting to handle None in recursive base cases.
- Confusing depth (from root) with height (to leaf).
- Returning wrong value when combining recursive subtree info (e.g., diameter vs depth).

When To Use a Plain Binary Tree:
- When structure comes from external shape (e.g., parsing expression trees, hierarchical decompositions) rather than ordered key operations.

Provided Functions:
- preorder / inorder / postorder / level_order
- diameter: longest path in edges
- invert_tree: mirror transformation

Extension Ideas:
- Add iterative traversal variants (using explicit stacks).
- Add serialization/deserialization (e.g., level-order with null markers).
- Add balance checking returning height in same pass.
"""
from __future__ import annotations
from typing import Any, List, Optional, Deque
from collections import deque

class BinaryTreeNode:
    __slots__=("value","left","right")
    def __init__(self,value:Any,left:Optional['BinaryTreeNode']=None,right:Optional['BinaryTreeNode']=None)->None:
        self.value=value; self.left=left; self.right=right

def preorder(root: Optional[BinaryTreeNode]) -> List[Any]:
    res: List[Any]=[]
    def dfs(n: Optional[BinaryTreeNode]):
        if not n: return
        res.append(n.value); dfs(n.left); dfs(n.right)
    dfs(root); return res

def inorder(root: Optional[BinaryTreeNode]) -> List[Any]:
    res: List[Any]=[]
    def dfs(n: Optional[BinaryTreeNode]):
        if not n: return
        dfs(n.left); res.append(n.value); dfs(n.right)
    dfs(root); return res

def postorder(root: Optional[BinaryTreeNode]) -> List[Any]:
    res: List[Any]=[]
    def dfs(n: Optional[BinaryTreeNode]):
        if not n: return
        dfs(n.left); dfs(n.right); res.append(n.value)
    dfs(root); return res

def level_order(root: Optional[BinaryTreeNode]) -> List[List[Any]]:
    if not root: return []
    q: Deque[BinaryTreeNode]=deque([root]); levels: List[List[Any]]=[]
    while q:
        size=len(q); cur: List[Any]=[]
        for _ in range(size):
            n=q.popleft(); cur.append(n.value)
            if n.left: q.append(n.left)
            if n.right: q.append(n.right)
        levels.append(cur)
    return levels

def diameter(root: Optional[BinaryTreeNode]) -> int:
    best=0
    def depth(n: Optional[BinaryTreeNode])->int:
        nonlocal best
        if not n: return 0
        l=depth(n.left); r=depth(n.right)
        best=max(best,l+r)
        return 1+max(l,r)
    depth(root); return best

def invert_tree(root: Optional[BinaryTreeNode]) -> Optional[BinaryTreeNode]:
    if not root: return None
    root.left,root.right=invert_tree(root.right),invert_tree(root.left)
    return root
