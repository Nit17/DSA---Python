"""Binary Tree utilities (relocated under trees/)."""
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
