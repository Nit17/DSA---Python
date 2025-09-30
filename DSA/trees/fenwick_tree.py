"""Fenwick Tree (Binary Indexed Tree) relocated under trees/."""
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
