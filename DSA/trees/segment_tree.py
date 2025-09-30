"""Segment Tree (range sum) relocated under trees/."""
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
