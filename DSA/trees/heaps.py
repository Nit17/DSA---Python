"""Heap implementations (relocated under trees/)."""
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
