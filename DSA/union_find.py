"""Disjoint Set Union (Union-Find) – Connectivity with Near-Constant Ops
=======================================================================
Data structure to efficiently track a partition of a set into disjoint subsets
with two core operations:
- find(x): return representative (root) of the set containing x
- union(x, y): merge the sets containing x and y

With path compression (during find) and union by rank/size, the amortized time
per operation is effectively constant: O(alpha(n)) where alpha is inverse Ackermann.

Common uses: connected components, Kruskal's MST, cycle detection in undirected
graphs, dynamic connectivity, DSU on trees, offline query processing.
"""

from __future__ import annotations
from typing import Dict, Hashable


class UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}
        self.size: Dict[Hashable, int] = {}

    def add(self, x: Hashable) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def find(self, x: Hashable) -> Hashable:
        """Find set representative with path compression."""
        if x not in self.parent:
            self.add(x)
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Union by rank. Returns True if merged, False if already in same set."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Attach smaller rank under larger rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def connected(self, a: Hashable, b: Hashable) -> bool:
        return self.find(a) == self.find(b)

    def component_size(self, x: Hashable) -> int:
        r = self.find(x)
        return self.size[r]


if __name__ == "__main__":
    uf = UnionFind()
    for i in range(1, 6):
        uf.add(i)
    uf.union(1, 2)
    uf.union(3, 4)
    uf.union(2, 3)
    print(uf.connected(1, 4), uf.connected(1, 5))  # True False
    print(uf.component_size(1))  # 4
