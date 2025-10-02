"""
GRAPHS - Representations and Traversal Algorithms
=================================================

This module provides graph data structures and traversal algorithms:
- Adjacency List (efficient for sparse graphs)
- Adjacency Matrix (useful for dense graphs or constant-time edge existence checks)
- BFS (Breadth-First Search)
- DFS (Depth-First Search) recursive and iterative
- Multi-source BFS (for simultaneous wavefront expansion)
- Unweighted shortest path (via BFS)
- Dijkstra's algorithm (skeleton for weighted graphs with non-negative weights)

Traversal Selection:
- Use BFS when you need the shortest number of edges or level structure.
- Use DFS for exploring all paths, detecting cycles, or generating topological order (in DAGs).
- Multi-source BFS when several starting points diffuse outward (e.g., rotting oranges, fire spread).

All traversals assume vertices are integers unless otherwise extended.
"""
from __future__ import annotations
from typing import List, Dict, Set, Optional, Deque, Iterable
from collections import deque, defaultdict
import heapq

# -------------------- Graph Representations --------------------
class AdjacencyListGraph:
    def __init__(self, directed: bool = False) -> None:
        self.adj_list: Dict[int, List[int]] = defaultdict(list)
        self.directed = directed
        self.vertices: Set[int] = set()

    def add_vertex(self, v: int) -> None:
        self.vertices.add(v)

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        # weight placeholder for future weighted extension
        self.vertices.update([u, v])
        self.adj_list[u].append(v)
        if not self.directed:
            self.adj_list[v].append(u)

    def get_neighbors(self, v: int) -> List[int]:
        return self.adj_list.get(v, [])

    def __str__(self) -> str:
        lines = []
        for v in sorted(self.vertices):
            lines.append(f"{v}: {sorted(self.adj_list[v])}")
        return "\n".join(lines)

class AdjacencyMatrixGraph:
    def __init__(self, vertices: int, directed: bool = False) -> None:
        self.vertices = vertices
        self.directed = directed
        self.matrix: List[List[int]] = [[0]*vertices for _ in range(vertices)]

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            self.matrix[u][v] = weight
            if not self.directed:
                self.matrix[v][u] = weight

    def get_neighbors(self, v: int) -> List[int]:
        return [i for i in range(self.vertices) if self.matrix[v][i] != 0]

    def __str__(self) -> str:
        lines = []
        for i in range(self.vertices):
            lines.append(f"{i}: {[self.matrix[i][j] for j in range(self.vertices)]}")
        return "\n".join(lines)

# -------------------- Algorithms --------------------
class GraphAlgorithms:
    @staticmethod
    def bfs(graph: AdjacencyListGraph, start: int) -> Dict[int, int]:
        distances: Dict[int, int] = {start: 0}
        q: Deque[int] = deque([start])
        visited = {start}
        while q:
            cur = q.popleft()
            for nb in graph.get_neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    distances[nb] = distances[cur] + 1
                    q.append(nb)
        return distances

    @staticmethod
    def multi_source_bfs(graph: AdjacencyListGraph, starts: Iterable[int]) -> Dict[int, int]:
        starts = list(starts)
        distances: Dict[int, int] = {s: 0 for s in starts}
        q: Deque[int] = deque(starts)
        visited = set(starts)
        while q:
            cur = q.popleft()
            for nb in graph.get_neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    distances[nb] = distances[cur] + 1
                    q.append(nb)
        return distances

    @staticmethod
    def dfs(graph: AdjacencyListGraph, start: int) -> List[int]:
        visited: Set[int] = set()
        order: List[int] = []
        def helper(v: int):
            visited.add(v)
            order.append(v)
            for nb in graph.get_neighbors(v):
                if nb not in visited:
                    helper(nb)
        helper(start)
        return order

    @staticmethod
    def dfs_iterative(graph: AdjacencyListGraph, start: int) -> List[int]:
        visited: Set[int] = set()
        order: List[int] = []
        stack: List[int] = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            order.append(v)
            # Reverse neighbors for similar order to recursive (optional)
            for nb in reversed(graph.get_neighbors(v)):
                if nb not in visited:
                    stack.append(nb)
        return order

    @staticmethod
    def shortest_path_unweighted(graph: AdjacencyListGraph, start: int, end: int) -> Optional[List[int]]:
        parent: Dict[int, Optional[int]] = {start: None}
        q: Deque[int] = deque([start])
        visited = {start}
        while q:
            cur = q.popleft()
            if cur == end:
                path: List[int] = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for nb in graph.get_neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    q.append(nb)
        return None

    @staticmethod
    def dijkstra(graph: AdjacencyListGraph, start: int) -> Dict[int, int]:
        dist = {v: float('inf') for v in graph.vertices}
        dist[start] = 0
        pq: List[tuple[int,int]] = [(0, start)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist[v]:
                continue
            for nb in graph.get_neighbors(v):
                w = 1  # placeholder uniform weight
                nd = d + w
                if nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        return dist

# -------------------- Demonstration --------------------
def demonstrate():
    g = AdjacencyListGraph()
    for u, v in [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4)]:
        g.add_edge(u,v)
    print("Adjacency List:\n", g)
    alg = GraphAlgorithms()
    print("BFS distances from 0:", alg.bfs(g,0))
    print("DFS recursive:", alg.dfs(g,0))
    print("DFS iterative:", alg.dfs_iterative(g,0))
    print("Shortest path 0->4:", alg.shortest_path_unweighted(g,0,4))
    print("Multi-source BFS from {0,4}:", alg.multi_source_bfs(g,[0,4]))

if __name__ == "__main__":
    demonstrate()
