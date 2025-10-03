"""Shortest Path Algorithms
===========================

This module provides weighted graph support plus classic shortest path algorithms:

Algorithms Included:
- Dijkstra (non-negative weights)         O((V+E) log V) with heap
- Bellman-Ford (handles negative)         O(V * E)
- Floyd-Warshall (all-pairs)              O(V^3)
- Johnson (sparse all-pairs, reweight)    O(V * E log V)
- DAG Shortest Path (topological order)   O(V + E)

Selection Guide:
- Need single-source, non-negative weights -> Dijkstra
- Need to detect negative cycle or handle negative edges -> Bellman-Ford
- Need distances between all pairs (small/medium dense graph) -> Floyd-Warshall
- Need all-pairs on sparse graph (may contain negative, no negative cycles) -> Johnson
- Single-source on DAG (incl. negative edges, no cycles) -> DAG shortest path

Conventions:
- Vertices assumed hashable (default int)
- WeightedAdjacencyListGraph stores edges as (neighbor, weight)
- Distances returned as dict: vertex -> distance (float('inf') if unreachable)
- Path reconstruction returns list of vertices from source to target

Limitations:
- Dijkstra implementation uses adjacency list + heap; does not decrease-key (push new pair, stale entries skipped)
- Floyd-Warshall includes path reconstruction via next matrix

"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Iterable
import heapq

class WeightedAdjacencyListGraph:
    """Weighted graph using adjacency list: vertex -> list[(neighbor, weight)]."""
    def __init__(self, directed: bool = True) -> None:
        self.adj: Dict[Any, List[Tuple[Any, float]]] = {}
        self.directed = directed

    def add_vertex(self, v: Any) -> None:
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u: Any, v: Any, w: float) -> None:
        if u not in self.adj: self.add_vertex(u)
        if v not in self.adj: self.add_vertex(v)
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def vertices(self):
        return list(self.adj.keys())

    def edges(self):
        for u, nbrs in self.adj.items():
            for v, w in nbrs:
                yield (u, v, w)

    # Convenience for DAG algorithms (out-neighbors)
    def neighbors(self, u: Any) -> Iterable[Tuple[Any, float]]:
        return self.adj.get(u, [])

# ---------------- Dijkstra ----------------

def dijkstra(graph: WeightedAdjacencyListGraph, source: Any) -> Dict[Any, float]:
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    dist[source] = 0.0
    pq: List[Tuple[float, Any]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue  # stale entry
        for v, w in graph.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# ---------------- Dijkstra Path ----------------

def dijkstra_with_path(graph: WeightedAdjacencyListGraph, source: Any, target: Any) -> Tuple[float, List[Any]]:
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    parent: Dict[Any, Optional[Any]] = {v: None for v in graph.vertices()}
    dist[source] = 0.0
    pq: List[Tuple[float, Any]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:  # early exit optional
            break
        if d > dist[u]:
            continue
        for v, w in graph.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    path: List[Any] = []
    if dist[target] != float('inf'):
        cur: Optional[Any] = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return dist[target], path

# ---------------- Bellman-Ford ----------------

def bellman_ford(graph: WeightedAdjacencyListGraph, source: Any) -> Tuple[Dict[Any, float], bool]:
    """Return (distances, has_negative_cycle)."""
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    dist[source] = 0.0
    verts = graph.vertices()
    for _ in range(len(verts) - 1):
        updated = False
        for u, v, w in graph.edges():
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    # Check negative cycle
    has_neg_cycle = False
    for u, v, w in graph.edges():
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break
    return dist, has_neg_cycle

# ---------------- Bellman-Ford with Negative Cycle Extraction ----------------

def bellman_ford_with_cycle(graph: WeightedAdjacencyListGraph, source: Any) -> Tuple[Dict[Any, float], List[Any]]:
    """Bellman-Ford that returns distances and a representative negative cycle (list of vertices) if one exists.

    Cycle extraction approach:
    - Track parent on relaxation.
    - After V-1 iterations, perform one more pass; if an edge can still relax, follow parent pointers V times from the updated vertex to guarantee landing inside a cycle, then walk until repeat to extract cycle.
    """
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    parent: Dict[Any, Optional[Any]] = {v: None for v in graph.vertices()}
    dist[source] = 0.0
    verts = graph.vertices()
    for _ in range(len(verts) - 1):
        updated = False
        for u, v, w in graph.edges():
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            return dist, []
    # Extra pass to detect & extract
    for u, v, w in graph.edges():
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            # v is on or reachable to a negative cycle
            x = v
            for _ in range(len(verts)):
                x = parent[x]  # move inside the cycle
            cycle: List[Any] = []
            cur = x
            while True:
                cycle.append(cur)
                cur = parent[cur]
                if cur is None or cur == x or cur in cycle:
                    if cur == x:
                        cycle.append(cur)
                    break
            # Normalize cycle ordering: rotate so smallest repr first (optional) – keep raw for clarity
            return dist, cycle
    return dist, []

# ---------------- Bellman-Ford Path ----------------

def bellman_ford_with_path(graph: WeightedAdjacencyListGraph, source: Any, target: Any) -> Tuple[float, List[Any], bool]:
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    parent: Dict[Any, Optional[Any]] = {v: None for v in graph.vertices()}
    dist[source] = 0.0
    verts = graph.vertices()
    for _ in range(len(verts) - 1):
        updated = False
        for u, v, w in graph.edges():
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break
    # Negative cycle detection
    for u, v, w in graph.edges():
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return float('-inf'), [], True
    # Reconstruct path
    path: List[Any] = []
    if dist[target] != float('inf'):
        cur: Optional[Any] = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return dist[target], path, False

# ---------------- Floyd-Warshall ----------------

def floyd_warshall(graph: WeightedAdjacencyListGraph) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, Optional[Any]]]]:
    verts = graph.vertices()
    # Map vertices to index
    index = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    dist_matrix = [[float('inf')] * n for _ in range(n)]
    next_matrix: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    for v in verts:
        dist_matrix[index[v]][index[v]] = 0.0
        next_matrix[index[v]][index[v]] = index[v]
    for u, v, w in graph.edges():
        ui, vi = index[u], index[v]
        if w < dist_matrix[ui][vi]:  # handle parallel edges
            dist_matrix[ui][vi] = w
            next_matrix[ui][vi] = vi
        if not graph.directed:
            if w < dist_matrix[vi][ui]:
                dist_matrix[vi][ui] = w
                next_matrix[vi][ui] = ui

    # Triple loop
    for k in range(n):
        for i in range(n):
            if dist_matrix[i][k] == float('inf'):
                continue
            dik = dist_matrix[i][k]
            for j in range(n):
                if dist_matrix[k][j] == float('inf'):
                    continue
                new = dik + dist_matrix[k][j]
                if new < dist_matrix[i][j]:
                    dist_matrix[i][j] = new
                    next_matrix[i][j] = next_matrix[i][k]

    # Convert to dict-of-dicts for usability
    dist_dict: Dict[Any, Dict[Any, float]] = {u: {} for u in verts}
    next_dict: Dict[Any, Dict[Any, Optional[Any]]] = {u: {} for u in verts}
    for u in verts:
        ui = index[u]
        for v in verts:
            vi = index[v]
            dist_dict[u][v] = dist_matrix[ui][vi]
            nxt = next_matrix[ui][vi]
            next_dict[u][v] = verts[nxt] if nxt is not None else None
    return dist_dict, next_dict

# ---------------- Floyd-Warshall Path ----------------

def reconstruct_fw_path(next_dict: Dict[Any, Dict[Any, Optional[Any]]], source: Any, target: Any) -> List[Any]:
    if next_dict[source][target] is None:
        return []
    path = [source]
    cur = source
    while cur != target:
        cur = next_dict[cur][target]
        if cur is None:  # should not happen if path exists
            return []
        path.append(cur)
    return path

# ---------------- Topological Sort (Kahn) ----------------

def topological_order(graph: WeightedAdjacencyListGraph) -> List[Any]:
    indeg: Dict[Any, int] = {v: 0 for v in graph.vertices()}
    for u, v, _ in graph.edges():
        indeg[v] += 1
    from collections import deque
    q = deque([v for v, d in indeg.items() if d == 0])
    order: List[Any] = []
    while q:
        u = q.popleft()
        order.append(u)
        for nbr, _ in graph.neighbors(u):
            indeg[nbr] -= 1
            if indeg[nbr] == 0:
                q.append(nbr)
    if len(order) != len(indeg):
        raise ValueError("Graph is not a DAG (cycle detected)")
    return order

# ---------------- DAG Shortest Paths ----------------

def dag_shortest_paths(graph: WeightedAdjacencyListGraph, source: Any, order: Optional[List[Any]] = None) -> Dict[Any, float]:
    """Compute single-source shortest paths on a DAG (allows negative weights).
    Raises ValueError if cycle detected.
    """
    if order is None:
        order = topological_order(graph)
    dist: Dict[Any, float] = {v: float('inf') for v in graph.vertices()}
    dist[source] = 0.0
    for u in order:
        if dist[u] == float('inf'):
            continue
        for v, w in graph.neighbors(u):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist

# ---------------- Johnson's Algorithm ----------------

def johnson_all_pairs(graph: WeightedAdjacencyListGraph) -> Tuple[Optional[Dict[Any, Dict[Any, float]]], Optional[List[Any]]]:
    """Johnson's algorithm for all-pairs shortest paths on sparse graphs.

    Returns (distance_dict, negative_cycle) where negative_cycle is a list of vertices
    if a negative cycle was detected (in the original graph) else None.
    """
    # Create a new super-source connecting to all vertices with 0 weight
    super_source = object()  # unique sentinel
    temp_graph = WeightedAdjacencyListGraph(directed=True)
    for v in graph.vertices():
        temp_graph.add_edge(super_source, v, 0.0)
    for u, v, w in graph.edges():
        temp_graph.add_edge(u, v, w)
    # Bellman-Ford from super source
    h, neg_cycle_flag = bellman_ford(temp_graph, super_source)
    if neg_cycle_flag:
        # Use extraction on original graph for clarity
        _, cycle = bellman_ford_with_cycle(temp_graph, super_source)
        return None, cycle
    # Reweight edges using potentials h
    vertices = list(graph.vertices())
    all_dist: Dict[Any, Dict[Any, float]] = {}
    for s in vertices:
        # Run Dijkstra with reweighted edges on-the-fly
        dist: Dict[Any, float] = {v: float('inf') for v in vertices}
        dist[s] = 0.0
        pq: List[Tuple[float, Any]] = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in graph.neighbors(u):
                # reweighted edge
                rw = w + h[u] - h[v]
                nd = d + rw
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        # Convert distances back
        for v in vertices:
            if dist[v] < float('inf'):
                dist[v] = dist[v] - h[s] + h[v]
        all_dist[s] = dist
    return all_dist, None

# ---------------- Demonstration ----------------

def _demo():  # pragma: no cover (manual demo)
    g = WeightedAdjacencyListGraph(directed=True)
    edges = [
        ('A','B',4), ('A','C',2), ('C','B',1), ('B','D',5), ('C','D',8), ('C','E',10), ('D','E',2), ('E','D',-2)
    ]
    for u,v,w in edges:
        g.add_edge(u,v,w)

    print("Dijkstra from A:", dijkstra(g,'A'))
    print("Dijkstra A->E:", dijkstra_with_path(g,'A','E'))
    bf_dist, neg = bellman_ford(g,'A')
    print("Bellman-Ford from A:", bf_dist, "neg_cycle=", neg)
    bf_de, path_de, neg2 = bellman_ford_with_path(g,'A','E')
    print("Bellman-Ford A->E:", bf_de, path_de, "neg_cycle=", neg2)
    all_dist, nxt = floyd_warshall(g)
    print("Floyd-Warshall A->E:", all_dist['A']['E'], reconstruct_fw_path(nxt,'A','E'))
    # Johnson demo
    j_dist, cycle = johnson_all_pairs(g)
    if cycle:
        print("Johnson detected negative cycle:", cycle)
    else:
        print("Johnson all-pairs from A to E:", j_dist['A']['E'])

if __name__ == '__main__':
    _demo()
