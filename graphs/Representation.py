"""
GRAPHS - Representations and Algorithms
========================================

Graphs are fundamental data structures for modeling relationships and networks.
This module covers graph representations and core traversal algorithms.

Graph Representations:
1. Adjacency List: List of lists where index i contains neighbors of vertex i
2. Adjacency Matrix: 2D matrix where matrix[i][j] indicates edge between i and j

Time Complexities:
- Adjacency List: Space O(V + E), traversal O(V + E)
- Adjacency Matrix: Space O(V²), traversal O(V²)
- BFS/DFS: O(V + E) for both representations

Common Use Cases:
- Social networks (friend connections)
- Web crawling (page links)
- Pathfinding (maps, mazes)
- Dependency resolution
- Network analysis
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, Deque
from collections import deque, defaultdict
import heapq

# ==================== GRAPH REPRESENTATIONS ====================

class AdjacencyListGraph:
    """
    Graph representation using adjacency list.
    Each vertex maps to a list of its neighbors.
    """

    def __init__(self, directed: bool = False) -> None:
        self.adj_list: Dict[int, List[int]] = defaultdict(list)
        self.directed = directed
        self.vertices: Set[int] = set()

    def add_vertex(self, vertex: int) -> None:
        """Add a vertex to the graph."""
        self.vertices.add(vertex)

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        """Add an edge between vertices u and v."""
        self.vertices.add(u)
        self.vertices.add(v)
        self.adj_list[u].append(v)
        if not self.directed:
            self.adj_list[v].append(u)

    def get_neighbors(self, vertex: int) -> List[int]:
        """Get neighbors of a vertex."""
        return self.adj_list.get(vertex, [])

    def __str__(self) -> str:
        result = []
        for vertex in sorted(self.vertices):
            neighbors = sorted(self.adj_list[vertex])
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)

class AdjacencyMatrixGraph:
    """
    Graph representation using adjacency matrix.
    matrix[i][j] = 1 if edge exists, 0 otherwise.
    """

    def __init__(self, vertices: int, directed: bool = False) -> None:
        self.vertices = vertices
        self.directed = directed
        self.matrix: List[List[int]] = [[0] * vertices for _ in range(vertices)]

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        """Add an edge between vertices u and v."""
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            self.matrix[u][v] = weight
            if not self.directed:
                self.matrix[v][u] = weight

    def has_edge(self, u: int, v: int) -> bool:
        """Check if edge exists between u and v."""
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            return self.matrix[u][v] != 0
        return False

    def get_neighbors(self, vertex: int) -> List[int]:
        """Get neighbors of a vertex."""
        neighbors = []
        for i in range(self.vertices):
            if self.matrix[vertex][i] != 0:
                neighbors.append(i)
        return neighbors

    def __str__(self) -> str:
        result = []
        for i in range(self.vertices):
            row = [str(self.matrix[i][j]) for j in range(self.vertices)]
            result.append(f"{i}: {row}")
        return "\n".join(result)

# ==================== GRAPH ALGORITHMS ====================

class GraphAlgorithms:
    """
    Core graph algorithms: BFS, DFS, shortest paths.
    """

    @staticmethod
    def bfs(graph: AdjacencyListGraph, start: int) -> Dict[int, int]:
        """
        Breadth-First Search traversal.
        Returns distance from start to each reachable vertex.

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        distances = {}
        visited = set()
        queue = deque([start])
        visited.add(start)
        distances[start] = 0

        while queue:
            current = queue.popleft()
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances

    @staticmethod
    def dfs(graph: AdjacencyListGraph, start: int) -> List[int]:
        """
        Depth-First Search traversal.
        Returns list of vertices in DFS order.

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        result = []

        def dfs_helper(vertex: int):
            visited.add(vertex)
            result.append(vertex)
            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs_helper(neighbor)

        dfs_helper(start)
        return result

    @staticmethod
    def shortest_path_unweighted(graph: AdjacencyListGraph, start: int, end: int) -> Optional[List[int]]:
        """
        Find shortest path in unweighted graph using BFS.
        Returns path as list of vertices, or None if no path.

        Time Complexity: O(V + E)
        """
        parent = {}
        visited = set()
        queue = deque([start])
        visited.add(start)
        parent[start] = None

        while queue:
            current = queue.popleft()
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None

    @staticmethod
    def dijkstra(graph: AdjacencyListGraph, start: int) -> Dict[int, int]:
        """
        Dijkstra's algorithm for shortest paths in weighted graph.
        Assumes non-negative weights.

        Time Complexity: O((V + E) log V) with heap
        Space Complexity: O(V)
        """
        distances = {vertex: float('inf') for vertex in graph.vertices}
        distances[start] = 0
        priority_queue = [(0, start)]  # (distance, vertex)

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor in graph.get_neighbors(current_vertex):
                # For simplicity, assume weight 1; extend for weighted edges
                weight = 1
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

# ==================== DEMONSTRATION ====================

def demonstrate_graphs():
    """
    Demonstrate graph representations and algorithms.
    """
    print("\nGRAPH DEMONSTRATIONS")
    print("=" * 50)

    # Create undirected graph
    print("\n1. Adjacency List Representation:")
    g_list = AdjacencyListGraph(directed=False)
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
    for u, v in edges:
        g_list.add_edge(u, v)

    print("Graph structure:")
    print(g_list)

    print("\nBFS from vertex 0:")
    bfs_distances = GraphAlgorithms.bfs(g_list, 0)
    print(bfs_distances)

    print("\nDFS from vertex 0:")
    dfs_order = GraphAlgorithms.dfs(g_list, 0)
    print(dfs_order)

    print("\nShortest path from 0 to 4:")
    path = GraphAlgorithms.shortest_path_unweighted(g_list, 0, 4)
    print(path)

    print("\n2. Adjacency Matrix Representation:")
    g_matrix = AdjacencyMatrixGraph(vertices=5, directed=False)
    for u, v in edges:
        g_matrix.add_edge(u, v)

    print("Graph structure:")
    print(g_matrix)

    print("\nNeighbors of vertex 1:")
    print(g_matrix.get_neighbors(1))

if __name__ == "__main__":
    demonstrate_graphs()