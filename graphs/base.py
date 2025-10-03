"""Graph base abstractions.
Provides a minimal interface that concrete graph classes can follow.
"""
from __future__ import annotations
from typing import Protocol, Iterable, Any, List, Tuple

class GraphBase(Protocol):
    directed: bool
    def vertices(self) -> Iterable[Any]: ...
    def add_vertex(self, v: Any) -> None: ...
    def add_edge(self, u: Any, v: Any, w: float | int = 1) -> None: ...

class WeightedGraphProtocol(GraphBase, Protocol):
    def edges(self) -> Iterable[Tuple[Any, Any, float]]: ...
