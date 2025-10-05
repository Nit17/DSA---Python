"""Greedy Algorithms – Classic Problems
======================================
This module implements two canonical greedy problems:

1) Activity Selection: choose the maximum number of non-overlapping intervals
   by sorting activities by finish time and greedily picking compatible ones.

2) Huffman Coding: optimal prefix-free coding based on symbol frequencies.
   Includes utilities to build the code map and to encode/decode strings.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq


class GreedyAlgorithms:
    """Greedy algorithms collection."""

    # ------------------------- Activity Selection -------------------------
    def activity_selection(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Select maximum number of mutually compatible activities.

        intervals: list of (start, finish) with start < finish
        Strategy: sort by finish time, then select if start >= last_finish
        Returns the chosen subset of intervals in the order selected.
        """
        if not intervals:
            return []
        intervals_sorted = sorted(intervals, key=lambda x: x[1])
        chosen: List[Tuple[int, int]] = []
        last_finish = -10**18
        for s, f in intervals_sorted:
            if s >= last_finish:
                chosen.append((s, f))
                last_finish = f
        return chosen

    # --------------------------- Huffman Coding ---------------------------
    @dataclass(order=True)
    class _Node:
        freq: int
        char: Optional[str] = None
        left: Optional['GreedyAlgorithms._Node'] = None
        right: Optional['GreedyAlgorithms._Node'] = None

    def huffman_codes(self, freq_map: Dict[str, int]) -> Tuple[Dict[str, str], Optional['_Node']]:
        """Build Huffman codes from a frequency map.

        Returns (code_map, root). If there is only one symbol, assigns code '0'.
        """
        if not freq_map:
            return {}, None

        heap: List[GreedyAlgorithms._Node] = []
        for ch, fr in freq_map.items():
            heapq.heappush(heap, GreedyAlgorithms._Node(fr, ch))

        if len(heap) == 1:
            # Single symbol edge case
            only = heap[0]
            return {only.char: '0'}, only

        # Build tree
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            parent = GreedyAlgorithms._Node(a.freq + b.freq, None, a, b)
            heapq.heappush(heap, parent)

        root = heap[0]
        codes: Dict[str, str] = {}

        def dfs(node: GreedyAlgorithms._Node, path: str) -> None:
            if node.char is not None:
                codes[node.char] = path or '0'
                return
            if node.left:
                dfs(node.left, path + '0')
            if node.right:
                dfs(node.right, path + '1')

        dfs(root, '')
        return codes, root

    def huffman_encode(self, text: str, codes: Dict[str, str]) -> str:
        """Encode text using a code map produced by huffman_codes."""
        return ''.join(codes[ch] for ch in text)

    def huffman_decode(self, bits: str, root: Optional['_Node']) -> str:
        """Decode a bitstring using the Huffman tree root."""
        if root is None:
            return ''
        # Single-node edge case
        if root.char is not None:
            return root.char * len(bits) if bits else root.char
        out: List[str] = []
        node = root
        for b in bits:
            node = node.left if b == '0' else node.right
            if node.char is not None:
                out.append(node.char)
                node = root
        return ''.join(out)


if __name__ == "__main__":
    g = GreedyAlgorithms()
    acts = [(1,4), (3,5), (0,6), (5,7), (8,9), (5,9)]
    print("activity_selection:", g.activity_selection(acts))
    text = "huffman huff!"
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    codes, root = g.huffman_codes(freq)
    enc = g.huffman_encode(text, codes)
    dec = g.huffman_decode(enc, root)
    print("huffman codes size:", len(codes), "ok:", dec == text)
