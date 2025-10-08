"""Suffix Array and LCP (Kasai) – Fast Substring Indexing
=========================================================
Provides:
- build_suffix_array(s): O(n log n) (doubling method)
- build_lcp(s, sa): O(n) Kasai’s algorithm
- substring_search(s, sa, pattern): O(|pattern| log n) via binary search on SA

Useful for fast substring queries, repeated pattern analysis, and as a base
for more advanced structures (suffix tree via SA + LCP, or suffix automaton).
"""

from __future__ import annotations
from typing import List


def build_suffix_array(s: str) -> List[int]:
    """Construct suffix array of s in O(n log n) using the doubling method."""
    n = len(s)
    sa = list(range(n))
    # Initial rank: by char code, then -1 for off-end
    rank = [ord(c) for c in s]
    tmp = [0] * n
    k = 1
    while True:
        # Sort by (rank[i], rank[i+k])
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        # Recompute tmp ranks
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            prev_pair = (rank[prev], rank[prev + k] if prev + k < n else -1)
            cur_pair = (rank[cur], rank[cur + k] if cur + k < n else -1)
            tmp[cur] = tmp[prev] + (1 if cur_pair != prev_pair else 0)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa


def build_lcp(s: str, sa: List[int]) -> List[int]:
    """Construct LCP array using Kasai’s algorithm in O(n).

    LCP[i] = lcp(s[sa[i]:], s[sa[i-1]:]) with LCP[0] = 0.
    """
    n = len(s)
    rank = [0] * n
    for i, pos in enumerate(sa):
        rank[pos] = i
    lcp = [0] * n
    k = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            k = 0
            continue
        j = sa[r - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[r] = k
        if k:
            k -= 1
    return lcp


def substring_search(s: str, sa: List[int], pattern: str) -> int:
    """Search pattern using suffix array and return the minimal index in s.

    Finds any matching suffix via binary search (comparing prefixes), then
    scans adjacent suffixes with the same prefix to compute the minimal
    starting index among all matches. Returns -1 if not found.
    """
    n = len(s)
    m = len(pattern)
    if m == 0:
        return 0
    lo, hi = 0, n - 1
    hit = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = sa[mid]
        seg = s[start:start + m]
        if seg == pattern:
            hit = mid
            break
        elif seg < pattern:
            lo = mid + 1
        else:
            hi = mid - 1
    if hit == -1:
        return -1
    # Expand to collect all contiguous matches around 'hit'
    best = sa[hit]
    i = hit - 1
    while i >= 0 and s[sa[i]:sa[i] + m] == pattern:
        if sa[i] < best:
            best = sa[i]
        i -= 1
    i = hit + 1
    while i < n and s[sa[i]:sa[i] + m] == pattern:
        if sa[i] < best:
            best = sa[i]
        i += 1
    return best


if __name__ == "__main__":
    s = "banana"
    sa = build_suffix_array(s)
    lcp = build_lcp(s, sa)
    print("SA:", sa)     # expected [5, 3, 1, 0, 4, 2]
    print("LCP:", lcp)   # expected [0, 1, 3, 0, 0, 2]
    print("find 'ana':", substring_search(s, sa, "ana"))
