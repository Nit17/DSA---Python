"""Trie (Prefix Tree)
=====================
The Trie (a.k.a. prefix tree / digital tree) stores strings over an alphabet by
decomposing keys into characters along root-to-leaf paths. Each node represents
the prefix formed by the characters along the path from the root.

Core Use Cases
--------------
- Autocomplete / typeahead suggestions
- Spell checking / dictionary membership
- Prefix, wildcard, or lexicographic queries
- Efficient retrieval of all keys sharing a prefix
- Word games (Boggle, word ladders with pruning)
- Compressed storage of many strings with overlapping prefixes

Node Structure
--------------
Each node holds:
        children: map char -> next node
        end: bool flag indicating that a complete word terminates here

Operations & Complexities (Let L = length of word/prefix)
--------------------------------------------------------
insert(word): traverse/allocate along characters  -> O(L)
search(word): traverse; check end flag            -> O(L)
starts_with(prefix): traverse only                -> O(L)
delete(word): traverse recursively; prune nodes   -> O(L)
list_words(prefix): O(P + K) where P = length to reach prefix node, K = total characters output over all returned words.

Why O(L)? The branching factor is proportional to alphabet size, but each
operation advances one level per character; we do not scan entire sets of words.

Memory Trade-offs
-----------------
Naïve tries can be memory heavy: each node stores a dictionary (hash map) for
children. For dense alphabets or large datasets this overhead dominates.
Optimizations (not implemented here for clarity):
        - Use arrays / fixed-size lists when alphabet is fixed & small (e.g., lowercase a-z)
        - Compress single-child chains (Radix / Patricia trie) to reduce height
        - Store terminal flags & counts in bitsets for compactness
        - DAWG (Directed Acyclic Word Graph) merges identical suffix subgraphs

Deletion Logic
--------------
We recursively unset the ``end`` flag when reaching the terminal node. During
unwinding, if a child becomes non-terminal and has no children, we prune it.
This ensures we don't remove nodes needed by other words sharing a prefix.

Listing Words
-------------
Depth-first traversal from the prefix node collecting all terminals provides a
natural lexicographic order *if* children are iterated in sorted character
order. Here we iterate in dictionary insertion order (Python 3.7+ preserves
insertion order); you can sort keys for deterministic alphabetical output.

Edge Cases / Pitfalls
---------------------
- Inserting the empty string: could set root.end = True (not covered here).
- Case sensitivity: 'Apple' vs 'apple' are distinct unless normalized.
- Unicode grapheme clusters: logical user-perceived characters may span
    multiple code points; a standard trie indexes raw code points.
- Memory blowup: large sparse branching sets (e.g., full Unicode) require
    compression strategies or alternative indices.

Alternatives
------------
- Hash set: O(1) average membership but no prefix enumeration efficiency.
- Sorted array + binary search: O(log n) membership, O(log n + k) prefix retrieval
    (k = matches) via lower_bound / upper_bound indices.
- Ternary search tree: hybrid between trie and BST saving space with char splits.

Possible Extensions
-------------------
- Add word frequency counts to support weighted autocomplete.
- Implement wildcard search (e.g., '?', '*').
- Add prefix deletion (remove all words sharing a prefix).
- Switch to compressed (Patricia) trie to reduce node count.

Implementation Notes
--------------------
We keep ``__slots__`` on TrieNode to reduce per-instance memory overhead by
avoiding the dynamic ``__dict__``. The interface favors clarity over advanced
features.
"""
from __future__ import annotations
from typing import Dict, List

class TrieNode:
    __slots__=("children","end")
    def __init__(self)->None:
        self.children: Dict[str, TrieNode] = {}; self.end=False

class Trie:
    def __init__(self)->None:
        self.root=TrieNode()
    def insert(self, word:str)->None:
        n=self.root
        for ch in word:
            if ch not in n.children: n.children[ch]=TrieNode()
            n=n.children[ch]
        n.end=True
    def search(self, word:str)->bool:
        n=self.root
        for ch in word:
            if ch not in n.children: return False
            n=n.children[ch]
        return n.end
    def starts_with(self,prefix:str)->bool:
        n=self.root
        for ch in prefix:
            if ch not in n.children: return False
            n=n.children[ch]
        return True
    def delete(self, word: str) -> bool:
        def _del(node: TrieNode, i: int) -> bool:
            if i == len(word):
                if not node.end: return False
                node.end = False
                return len(node.children) == 0
            ch = word[i]
            if ch not in node.children: return False
            should_prune = _del(node.children[ch], i+1)
            if should_prune:
                del node.children[ch]
            return not node.end and len(node.children) == 0
        return _del(self.root, 0)
    def list_words(self, prefix: str = "") -> List[str]:
        words: List[str] = []
        n=self.root
        for ch in prefix:
            if ch not in n.children: return []
            n = n.children[ch]
        def dfs(cur: TrieNode, path: List[str]):
            if cur.end: words.append(prefix + ''.join(path))
            for ch, child in cur.children.items():
                path.append(ch); dfs(child, path); path.pop()
        dfs(n, [])
        return words
