"""Trie implementation (relocated under trees/)."""
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
