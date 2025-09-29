"""
HASHING / DICTIONARIES / SETS - Implementations and Algorithms
==============================================================

This module covers:
1. Custom hash table implementations (separate chaining, open addressing)
2. Core dictionary and set usage patterns
3. Classic hashing-based algorithm problems
4. Collision handling strategies
5. Performance considerations and load factor discussion

Time Complexities (Average Case):
- Insert: O(1)
- Search: O(1)
- Delete: O(1)
Worst-case can degrade to O(n) if all keys collide into one bucket (poor hash).

Load Factor = number_of_elements / number_of_buckets
- Keep load factor <= 0.75 for good performance
- Resize (rehash) when threshold exceeded
"""
from __future__ import annotations
from typing import Any, List, Optional, Iterable, Tuple, Dict

# ==================== SEPARATE CHAINING HASH TABLE ====================

class ChainingHashTable:
    """
    Basic hash table using separate chaining with Python lists for buckets.
    Each bucket stores a list of (key, value) pairs.
    """

    def __init__(self, initial_capacity: int = 8, load_factor: float = 0.75) -> None:
        self._capacity = max(4, initial_capacity)
        self._buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(self._capacity)]
        self._size = 0
        self._load_factor_threshold = load_factor

    def __len__(self) -> int:
        return self._size

    def _bucket_index(self, key: Any) -> int:
        return hash(key) % self._capacity

    def _rehash(self) -> None:
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)

    def put(self, key: Any, value: Any) -> None:
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > self._load_factor_threshold:
            self._rehash()

    def get(self, key: Any) -> Optional[Any]:
        idx = self._bucket_index(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return None

    def remove(self, key: Any) -> bool:
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def contains(self, key: Any) -> bool:
        return self.get(key) is not None

    def keys(self) -> List[Any]:
        return [k for bucket in self._buckets for k, _ in bucket]

    def values(self) -> List[Any]:
        return [v for bucket in self._buckets for _, v in bucket]

    def items(self) -> List[Tuple[Any, Any]]:
        return [pair for bucket in self._buckets for pair in bucket]

# ==================== OPEN ADDRESSING (LINEAR PROBING) ====================

class LinearProbingHashTable:
    """
    Hash table with open addressing using linear probing for collisions.
    Deleted slots marked with a special sentinel so probing continues.
    """

    _DELETED = object()

    def __init__(self, initial_capacity: int = 8, load_factor: float = 0.5) -> None:
        self._capacity = max(4, initial_capacity)
        self._keys: List[Optional[Any]] = [None] * self._capacity
        self._values: List[Optional[Any]] = [None] * self._capacity
        self._size = 0
        self._load_factor_threshold = load_factor

    def __len__(self) -> int:
        return self._size

    def _probe(self, key: Any) -> int:
        idx = hash(key) % self._capacity
        first_deleted = None
        while True:
            k = self._keys[idx]
            if k is None:
                return first_deleted if first_deleted is not None else idx
            if k is LinearProbingHashTable._DELETED and first_deleted is None:
                first_deleted = idx
            elif k == key:
                return idx
            idx = (idx + 1) % self._capacity

    def _rehash(self) -> None:
        old_keys = self._keys
        old_values = self._values
        self._capacity *= 2
        self._keys = [None] * self._capacity
        self._values = [None] * self._capacity
        self._size = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not LinearProbingHashTable._DELETED:
                self.put(k, v)

    def put(self, key: Any, value: Any) -> None:
        if self._size / self._capacity > self._load_factor_threshold:
            self._rehash()
        idx = self._probe(key)
        if self._keys[idx] is None or self._keys[idx] is LinearProbingHashTable._DELETED:
            self._size += 1
        self._keys[idx] = key
        self._values[idx] = value

    def get(self, key: Any) -> Optional[Any]:
        idx = hash(key) % self._capacity
        probes = 0
        while probes < self._capacity:
            k = self._keys[idx]
            if k is None:
                return None
            if k is not LinearProbingHashTable._DELETED and k == key:
                return self._values[idx]
            idx = (idx + 1) % self._capacity
            probes += 1
        return None

    def remove(self, key: Any) -> bool:
        idx = hash(key) % self._capacity
        probes = 0
        while probes < self._capacity:
            k = self._keys[idx]
            if k is None:
                return False
            if k is not LinearProbingHashTable._DELETED and k == key:
                self._keys[idx] = LinearProbingHashTable._DELETED
                self._values[idx] = None
                self._size -= 1
                return True
            idx = (idx + 1) % self._capacity
            probes += 1
        return False

    def contains(self, key: Any) -> bool:
        return self.get(key) is not None

    def keys(self) -> List[Any]:
        return [k for k in self._keys if k is not None and k is not LinearProbingHashTable._DELETED]

# ==================== HASHING ALGORITHMS / PATTERNS ====================

class HashingAlgorithms:
    """Classic hashing-based problems using dict/set in Python."""

    # ----- Two Sum -----
    def two_sum(self, nums: List[int], target: int) -> Optional[Tuple[int, int]]:
        seen: Dict[int, int] = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return (seen[complement], i)
            seen[num] = i
        return None

    # ----- Longest Substring Without Repeating Characters -----
    def longest_unique_substring(self, s: str) -> int:
        last_seen: Dict[str, int] = {}
        start = 0
        max_len = 0
        for i, ch in enumerate(s):
            if ch in last_seen and last_seen[ch] >= start:
                start = last_seen[ch] + 1
            last_seen[ch] = i
            max_len = max(max_len, i - start + 1)
        return max_len

    # ----- Longest Consecutive Sequence -----
    def longest_consecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        longest = 0
        for n in num_set:
            if n - 1 not in num_set:  # Start of sequence
                length = 1
                while n + length in num_set:
                    length += 1
                longest = max(longest, length)
        return longest

    # ----- Group Anagrams -----
    def group_anagrams(self, words: List[str]) -> List[List[str]]:
        groups: Dict[Tuple[int, ...], List[str]] = {}
        for word in words:
            freq = [0] * 26
            for ch in word:
                if 'a' <= ch <= 'z':
                    freq[ord(ch) - ord('a')] += 1
                else:
                    # Non-lowercase letter fallback: use sorted tuple
                    freq = None
                    break
            key = tuple(freq) if freq else tuple(sorted(word))
            groups.setdefault(key, []).append(word)
        return list(groups.values())

    # ----- Subarray Sum Equals K -----
    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        count = 0
        prefix_sum = 0
        freq = {0: 1}
        for num in nums:
            prefix_sum += num
            count += freq.get(prefix_sum - k, 0)
            freq[prefix_sum] = freq.get(prefix_sum, 0) + 1
        return count

    # ----- First Non-Repeating Character -----
    def first_unique_char(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        for i, ch in enumerate(s):
            if freq[ch] == 1:
                return i
        return -1

    # ----- Check Isomorphic Strings -----
    def are_isomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        map_st: Dict[str, str] = {}
        map_ts: Dict[str, str] = {}
        for a, b in zip(s, t):
            if (a in map_st and map_st[a] != b) or (b in map_ts and map_ts[b] != a):
                return False
            map_st[a] = b
            map_ts[b] = a
        return True

    # ----- Valid Anagram -----
    def is_anagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        from collections import Counter
        return Counter(s) == Counter(t)

    # ----- Word Pattern -----
    def word_pattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        if len(pattern) != len(words):
            return False
        map_pw: Dict[str, str] = {}
        map_wp: Dict[str, str] = {}
        for p, w in zip(pattern, words):
            if (p in map_pw and map_pw[p] != w) or (w in map_wp and map_wp[w] != p):
                return False
            map_pw[p] = w
            map_wp[w] = p
        return True

    # ----- Top K Frequent Elements -----
    def top_k_frequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        freq = Counter(nums)
        # Use bucket sort approach
        buckets: List[List[int]] = [[] for _ in range(len(nums) + 1)]
        for num, c in freq.items():
            buckets[c].append(num)
        result: List[int] = []
        for count in range(len(buckets) - 1, 0, -1):
            for num in buckets[count]:
                result.append(num)
                if len(result) == k:
                    return result
        return result

# ==================== DEMONSTRATIONS ====================

def demonstrate_hash_tables() -> None:
    print("\nHASH TABLE DEMONSTRATIONS")
    print("=" * 50)

    # Separate Chaining
    print("\nChainingHashTable:")
    cht = ChainingHashTable()
    cht.put('apple', 1)
    cht.put('banana', 2)
    cht.put('grape', 3)
    cht.put('apple', 10)  # update
    print("Items:", cht.items())
    print("Get apple:", cht.get('apple'))
    print("Contains banana:", cht.contains('banana'))

    # Linear Probing
    print("\nLinearProbingHashTable:")
    lp = LinearProbingHashTable()
    for k, v in [(1, 'A'), (9, 'B'), (17, 'C')]:  # Likely collisions modulo capacity
        lp.put(k, v)
    print("Keys:", lp.keys())
    print("Get 9:", lp.get(9))
    lp.remove(9)
    print("After removing 9, Get 9:", lp.get(9))

    # Algorithms
    print("\nHashing Algorithms:")
    alg = HashingAlgorithms()
    nums = [2, 7, 11, 15]
    print("Two Sum (target=9):", alg.two_sum(nums, 9))
    s = "abcabcbb"
    print("Longest Unique Substring (abcabcbb):", alg.longest_unique_substring(s))
    seq_nums = [100, 4, 200, 1, 3, 2]
    print("Longest Consecutive Sequence:", alg.longest_consecutive(seq_nums))
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print("Group Anagrams:", alg.group_anagrams(words))
    arr = [1, 2, 3, -2, 2, -2, 3]
    print("Subarray Sum Equals 3:", alg.subarray_sum_equals_k(arr, 3))
    print("First Unique Char (leetcode):", alg.first_unique_char("leetcode"))
    print("Isomorphic (egg, add):", alg.are_isomorphic("egg", "add"))
    print("Valid Anagram (anagram, nagaram):", alg.is_anagram("anagram", "nagaram"))
    print("Word Pattern (abba, 'dog cat cat dog'):", alg.word_pattern("abba", "dog cat cat dog"))
    print("Top 2 Frequent [1,1,1,2,2,3]:", alg.top_k_frequent([1,1,1,2,2,3], 2))


def hashing_tips_and_tricks() -> None:
    print("\nHASHING TIPS AND TRICKS")
    print("=" * 50)
    tips = [
        "Use dictionaries for O(1) average insert/lookup",
        "Sets are ideal for membership and uniqueness checks",
        "Prefetch complements in Two Sum using a hash map",
        "Track last seen indices for sliding window duplicate handling",
        "Use a set to detect starts of consecutive sequences",
        "Group anagrams by sorted string or frequency signature",
        "Prefix sums + hashmap solve subarray sum problems efficiently",
        "Bidirectional mapping handles isomorphism and pattern tasks",
        "Bucket sort can avoid full sorting for frequency selection",
        "Resize hash tables when load factor grows to keep performance"
    ]
    for t in tips:
        print(f"- {t}")

if __name__ == "__main__":
    demonstrate_hash_tables()
    hashing_tips_and_tricks()