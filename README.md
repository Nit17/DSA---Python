# DSA - Python 🐍

A comprehensive collection of Data Structures and Algorithms implemented in Python with detailed explanations, examples, and performance analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Modules](#modules)
  - [Time and Space Complexity](#time-and-space-complexity)
  - [Recursion](#recursion)
  - [Arrays/Lists](#arrayslists)
  - [Strings](#strings)
  - [Linked Lists](#linked-lists)
  - [Stacks](#stacks)
  - [Queues](#queues)
  - [Hashing / Dictionaries / Sets](#hashing--dictionaries--sets)
  - [Trees / Heaps / Tries](#trees--heaps--tries)
- [Features](#features)
- [Performance Analysis](#performance-analysis)
- [Contributing](#contributing)
- [Author](#author)

## 🎯 Overview

This repository contains implementations of fundamental data structures and algorithms in Python. Each module is designed to be:

- **Educational**: Clear explanations and step-by-step examples
- **Practical**: Real-world problem solutions
- **Performance-focused**: Time and space complexity analysis
- **Interview-ready**: Common coding interview problems and patterns

> NOTE: Every major module now starts with an expanded in-file theory docstring covering: core definitions, invariants, complexity tables, trade-offs, pitfalls, implementation details, and selection guidance. Open the source to study deeper theory alongside code.

## 🧭 Data Structure Selection Cheat Sheet

| Goal / Need | Recommended Structure | Key Ops (Avg) | Why | Caveats |
|-------------|-----------------------|---------------|-----|---------|
| Fast random index access | List (array) | get/set O(1) | Contiguous memory, cache friendly | Middle insert/delete O(n) |
| Frequent inserts/deletes at ends (both) | Deque | append/pop ends O(1) | Doubly-ended optimized | No random index O(1) |
| Maintain sorted order with occasional search | (Not implemented) Balanced BST / `bisect` + list | search/log n (BST) | Ordered iteration, predecessor queries | Python list inserts O(n) |
| Priority processing (min first) | MinHeap | push/pop O(log n), peek O(1) | Partial order minimal overhead | No fast arbitrary deletion |
| Dynamic set / fast membership | Set / Hash Table | add/find/remove O(1) | Expected constant time | Worst-case O(n) collisions |
| Key → Value mapping | Dict / Hash Table | get/put O(1) | Ubiquitous, optimized | Unordered (insertion order preserved but not sorted) |
| Prefix queries / autocomplete | Trie | insert/search O(L) | Shared prefix compression | Higher memory overhead |
| Range sum + point update | Fenwick Tree | update/query O(log n) | Simple, low memory | Hard for min/max ops |
| Range sum/min + range updates (extensible) | Segment Tree (+lazy) | query/update O(log n) | Supports many associative ops | More code + memory |
| Streaming top-k (k small) | Heap (size k) | push/pop O(log k) | Keeps k best efficiently | Not sorted internally |
| All permutations / combinations generation | Backtracking + recursion | – | Natural tree exploration | Exponential time |
| Detect cycles / predecessor path length | Hash set (visited) | O(1) per step | Cycle detection in graphs | Memory for visited |
| LRU Cache | Ordered dict / deque+dict | O(1) | Quick recency tracking | Need to manage eviction logic |
| Monotonic next greater / sliding window max | Monotonic Stack / Deque | O(n) total | Each element pushed/popped ≤ once | Logic bugs easy if conditions wrong |
| Shortest path unweighted | Queue (BFS) | O(V+E) | Layered expansion ensures minimal edges | Requires adjacency structure |
| Balanced frequency counts | Hash map + counters | O(1) updates | Direct tallies | Large key diversity memory cost |

Legend: L = length of key/string; V = vertices; E = edges; k = retained subset size.

> Selection rule of thumb: prefer the simplest structure that meets performance constraints; introduce specialized trees/tries/heaps only when profiling or requirements justify added complexity.

## 📁 Repository Structure

```
DSA-Python/
├── DSA/
│   ├── time_space_complexity.py    # Complexity analysis and examples
│   ├── recursion.py                # Recursion patterns and problems
│   ├── arrays_lists.py             # Array operations and algorithms
│   ├── strings.py                  # String operations and algorithms
│   ├── linked_lists.py             # Linked list implementations
│   ├── stacks.py                   # Stack implementations and algorithms
│   ├── queues.py                   # Queue implementations and algorithms
│   ├── hashing.py                  # Hash tables, hashing algorithms
│   ├── trees/                      # Tree & hierarchical structures package
│   │   ├── __init__.py             # Exports aggregated tree structures
│   │   ├── binary_tree.py          # Generic binary tree traversals & utilities
│   │   ├── bst.py                  # Binary Search Tree implementation
│   │   ├── heaps.py                # MinHeap / MaxHeap implementations
│   │   ├── trie.py                 # Trie (prefix tree)
│   │   ├── segment_tree.py         # Segment Tree (range sum)
│   │   └── fenwick_tree.py         # Fenwick Tree / Binary Indexed Tree
├── README.md                       # This file
└── .git/                          # Git repository files
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Git (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/Nit17/DSA---Python.git
cd DSA-Python
```

### No Additional Dependencies Required
All modules use only Python's built-in libraries, so no pip installations are needed!

## ▶️ How to Run

### Option 1: Run Individual Modules

Each module can be executed independently to see demonstrations:

```bash
# Run time and space complexity examples
python DSA/time_space_complexity.py

# Run recursion demonstrations
python DSA/recursion.py

# Run arrays/lists demonstrations
python DSA/arrays_lists.py

# Run strings demonstrations
python DSA/strings.py
```

### Option 2: Import and Use in Your Code

```python
# Import specific classes
from DSA.recursion import RecursionExamples
from DSA.arrays_lists import ArrayOperations, ArrayAlgorithms

ops = ArrayOperations()

```

### Recursion

**File**: `DSA/recursion.py`

**What it covers**:
- Core recursion theory (base/recursive case design, stack frames)
- Patterns: linear, binary, tail, mutual, backtracking, divide & conquer
- Memoization vs bottom-up dynamic programming
- Tree/graph style problems (permutations, subsets, N-Queens, Sudoku)
- Performance comparisons (naive vs optimized Fibonacci)

**Key Classes**:
- `RecursionExamples`: Wide catalog of classic recursive problems
- `RecursionPatterns`: Demonstrates canonical recursion archetypes

**Sample Concepts**:
- Factorial, Fibonacci (naive + memo + fast power), GCD
- Backtracking (N-Queens, Sudoku), combinatorial generation

**Tip**: View the top-level docstring for advanced sections on memoization trade-offs and recurrence modeling.

### Stacks

**File**: `DSA/stacks.py`

**What it covers**:
- Array-based stack and linked-list-based stack implementations
- MinStack with O(1) get_min
- Expression parsing: infix to postfix and evaluation
- Parentheses/Bracket validation
- Monotonic stack patterns: next greater elements, daily temperatures
- Classic hard problems: largest rectangle in histogram, trapping rain water
- Path simplification

**Key Classes**:
- `ArrayStack`: Simple list-backed stack
- `LinkedListStack`: Singly-linked stack
- `MinStack`: Stack with O(1) minimum retrieval
- `StackAlgorithms`: Algorithms using stacks and monotonic patterns

**Featured Operations/Algorithms**:
- push, pop, peek/top, is_empty, size, clear
- is_valid_parentheses()
- infix_to_postfix(), evaluate_postfix()
- next_greater_elements(), daily_temperatures()
- largest_rectangle_histogram(), trapping_rain_water()
- simplify_path()

**Sample Usage**:
```python
from DSA.stacks import ArrayStack, LinkedListStack, MinStack, StackAlgorithms

# Basic stacks
s = ArrayStack()
- Advanced algorithms (N-Queens, Sudoku solver)
- Optimization techniques (memoization)
- Common recursion patterns and when to use them

**Key Classes**:
- `RecursionExamples`: 20+ recursion problems with solutions
- `RecursionPatterns`: Common patterns and techniques

**Featured Algorithms**:
ms = MinStack()
for v in [3, 5, 2, 2, 4]:
  ms.push(v)
print(ms.get_min())  # 2
ms.pop(); ms.pop(); print(ms.get_min())  # 2
ms.pop(); print(ms.get_min())  # 3

# Algorithms
algo = StackAlgorithms()
print(algo.is_valid_parentheses('{[()]}()'))  # True
post = algo.infix_to_postfix('3+(4*5-6)/(1+2)')
print(' '.join(post))  # 3 4 5 * 6 - 1 2 + / +
print(algo.evaluate_postfix(post))  # 7
print(algo.next_greater_elements([2,1,2,4,3]))  # [4,2,4,-1,-1]
print(algo.daily_temperatures([73,74,75,71,69,72,76,73]))  # [1,1,4,2,1,1,0,0]
print(algo.largest_rectangle_histogram([2,1,5,6,2,3]))  # 10
print(algo.trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(algo.simplify_path('/a//b////c/d//././/..'))  # /a/b/c
```

### Queues

**File**: `DSA/queues.py`

**What it covers**:
- ArrayQueue with circular indexing and capacity management
- LinkedListQueue with unlimited capacity
- CircularQueue with fixed-size circular buffer
- ArrayDeque (double-ended queue) for both ends operations
- PriorityQueue using min-heap for priority-based processing
- BFS algorithms for graphs and trees
- Sliding window maximum/minimum with monotonic deque
- Task scheduling and LRU cache implementations

**Key Classes**:
- `ArrayQueue`: Efficient array-based queue with front/rear pointers
- `LinkedListQueue`: Node-based queue with unlimited capacity
- `CircularQueue`: Fixed-size circular buffer implementation
- `ArrayDeque`: Double-ended queue using collections.deque
- `PriorityQueue`: Min-heap based priority queue
- `QueueAlgorithms`: BFS, sliding window, and scheduling algorithms

**Featured Operations**:
- enqueue, dequeue, front, rear, is_empty, is_full
- add_front, add_rear, remove_front, remove_rear (deque)
- enqueue with priority, peek_priority (priority queue)

**Featured Algorithms**:
- BFS tree level-order traversal
- BFS graph traversal and shortest path
- Sliding window maximum/minimum using monotonic deque
- CPU task scheduling with cooldown periods
- LRU cache simulation using deque + dictionary

**Sample Usage**:
```python
from DSA.queues import ArrayQueue, LinkedListQueue, CircularQueue, ArrayDeque, PriorityQueue, QueueAlgorithms

# Basic queues
aq = ArrayQueue(10)
aq.enqueue(1); aq.enqueue(2); aq.enqueue(3)
print(f"Front: {aq.front()}, Size: {aq.size()}")  # Front: 1, Size: 3
print(f"Dequeue: {aq.dequeue()}")  # Dequeue: 1

llq = LinkedListQueue()
llq.enqueue('A'); llq.enqueue('B')
print(f"Rear: {llq.rear()}")  # Rear: B

# Circular queue
cq = CircularQueue(3)
for i in [10, 20, 30]: cq.enqueue(i)
print(f"Is Full: {cq.is_full()}")  # Is Full: True

# Deque (double-ended)
dq = ArrayDeque()
dq.add_front(1); dq.add_rear(2); dq.add_front(0)
print(dq)  # Deque([0, 1, 2])
print(f"Remove rear: {dq.remove_rear()}")  # Remove rear: 2

# Priority queue
pq = PriorityQueue()
pq.enqueue('Task C', 3); pq.enqueue('Task A', 1); pq.enqueue('Task B', 2)
print(f"Highest priority: {pq.dequeue()}")  # Highest priority: Task A

# Algorithms
algo = QueueAlgorithms()
graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
print(algo.bfs_graph(graph, 'A'))  # ['A', 'B', 'C', 'D']
print(algo.shortest_path_unweighted(graph, 'A', 'D'))  # ['A', 'B', 'D']

# Sliding window maximum
nums = [1, 3, -1, -3, 5, 3, 6, 7]
print(algo.sliding_window_maximum(nums, 3))  # [3, 3, 5, 5, 6, 7]

# LRU Cache simulation
ops = [('put', 1, 1), ('put', 2, 2), ('get', 1), ('put', 3, 3), ('get', 2)]
results = algo.lru_cache_simulation(2, ops)
print([r for r in results if r is not None])  # [1]
```

### Hashing / Dictionaries / Sets

**File**: `DSA/hashing.py`

**What it covers**:
- Separate chaining & linear probing hash tables
- Collision strategies and load factor management
- Algorithmic patterns: Two Sum, frequency counting, longest unique substring, longest consecutive sequence
- Performance & security pitfalls (clustering, adversarial collisions)

**Key Classes**:
- `ChainingHashTable`: Bucketed list-of-pairs approach
- `LinearProbingHashTable`: Open addressing with tombstones
- `HashingAlgorithms`: Interview-style hashing problems

**Theory Highlights**:
- Hash function desiderata: uniformity, determinism, speed
- Load factor α impact on expected chain/probe length
- Amortized O(1) insertion via resize doubling
- Primary clustering in linear probing; mitigation techniques
- Hash randomization for collision attack resistance

**Sample Usage**:
```python
from DSA.hashing import ChainingHashTable, LinearProbingHashTable, HashingAlgorithms

cht = ChainingHashTable(); cht.put('apple', 1); cht.put('apple', 5)
print(cht.get('apple'))  # 5

lp = LinearProbingHashTable(); lp.put(10, 'X'); lp.put(18, 'Y')
print(lp.get(18))  # 'Y'

alg = HashingAlgorithms()
print(alg.two_sum([2,7,11,15], 9))                # (0,1)
print(alg.longest_unique_substring('abcabcbb'))   # 3
print(alg.longest_consecutive([100,4,200,1,3,2])) # 4
```

**What it covers**:
- Custom hash tables: Separate Chaining & Linear Probing (open addressing)
- Core dict/set usage patterns and performance
- Collision handling and load factor management
- Classic hash-based interview problems
- Frequency counting, grouping, membership optimization

**Key Classes**:
- `ChainingHashTable`: Buckets of key-value lists with rehashing
- `LinearProbingHashTable`: Open addressing with tombstone handling
- `HashingAlgorithms`: Two Sum, longest unique substring, etc.

**Featured Algorithms**:
- Two Sum (O(n) hash map lookup)
- Longest substring without repeating characters (sliding window + map)
- Longest consecutive sequence (set boundary expansion)
- Group anagrams (frequency signature hashing)
- Subarray sum equals K (prefix sums + hashmap)
- First unique character detection
- Isomorphic strings / word pattern mapping
- Top K frequent elements (bucket technique)

**Sample Usage**:
```python
from DSA.hashing import ChainingHashTable, LinearProbingHashTable, HashingAlgorithms

# Custom hash tables
cht = ChainingHashTable()
cht.put('apple', 1)
cht.put('banana', 2)
print(cht.get('apple'))  # 1

lp = LinearProbingHashTable()
lp.put(10, 'X'); lp.put(18, 'Y')
print(lp.get(18))  # 'Y'

# Algorithms
alg = HashingAlgorithms()
print(alg.two_sum([2,7,11,15], 9))  # (0,1)
print(alg.longest_unique_substring('abcabcbb'))  # 3
print(alg.longest_consecutive([100,4,200,1,3,2]))  # 4
print(alg.group_anagrams(["eat","tea","tan","ate","nat","bat"]))
print(alg.subarray_sum_equals_k([1,2,3,-2,2,-2,3], 3))
```

### Trees / Heaps / Tries

This collection covers fundamental hierarchical and indexed data structures used for ordered data, priority scheduling, prefix querying, and efficient range queries.

**Files**:
- `binary_tree.py` – Generic binary tree node, traversals, structural algorithms
- `bst.py` – Binary Search Tree with insert/search/delete/validate
- `heaps.py` – MinHeap and MaxHeap (priority queues)
- `trie.py` – Prefix tree for word dictionaries and autocomplete
- `segment_tree.py` – Range sum Segment Tree (build/query/update)
- `fenwick_tree.py` – Fenwick Tree (Binary Indexed Tree) for prefix/range sums

#### Binary Tree (`binary_tree.py`)
**What it covers**:
- Preorder, inorder, postorder, level-order traversals
- Height, balance check, diameter
- Count nodes, leaves
- Max depth, min depth
- Path sum existence and all root→leaf paths
- Lowest Common Ancestor (general binary tree)
- Invert (mirror) tree

**Sample Usage**:
```python
from DSA.binary_tree import BinaryTreeNode, preorder, inorder, level_order, height, diameter, invert_tree

root = BinaryTreeNode(1,
     BinaryTreeNode(2, BinaryTreeNode(4), BinaryTreeNode(5)),
     BinaryTreeNode(3))
print(preorder(root))        # [1,2,4,5,3]
print(inorder(root))         # [4,2,5,1,3]
print(level_order(root))     # [[1],[2,3],[4,5]]
print(height(root))          # 3
print(diameter(root))        # 3 (edges)
invert_tree(root)
print(preorder(root))        # mirrored order
```

#### Binary Search Tree (`bst.py`)
**What it covers**:
- Insert, search, delete (all cases)
- Inorder traversal (sorted order)
- Min / max value
- Height and validation

**Sample Usage**:
```python
from DSA.bst import BinarySearchTree
bst = BinarySearchTree()
for v in [8,3,10,1,6,14,4,7,13]:
  bst.insert(v)
print(bst.inorder())     # [1,3,4,6,7,8,10,13,14]
print(bst.search(6))     # True
bst.delete(3)
print(bst.inorder())     # [1,4,6,7,8,10,13,14]
```

#### Heaps (`heaps.py`)
**What it covers**:
- MinHeap: push/pop/peek/heapify
- MaxHeap: built via value negation layer over MinHeap
- Priority queue behavior (always extract smallest/largest)

**Sample Usage**:
```python
from DSA.heaps import MinHeap, MaxHeap

mh = MinHeap([5,3,8,1])
mh.push(0)
print(mh.pop())      # 0
print(mh.pop())      # 1

xh = MaxHeap([5,3,8,1])
xh.push(10)
print(xh.pop())      # 10
```

#### Trie (`trie.py`)
**What it covers**:
- Insert, exact word search
- Prefix search (starts_with)
- Delete words (prunes nodes safely)
- Enumerate words (optionally by prefix)

**Sample Usage**:
```python
from DSA.trie import Trie
tr = Trie()
for w in ["apple","app","apex","bat"]:
  tr.insert(w)
print(tr.search("app"))           # True
print(tr.starts_with("ap"))       # True
print(tr.list_words("ap"))        # ['app','apple','apex'] (order may vary)
tr.delete("app")
print(tr.search("app"))           # False
```

#### Segment Tree (`segment_tree.py`)
**What it covers**:
- Build from array (range sums)
- Range sum queries
- Point updates
- O(log n) per query/update

**Sample Usage**:
```python
from DSA.segment_tree import SegmentTree
data = [2,1,5,3,4]
st = SegmentTree(data)
print(st.range_sum(1,3))  # 1+5+3 = 9
st.update(2,10)
print(st.range_sum(0,4))  # 2+1+10+3+4 = 20
```

#### Fenwick Tree / Binary Indexed Tree (`fenwick_tree.py`)
**What it covers**:
- Point updates and prefix sums
- Range sum via prefix differences
- More memory-efficient than segment tree for some use-cases

**Sample Usage**:
```python
from DSA.fenwick_tree import FenwickTree
data = [2,1,5,3,4]
ft = FenwickTree.build(data)
print(ft.prefix_sum(2))     # 2+1+5 = 8
ft.update(2,5)              # add 5 -> element becomes 10
print(ft.range_sum(0,4))    # total updated sum
```

**Choosing Between Structures**:
- Use a simple binary tree for traversal/path problems.
- Use BST for ordered data with dynamic inserts (balanced tree variants like AVL/Red-Black improve worst-case).
- Use heaps for priority scheduling (min or max extraction).
- Use trie for fast prefix querying (autocomplete, dictionaries).
- Use segment tree for frequent range queries + updates on static-sized arrays.
- Use Fenwick (BIT) for prefix/range sums when operations are additive and memory efficiency matters.

- Factorial, Fibonacci (naive and optimized)
- String reversal, palindrome checking
- Array operations (sum, max, binary search)
- Sorting algorithms (merge sort, quick sort)
- Backtracking problems (N-Queens, Sudoku)
- Tree-like recursion (permutations, subsets)

**Sample Usage**:
```python
from DSA.recursion import RecursionExamples

rec = RecursionExamples()
print(rec.factorial(5))                    # 120
print(rec.fibonacci_optimized(10))         # 55 (fast!)
print(rec.is_palindrome("racecar"))        # True
```

### Arrays/Lists

**File**: `DSA/arrays_lists.py`

**What it covers**:
- Fundamental array operations (CRUD)
- Search algorithms (linear, binary)
- Sorting algorithms (bubble, selection, insertion, quick, merge)
- Classic array problems and interview questions
- Array manipulation techniques
- Common patterns and optimization strategies

**Key Classes**:
- `ArrayOperations`: Basic array operations and manipulations
- `ArrayAlgorithms`: Advanced algorithms and classic problems
- `ArrayPatterns`: Common problem-solving patterns

**Featured Problems**:
- Two Sum, Three Sum
- Maximum Subarray Sum (Kadane's Algorithm)
- Buy/Sell Stock for maximum profit
- Container with Most Water
- Trapping Rainwater
- Finding missing/duplicate numbers
- Array rotation and reversal

**Sample Usage**:
```python
from DSA.arrays_lists import ArrayOperations, ArrayAlgorithms

ops = ArrayOperations()
algos = ArrayAlgorithms()

# Basic operations
arr = [1, 2, 3, 4, 5]
ops.rotate_right(arr, 2)  # [4, 5, 1, 2, 3]

# Classic problems
indices = algos.two_sum([2, 7, 11, 15], 9)  # [0, 1]
max_sum = algos.maximum_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])  # 6
```

### Strings

**File**: `DSA/strings.py`

**What it covers**:
- Basic string operations (access, search, manipulation)
- String algorithms and pattern matching
- Palindrome problems and solutions
- Anagram detection and grouping
- Subsequence and substring problems
- String transformation algorithms
- Advanced pattern matching (KMP, regex)
- Common string patterns and optimization techniques

**Key Classes**:
- `StringOperations`: Basic string operations and manipulations
- `StringAlgorithms`: Advanced algorithms and classic problems
- `StringPatterns`: Common problem-solving patterns

**Featured Algorithms**:
- Palindrome detection and longest palindromic substring
- Anagram problems (detection, finding, grouping)
- String search (naive and KMP algorithm)
- Edit distance (Levenshtein distance)
- Minimum window substring
- Longest common subsequence
- Pattern matching and word patterns
- String reversal and manipulation

**Featured Problems**:
- Two Sum, Three Sum (adapted for strings)
- Palindrome variations
- Anagram detection and grouping
- Edit distance calculation
- Minimum window substring
- Pattern matching with wildcards
- String compression and decompression
- Substring search optimization

**Sample Usage**:
```python
from DSA.strings import StringOperations, StringAlgorithms

ops = StringOperations()
algos = StringAlgorithms()

# Basic operations
text = "Hello World"
ops.reverse_string(text)  # "dlroW olleH"
ops.find_all_occurrences(text, 'l')  # [2, 3, 9]

# Classic problems
algos.is_palindrome("A man a plan a canal Panama")  # True
algos.are_anagrams("listen", "silent")  # True
algos.edit_distance("kitten", "sitting")  # 3
algos.min_window_substring("ADOBECODEBANC", "ABC")  # "BANC"
```

### Linked Lists

**File**: `DSA/linked_lists.py`

**What it covers**:
- Singly Linked Lists with comprehensive operations
- Doubly Linked Lists with bidirectional traversal
- Circular Linked Lists with special properties
- Advanced linked list algorithms and problem-solving
- Memory-efficient implementations
- Classic linked list problems and patterns

**Key Classes**:
- `SinglyLinkedList`: Standard linked list implementation
- `DoublyLinkedList`: Bidirectional linked list with head and tail pointers
- `CircularLinkedList`: Circular implementation for special use cases
- `LinkedListAlgorithms`: Advanced algorithms and classic problems

**Featured Operations**:
- Insertion (beginning, end, specific position)
- Deletion (by value, by position, from ends)
- Search and traversal operations
- Reversal and manipulation algorithms
- Memory-efficient bidirectional operations

**Featured Algorithms**:
- Cycle detection (Floyd's algorithm)
- Finding cycle start position
- Middle node detection using two pointers
- Nth node from end calculation
- Merging sorted linked lists
- Reversing in groups of k
- Adding numbers represented as linked lists
- Duplicate removal from sorted lists

**Featured Problems**:
- Two pointer techniques for optimization
- Cycle detection and analysis
- List merging and splitting
- Position-based operations
- Memory-efficient manipulations
- Classic interview problems

**Sample Usage**:
```python
from DSA.linked_lists import SinglyLinkedList, DoublyLinkedList, LinkedListAlgorithms

# Singly linked list
sll = SinglyLinkedList()
sll.append(1)
sll.append(2)
sll.prepend(0)
print(sll)  # 0 -> 1 -> 2 -> None

# Doubly linked list
dll = DoublyLinkedList()
dll.append('A')
dll.append('B')
dll.prepend('Z')
print(dll.to_list())  # ['Z', 'A', 'B']
print(dll.to_list_reverse())  # ['B', 'A', 'Z']

# Advanced algorithms
algos = LinkedListAlgorithms()
middle = algos.find_middle(head_node)
has_cycle = algos.detect_cycle(head_node)
```

## ✨ Features

### 🎯 **Educational Focus**
- Clear, well-commented code
- Step-by-step explanations
- Multiple examples for each concept
- Real-world applications

### 📊 **Performance Analysis**
- Time and space complexity for every function
- Performance comparisons between algorithms
- Benchmarking tools with timing measurements
- Optimization techniques and trade-offs

### 🔧 **Problem-Solving Patterns**
- Two Pointers technique
- Sliding Window pattern
- Fast and Slow Pointers
- Prefix Sum technique
- Backtracking strategies
- Dynamic Programming with memoization

### 💼 **Interview Preparation**
- Common coding interview problems
- Multiple solution approaches
- Optimization from brute force to optimal
- Pattern recognition skills

### 🏗️ **Clean Code Practices**
- Professional formatting (no emojis in code)
- Consistent naming conventions
- Comprehensive docstrings
- Type hints for better code clarity

## 📈 Performance Analysis

Each module includes performance analysis and comparisons:

### Recursion Performance Example:
```
Fibonacci(30) comparison:
Normal recursion: 832040 (Time: 0.0865s)
Memoized recursion: 832040 (Time: 0.0000s)
Speedup: 22669.50x
```

### Sorting Performance Example:
```
Sorting 1000 elements:
Bubble Sort: 0.1234 seconds
Selection Sort: 0.0892 seconds
Insertion Sort: 0.0445 seconds
Quick Sort: 0.0012 seconds
Merge Sort: 0.0015 seconds
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-algorithm`
3. **Add your implementation** with proper documentation
4. **Include tests and examples**
5. **Follow the existing code style** (no emojis in code)
6. **Submit a pull request**

### Contribution Guidelines:
- Add comprehensive docstrings with time/space complexity
- Include practical examples and test cases
- Follow the existing module structure
- Update README if adding new modules

## 👨‍💻 Author

**Nit17**
- GitHub: [@Nit17](https://github.com/Nit17)
- Email: nithinbm17@gmail.com

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🎓 Learning Path

**Recommended study order**:

1. **Start with**: `time_space_complexity.py` - Understand the fundamentals
2. **Then move to**: `recursion.py` - Master recursive thinking
3. **Continue with**: `arrays_lists.py` - Apply concepts to solve real problems
4. **Finally**: `strings.py` - Master string manipulation and pattern matching

Each module builds upon previous concepts, so following this order will give you the best learning experience.

## 🚀 Quick Start Example

```python
# Clone and run in 3 commands!
git clone https://github.com/Nit17/DSA---Python.git
cd DSA-Python
python DSA/recursion.py

# Start coding immediately
from DSA.arrays_lists import ArrayAlgorithms
algos = ArrayAlgorithms()
result = algos.two_sum([2, 7, 11, 15], 9)
print(f"Two sum result: {result}")  # [0, 1]
```

---

**Happy Coding!** 🚀 