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

## 📁 Repository Structure

```
DSA-Python/
├── DSA/
│   ├── time_space_complexity.py    # Complexity analysis and examples
│   ├── recursion.py                # Recursion patterns and problems
│   ├── arrays_lists.py             # Array operations and algorithms
│   ├── strings.py                  # String operations and algorithms
│   └── linked_lists.py             # Linked list implementations
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
from DSA.strings import StringOperations, StringAlgorithms
from DSA.time_space_complexity import ComplexityAnalysis

# Create instances and use methods
rec = RecursionExamples()
result = rec.factorial(5)  # Returns 120

ops = ArrayOperations()
index = ops.binary_search([1, 2, 3, 4, 5], 3)  # Returns 2
```

### Option 3: Interactive Python Session

```python
# Start Python interpreter in the repository root
python3

# Import and explore
>>> import sys
>>> sys.path.append('DSA')
>>> from recursion import *
>>> from arrays_lists import *

# Try out functions
>>> rec = RecursionExamples()
>>> rec.fibonacci(10)
55
```

## 📚 Modules

### Time and Space Complexity

**File**: `DSA/time_space_complexity.py`

**What it covers**:
- Big O notation explanations and examples
- Time complexity analysis for common operations
- Space complexity concepts
- Practical examples with performance measurements
- Best, average, and worst-case scenarios

**Key Classes**:
- `ComplexityAnalysis`: Demonstrates different complexity classes
- `ComplexityComparison`: Performance benchmarking tools

**Sample Usage**:
```python
from DSA.time_space_complexity import ComplexityAnalysis

analyzer = ComplexityAnalysis()
analyzer.demonstrate_time_complexities()
```

### Recursion

**File**: `DSA/recursion.py`

**What it covers**:
- Basic recursion concepts and patterns
- String and array recursion problems
- Advanced algorithms (N-Queens, Sudoku solver)
- Optimization techniques (memoization)
- Common recursion patterns and when to use them

**Key Classes**:
- `RecursionExamples`: 20+ recursion problems with solutions
- `RecursionPatterns`: Common patterns and techniques

**Featured Algorithms**:
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