"""
RECURSION - Complete Guide with Examples and Problems
====================================================

Recursion is a programming technique where a function calls itself to solve a problem.
Every recursive function has two main components:
1. Base case: The condition that stops the recursion
2. Recursive case: The function calling itself with modified parameters

Time Complexity: Usually O(number of recursive calls)
Space Complexity: O(recursion depth) due to call stack
"""

import time
import sys
from typing import List, Optional

# Set recursion limit for deeper recursions
sys.setrecursionlimit(10000)

class RecursionExamples:
    """
    A comprehensive collection of recursion examples and problems
    """
    
    def __init__(self):
        print("🔄 Recursion Examples Initialized")
        print("=" * 50)
    
    # ==================== BASIC RECURSION EXAMPLES ====================
    
    def factorial(self, n: int) -> int:
        """
        Calculate factorial of n using recursion
        Base case: n <= 1
        Recursive case: n * factorial(n-1)
        
        Time Complexity: O(n)
        Space Complexity: O(n) - call stack
        """
        if n <= 1:  # Base case
            return 1
        return n * self.factorial(n - 1)  # Recursive case
    
    def fibonacci(self, n: int) -> int:
        """
        Calculate nth Fibonacci number using recursion (inefficient version)
        
        Time Complexity: O(2^n) - exponential
        Space Complexity: O(n) - maximum depth of recursion
        """
        if n <= 1:  # Base cases
            return n
        return self.fibonacci(n - 1) + self.fibonacci(n - 2)
    
    def fibonacci_optimized(self, n: int, memo: dict = None) -> int:
        """
        Calculate nth Fibonacci number using memoization
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 1:
            return n
        
        memo[n] = self.fibonacci_optimized(n - 1, memo) + self.fibonacci_optimized(n - 2, memo)
        return memo[n]
    
    def power(self, base: int, exponent: int) -> int:
        """
        Calculate base^exponent using recursion
        
        Time Complexity: O(log n) with optimization, O(n) without
        Space Complexity: O(log n)
        """
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        
        # Optimization: use divide and conquer
        if exponent % 2 == 0:
            half_power = self.power(base, exponent // 2)
            return half_power * half_power
        else:
            return base * self.power(base, exponent - 1)
    
    def sum_natural_numbers(self, n: int) -> int:
        """
        Calculate sum of first n natural numbers
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 0:
            return 0
        return n + self.sum_natural_numbers(n - 1)
    
    def gcd(self, a: int, b: int) -> int:
        """
        Calculate Greatest Common Divisor using Euclidean algorithm
        
        Time Complexity: O(log(min(a,b)))
        Space Complexity: O(log(min(a,b)))
        """
        if b == 0:
            return a
        return self.gcd(b, a % b)
    
    # ==================== STRING RECURSION ====================
    
    def reverse_string(self, s: str) -> str:
        """
        Reverse a string using recursion
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if len(s) <= 1:
            return s
        return s[-1] + self.reverse_string(s[:-1])
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if string is palindrome using recursion
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        # Remove spaces and convert to lowercase
        s = ''.join(s.split()).lower()
        
        if len(s) <= 1:
            return True
        
        if s[0] != s[-1]:
            return False
        
        return self.is_palindrome(s[1:-1])
    
    def count_occurrences(self, s: str, char: str) -> int:
        """
        Count occurrences of a character in string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not s:
            return 0
        
        count = 1 if s[0] == char else 0
        return count + self.count_occurrences(s[1:], char)
    
    # ==================== ARRAY RECURSION ====================
    
    def array_sum(self, arr: List[int], index: int = 0) -> int:
        """
        Calculate sum of array elements using recursion
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if index >= len(arr):
            return 0
        return arr[index] + self.array_sum(arr, index + 1)
    
    def find_max(self, arr: List[int], index: int = 0) -> int:
        """
        Find maximum element in array using recursion
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if index == len(arr) - 1:
            return arr[index]
        
        max_rest = self.find_max(arr, index + 1)
        return max(arr[index], max_rest)
    
    def binary_search(self, arr: List[int], target: int, left: int = 0, right: int = None) -> int:
        """
        Binary search using recursion
        
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            return self.binary_search(arr, target, left, mid - 1)
        else:
            return self.binary_search(arr, target, mid + 1, right)
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """
        Merge sort using recursion
        
        Time Complexity: O(n log n)
        Space Complexity: O(n log n)
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        """Helper function for merge sort"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def quick_sort(self, arr: List[int]) -> List[int]:
        """
        Quick sort using recursion
        
        Average Time Complexity: O(n log n)
        Worst Time Complexity: O(n²)
        Space Complexity: O(log n) average, O(n) worst
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    # ==================== TREE-LIKE RECURSION ====================
    
    def generate_permutations(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all permutations of a list
        
        Time Complexity: O(n!)
        Space Complexity: O(n!)
        """
        if len(nums) <= 1:
            return [nums]
        
        result = []
        for i in range(len(nums)):
            element = nums[i]
            remaining = nums[:i] + nums[i+1:]
            
            for perm in self.generate_permutations(remaining):
                result.append([element] + perm)
        
        return result
    
    def generate_subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all subsets (power set) of a list
        
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        if not nums:
            return [[]]
        
        first = nums[0]
        rest_subsets = self.generate_subsets(nums[1:])
        
        # Include first element in half of the subsets
        with_first = [[first] + subset for subset in rest_subsets]
        
        return rest_subsets + with_first
    
    def tower_of_hanoi(self, n: int, source: str, destination: str, auxiliary: str) -> List[str]:
        """
        Solve Tower of Hanoi problem
        
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        moves = []
        
        def hanoi_helper(n: int, src: str, dest: str, aux: str):
            if n == 1:
                moves.append(f"Move disk from {src} to {dest}")
                return
            
            # Move n-1 disks from source to auxiliary
            hanoi_helper(n - 1, src, aux, dest)
            
            # Move the largest disk from source to destination
            moves.append(f"Move disk from {src} to {dest}")
            
            # Move n-1 disks from auxiliary to destination
            hanoi_helper(n - 1, aux, dest, src)
        
        hanoi_helper(n, source, destination, auxiliary)
        return moves
    
    # ==================== ADVANCED RECURSION PROBLEMS ====================
    
    def n_queens(self, n: int) -> List[List[str]]:
        """
        Solve N-Queens problem using backtracking
        
        Time Complexity: O(N!)
        Space Complexity: O(N²)
        """
        def is_safe(board: List[List[str]], row: int, col: int) -> bool:
            # Check column
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # Check diagonal (top-left to bottom-right)
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            # Check diagonal (top-right to bottom-left)
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True
        
        def solve(board: List[List[str]], row: int) -> bool:
            if row == n:
                solutions.append([''.join(row) for row in board])
                return True
            
            found = False
            for col in range(n):
                if is_safe(board, row, col):
                    board[row][col] = 'Q'
                    found = solve(board, row + 1) or found
                    board[row][col] = '.'  # Backtrack
            
            return found
        
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        solve(board, 0)
        return solutions
    
    def solve_sudoku(self, board: List[List[str]]) -> bool:
        """
        Solve Sudoku puzzle using backtracking
        
        Time Complexity: O(9^(n*n)) where n=9
        Space Complexity: O(n*n)
        """
        def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check 3x3 box
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def solve() -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if is_valid(board, i, j, num):
                                board[i][j] = num
                                if solve():
                                    return True
                                board[i][j] = '.'  # Backtrack
                        return False
            return True
        
        return solve()

class RecursionPatterns:
    """
    Common recursion patterns and techniques
    """
    
    @staticmethod
    def tail_recursive_factorial(n, accumulator=1):
        """
        Tail Recursion Pattern:
        - Recursive call is the last operation
        - Can be optimized to iteration
        - Examples: factorial with accumulator
        
        Time Complexity: O(n)
        Space Complexity: O(n) - but can be optimized to O(1) by compiler
        """
        if n <= 1:
            return accumulator
        return RecursionPatterns.tail_recursive_factorial(n - 1, n * accumulator)
    
    @staticmethod
    def is_even(n):
        """
        Mutual Recursion Pattern:
        - Two or more functions call each other
        - Examples: even/odd checker, parsing
        """
        if n == 0:
            return True
        return RecursionPatterns.is_odd(n - 1)
    
    @staticmethod
    def is_odd(n):
        """Helper function for mutual recursion example"""
        if n == 0:
            return False
        return RecursionPatterns.is_even(n - 1)
    
    @staticmethod
    def show_patterns():
        """
        Display common recursion patterns with examples
        """
        print("\n📐 COMMON RECURSION PATTERNS:")
        print("=" * 40)
        
        print("\n1. LINEAR RECURSION:")
        print("   - Function calls itself once")
        print("   - Reduces problem size by constant amount")
        print("   - Pattern: f(n) = operation(n, f(n-1))")
        print("   - Examples: factorial, sum, reverse string")
        
        print("\n2. BINARY RECURSION:")
        print("   - Function calls itself twice")
        print("   - Divides problem into two subproblems")
        print("   - Pattern: f(n) = combine(f(left), f(right))")
        print("   - Examples: Fibonacci, tree traversal, merge sort")
        
        print("\n3. TAIL RECURSION:")
        print("   - Recursive call is the last operation")
        print("   - Can be optimized to iteration")
        print("   - Uses accumulator parameters")
        print(f"   - Example: factorial(5) = {RecursionPatterns.tail_recursive_factorial(5)}")
        
        print("\n4. MUTUAL RECURSION:")
        print("   - Two or more functions call each other")
        print("   - Useful for state machines and parsing")
        print(f"   - Example: is_even(4) = {RecursionPatterns.is_even(4)}")
        print(f"   - Example: is_odd(4) = {RecursionPatterns.is_odd(4)}")
        
        print("\n5. MULTIPLE RECURSION:")
        print("   - Function makes multiple recursive calls")
        print("   - Often exponential time complexity")
        print("   - Examples: tree traversal, backtracking")
        
        print("\n6. INDIRECT RECURSION:")
        print("   - Function calls another function that eventually calls back")
        print("   - More complex call chain")
        print("   - Examples: complex parsing, state machines")

def demonstrate_recursion():
    """
    Demonstrate various recursion examples with timing
    """
    print("\n🔄 RECURSION DEMONSTRATIONS")
    print("=" * 50)
    
    recursion = RecursionExamples()
    patterns = RecursionPatterns()
    
    # Basic examples
    print("\n📋 BASIC RECURSION EXAMPLES:")
    print(f"Factorial(5): {recursion.factorial(5)}")
    print(f"Fibonacci(10): {recursion.fibonacci(10)}")
    print(f"Fibonacci Optimized(10): {recursion.fibonacci_optimized(10)}")
    print(f"Power(2, 10): {recursion.power(2, 10)}")
    print(f"Sum of 1 to 10: {recursion.sum_natural_numbers(10)}")
    print(f"GCD(48, 18): {recursion.gcd(48, 18)}")
    
    # String examples
    print("\n📝 STRING RECURSION:")
    print(f"Reverse 'hello': '{recursion.reverse_string('hello')}'")
    print(f"Is 'racecar' palindrome: {recursion.is_palindrome('racecar')}")
    print(f"Count 'l' in 'hello': {recursion.count_occurrences('hello', 'l')}")
    
    # Array examples
    print("\n🔢 ARRAY RECURSION:")
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Array: {arr}")
    print(f"Sum: {recursion.array_sum(arr)}")
    print(f"Max: {recursion.find_max(arr)}")
    
    sorted_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"Binary search for 5 in {sorted_arr}: {recursion.binary_search(sorted_arr, 5)}")
    
    print(f"Merge sort {arr}: {recursion.merge_sort(arr)}")
    print(f"Quick sort {arr}: {recursion.quick_sort(arr)}")
    
    # Complex examples
    print("\n🌳 COMPLEX RECURSION:")
    small_list = [1, 2, 3]
    print(f"Permutations of {small_list}:")
    perms = recursion.generate_permutations(small_list)
    for i, perm in enumerate(perms):
        print(f"  {i+1}: {perm}")
    
    print(f"\nSubsets of {small_list}:")
    subsets = recursion.generate_subsets(small_list)
    for i, subset in enumerate(subsets):
        print(f"  {i+1}: {subset}")
    
    print(f"\nTower of Hanoi (3 disks):")
    moves = recursion.tower_of_hanoi(3, 'A', 'C', 'B')
    for i, move in enumerate(moves):
        print(f"  {i+1}: {move}")
    
    # Show recursion patterns
    patterns.show_patterns()
    
    # Performance comparison
    print("\n⏱️ PERFORMANCE COMPARISON:")
    
    # Fibonacci comparison
    n = 30  # Reduced from 35 for faster execution
    print(f"\nFibonacci({n}) comparison:")
    
    start_time = time.time()
    result_normal = recursion.fibonacci(n)
    normal_time = time.time() - start_time
    
    start_time = time.time()
    result_optimized = recursion.fibonacci_optimized(n)
    optimized_time = time.time() - start_time
    
    print(f"Normal recursion: {result_normal} (Time: {normal_time:.4f}s)")
    print(f"Memoized recursion: {result_optimized} (Time: {optimized_time:.4f}s)")
    if optimized_time > 0:
        print(f"Speedup: {normal_time/optimized_time:.2f}x")

def recursion_tips_and_tricks():
    """
    Tips for writing effective recursive functions
    """
    print("\n💡 RECURSION TIPS AND TRICKS")
    print("=" * 50)
    
    tips = [
        "1. Always define a clear base case",
        "2. Ensure the problem size decreases with each recursive call",
        "3. Use memoization for overlapping subproblems",
        "4. Consider iterative solutions for tail recursion",
        "5. Be aware of stack overflow for deep recursions",
        "6. Use helper functions with accumulators when needed",
        "7. Visualize the recursion tree for complex problems",
        "8. Test with small inputs first",
        "9. Consider the trade-off between code clarity and efficiency",
        "10. Use debugger to trace recursive calls"
    ]
    
    for tip in tips:
        print(f"✅ {tip}")

if __name__ == "__main__":
    demonstrate_recursion()
    recursion_tips_and_tricks()
    
    print("\n🎯 RECURSION STUDY COMPLETE!")
    print("Practice more problems to master recursion patterns!")