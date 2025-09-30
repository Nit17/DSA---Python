"""Arrays / Lists – Theory, Internals & Patterns
===============================================
In lower-level languages an array is a fixed-size block of contiguous memory
holding same-typed elements. Python's built-in ``list`` is a *dynamic array* –
it over-allocates capacity to allow amortized O(1) append operations while
still providing O(1) index access.

1. Contiguous Memory & Cache Locality
-------------------------------------
Sequential storage improves spatial locality: iterating forward uses CPU cache
efficiently. Linked lists sacrifice this locality (pointer chasing) while
offering different insertion characteristics.

2. Python List Implementation (High Level)
-----------------------------------------
- Under the hood a list stores pointers (PyObject*) to objects, not raw values.
- Over-allocation growth strategy (approx 9/8 * old + constant) reduces the
    frequency of costly resize (reallocate + copy) operations.
- Thus append is amortized O(1); occasional resize is O(n) but rare.

3. Operation Complexities (Average / Worst)
------------------------------------------
Index access / update: O(1)
Append (amortized):     O(1)    (worst: O(n) on resize)
Pop end:                O(1)
Insert/delete front:    O(n)    (shift all following elements)
Membership (x in list): O(n)
Slice copy a[:k]:       O(k)
Extend by iterable:     O(m)    (m = length of iterable)

4. Common Use Patterns
----------------------
- Unordered bag with occasional scans (list)
- Need frequent min/max extraction? Prefer heap or balanced tree
- Need uniqueness & O(1) membership? Use set / dict keys
- Need sorted iteration + insert/remove? Consider bisect + list (accept O(n)) or a tree structure

5. In-Place Transform Techniques
--------------------------------
Two pointers, fast/slow pointer for compaction, reversal by symmetric swaps,
partitioning (quicksort-style) for selection problems.

6. Stability in Sorting
-----------------------
Python's ``sorted`` / list.sort() implement Timsort (stable, adaptive). Stability
lets you sort by multiple keys sequentially (secondary ordering preserved).
Educational implementations here (bubble/selection/insertion/quick/merge) trade
practical performance for clarity.

7. Memory Footprint Caveat
--------------------------
``list`` holds pointers. A list of 1 million small integers consumes more than
8MB because each int is a PyObject with overhead. For dense numeric arrays
prefer ``array`` module, ``numpy.ndarray`` or ``memoryview`` for efficiency.

8. Avoiding Quadratic Pitfalls
------------------------------
- Repeated ``insert(0, x)`` or ``pop(0)`` is O(n) each → use ``collections.deque``.
- Building a string with ``+=`` in a loop is O(n^2) in total → collect pieces in
    a list then ``''.join`` at end.

9. Algorithmic Patterns Highlighted Below
-----------------------------------------
Two Pointers, Sliding Window, Fast/Slow (Floyd), Prefix Sum, Sort + Sweep,
Hashing for complement lookups (Two Sum), Dynamic Programming (Kadane as DP in disguise).

10. Choosing the Right Structure
--------------------------------
- Need random access + amortized append: list
- Need FIFO queue O(1) both ends: deque
- Need frequent membership tests: set / dict
- Need stable ordering + priority extractions: heap (``heapq``) + list wrapper
- Need many front insertions/removals: deque or linked list

11. Copy vs View
----------------
Slicing creates copies: ``b = a[:]`` duplicates references (shallow copy). Large
data slices cost O(k) time & memory. For constant-time views you need other
structures (iterators, memoryview for bytes/bytearray, NumPy slices for arrays).

12. Testing & Bench Strategy
----------------------------
- Micro-bench time-critical loops (``timeit`` module)
- Validate algorithmic complexity by doubling input sizes and observing time growth

This file couples theoretical narrative with practical implementations for a
holistic understanding of array-based problem solving.
"""

import time
import random
from typing import List, Optional, Tuple, Union

class ArrayOperations:
    """
    Fundamental array operations and manipulations
    """
    
    def __init__(self):
        print("Array Operations Initialized")
        print("=" * 50)
    
    # ==================== BASIC OPERATIONS ====================
    
    def create_array(self, size: int, default_value=0) -> List[int]:
        """
        Create an array of given size with default values
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return [default_value] * size
    
    def access_element(self, arr: List[int], index: int) -> Optional[int]:
        """
        Access element at given index with bounds checking
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if 0 <= index < len(arr):
            return arr[index]
        return None
    
    def insert_element(self, arr: List[int], index: int, value: int) -> List[int]:
        """
        Insert element at given index
        
        Time Complexity: O(n) - worst case when inserting at beginning
        Space Complexity: O(1)
        """
        if 0 <= index <= len(arr):
            arr.insert(index, value)
        return arr
    
    def delete_element(self, arr: List[int], index: int) -> List[int]:
        """
        Delete element at given index
        
        Time Complexity: O(n) - worst case when deleting from beginning
        Space Complexity: O(1)
        """
        if 0 <= index < len(arr):
            arr.pop(index)
        return arr
    
    def update_element(self, arr: List[int], index: int, value: int) -> List[int]:
        """
        Update element at given index
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if 0 <= index < len(arr):
            arr[index] = value
        return arr
    
    # ==================== TRAVERSAL OPERATIONS ====================
    
    def linear_search(self, arr: List[int], target: int) -> int:
        """
        Find index of target element using linear search
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    def binary_search(self, arr: List[int], target: int) -> int:
        """
        Find index of target element using binary search (requires sorted array)
        
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    def find_min_max(self, arr: List[int]) -> Tuple[int, int]:
        """
        Find minimum and maximum elements
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr:
            return None, None
        
        min_val = max_val = arr[0]
        for num in arr[1:]:
            if num < min_val:
                min_val = num
            elif num > max_val:
                max_val = num
        
        return min_val, max_val
    
    def count_elements(self, arr: List[int], target: int) -> int:
        """
        Count occurrences of target element
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = 0
        for num in arr:
            if num == target:
                count += 1
        return count
    
    # ==================== ARRAY MANIPULATION ====================
    
    def reverse_array(self, arr: List[int]) -> List[int]:
        """
        Reverse array in-place
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
        
        return arr
    
    def rotate_left(self, arr: List[int], k: int) -> List[int]:
        """
        Rotate array left by k positions
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr or k == 0:
            return arr
        
        n = len(arr)
        k = k % n  # Handle k > n
        
        # Reverse entire array
        self.reverse_array(arr)
        # Reverse first n-k elements
        self.reverse_subarray(arr, 0, n - k - 1)
        # Reverse last k elements
        self.reverse_subarray(arr, n - k, n - 1)
        
        return arr
    
    def rotate_right(self, arr: List[int], k: int) -> List[int]:
        """
        Rotate array right by k positions
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr or k == 0:
            return arr
        
        n = len(arr)
        k = k % n  # Handle k > n
        
        # Reverse entire array
        self.reverse_array(arr)
        # Reverse first k elements
        self.reverse_subarray(arr, 0, k - 1)
        # Reverse remaining elements
        self.reverse_subarray(arr, k, n - 1)
        
        return arr
    
    def reverse_subarray(self, arr: List[int], start: int, end: int) -> None:
        """Helper function to reverse subarray from start to end"""
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    def remove_duplicates(self, arr: List[int]) -> List[int]:
        """
        Remove duplicates from sorted array
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(arr) <= 1:
            return arr
        
        write_index = 1
        
        for read_index in range(1, len(arr)):
            if arr[read_index] != arr[read_index - 1]:
                arr[write_index] = arr[read_index]
                write_index += 1
        
        return arr[:write_index]
    
    def merge_sorted_arrays(self, arr1: List[int], arr2: List[int]) -> List[int]:
        """
        Merge two sorted arrays
        
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        result = []
        i, j = 0, 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        # Add remaining elements
        result.extend(arr1[i:])
        result.extend(arr2[j:])
        
        return result

class ArrayAlgorithms:
    """
    Advanced array algorithms and problem-solving techniques
    """
    
    def __init__(self):
        print("Array Algorithms Initialized")
        print("=" * 50)
    
    # ==================== SORTING ALGORITHMS ====================
    
    def bubble_sort(self, arr: List[int]) -> List[int]:
        """
        Bubble sort implementation
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            if not swapped:  # Array is sorted
                break
        
        return arr
    
    def selection_sort(self, arr: List[int]) -> List[int]:
        """
        Selection sort implementation
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        return arr
    
    def insertion_sort(self, arr: List[int]) -> List[int]:
        """
        Insertion sort implementation
        
        Time Complexity: O(n²) worst case, O(n) best case
        Space Complexity: O(1)
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        return arr
    
    def quick_sort(self, arr: List[int]) -> List[int]:
        """
        Quick sort implementation
        
        Average Time Complexity: O(n log n)
        Worst Time Complexity: O(n²)
        Space Complexity: O(log n)
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """
        Merge sort implementation
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
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
    
    # ==================== CLASSIC ARRAY PROBLEMS ====================
    
    def two_sum(self, arr: List[int], target: int) -> List[int]:
        """
        Find two numbers that add up to target
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        num_map = {}
        
        for i, num in enumerate(arr):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
        
        return []
    
    def three_sum(self, arr: List[int]) -> List[List[int]]:
        """
        Find all unique triplets that sum to zero
        
        Time Complexity: O(n²)
        Space Complexity: O(1) excluding output
        """
        arr.sort()
        result = []
        
        for i in range(len(arr) - 2):
            if i > 0 and arr[i] == arr[i - 1]:
                continue
            
            left, right = i + 1, len(arr) - 1
            
            while left < right:
                total = arr[i] + arr[left] + arr[right]
                
                if total == 0:
                    result.append([arr[i], arr[left], arr[right]])
                    
                    while left < right and arr[left] == arr[left + 1]:
                        left += 1
                    while left < right and arr[right] == arr[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        
        return result
    
    def maximum_subarray_sum(self, arr: List[int]) -> int:
        """
        Find maximum sum of contiguous subarray (Kadane's Algorithm)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        max_sum = current_sum = arr[0]
        
        for num in arr[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maximum_product_subarray(self, arr: List[int]) -> int:
        """
        Find maximum product of contiguous subarray
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not arr:
            return 0
        
        max_product = min_product = result = arr[0]
        
        for num in arr[1:]:
            if num < 0:
                max_product, min_product = min_product, max_product
            
            max_product = max(num, max_product * num)
            min_product = min(num, min_product * num)
            
            result = max(result, max_product)
        
        return result
    
    def find_missing_number(self, arr: List[int], n: int) -> int:
        """
        Find missing number in array of 1 to n
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(arr)
        return expected_sum - actual_sum
    
    def find_duplicate_number(self, arr: List[int]) -> int:
        """
        Find duplicate number in array (Floyd's Cycle Detection)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        # Phase 1: Find intersection point
        slow = fast = arr[0]
        
        while True:
            slow = arr[slow]
            fast = arr[arr[fast]]
            if slow == fast:
                break
        
        # Phase 2: Find entrance to cycle
        slow = arr[0]
        while slow != fast:
            slow = arr[slow]
            fast = arr[fast]
        
        return slow
    
    def buy_sell_stock(self, prices: List[int]) -> int:
        """
        Find maximum profit from buying and selling stock once
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(prices) < 2:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for price in prices[1:]:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
        
        return max_profit
    
    def container_with_most_water(self, heights: List[int]) -> int:
        """
        Find container that can hold most water
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(heights) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            height = min(heights[left], heights[right])
            area = width * height
            max_area = max(max_area, area)
            
            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    def trapping_rainwater(self, heights: List[int]) -> int:
        """
        Calculate trapped rainwater
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(heights) < 3:
            return 0
        
        left, right = 0, len(heights) - 1
        left_max = right_max = water = 0
        
        while left < right:
            if heights[left] < heights[right]:
                if heights[left] >= left_max:
                    left_max = heights[left]
                else:
                    water += left_max - heights[left]
                left += 1
            else:
                if heights[right] >= right_max:
                    right_max = heights[right]
                else:
                    water += right_max - heights[right]
                right -= 1
        
        return water

class ArrayPatterns:
    """
    Common array patterns and techniques
    """
    
    @staticmethod
    def two_pointers_pattern():
        """
        Two Pointers Pattern:
        - Use two pointers moving towards each other or in same direction
        - Useful for sorted arrays, palindromes, pair problems
        - Examples: Two Sum (sorted), Container With Most Water
        """
        print("\nTWO POINTERS PATTERN:")
        print("- Use two pointers: start and end, or slow and fast")
        print("- Move pointers based on conditions")
        print("- Reduces time complexity from O(n²) to O(n)")
        print("- Common in: pair sum, palindrome, cycle detection")
    
    @staticmethod
    def sliding_window_pattern():
        """
        Sliding Window Pattern:
        - Maintain a window of elements and slide it through array
        - Useful for subarray problems with constraints
        - Examples: Maximum sum subarray of size k, Longest substring
        """
        print("\nSLIDING WINDOW PATTERN:")
        print("- Maintain window with start and end pointers")
        print("- Expand/contract window based on conditions")
        print("- Avoids nested loops for subarray problems")
        print("- Common in: subarray sum, substring problems")
    
    @staticmethod
    def fast_slow_pointers():
        """
        Fast and Slow Pointers:
        - Two pointers moving at different speeds
        - Useful for cycle detection, finding middle element
        - Examples: Linked list cycle, Finding duplicate number
        """
        print("\nFAST AND SLOW POINTERS:")
        print("- Slow pointer moves one step, fast moves two steps")
        print("- Detect cycles when pointers meet")
        print("- Find middle element when fast reaches end")
        print("- Common in: cycle detection, middle element")
    
    @staticmethod
    def prefix_sum_pattern():
        """
        Prefix Sum Pattern:
        - Precompute cumulative sums for range queries
        - Useful for subarray sum problems
        - Examples: Range sum queries, Subarray sum equals k
        """
        print("\nPREFIX SUM PATTERN:")
        print("- Precompute cumulative sums: prefix[i] = sum(arr[0:i+1])")
        print("- Range sum = prefix[j] - prefix[i-1]")
        print("- Reduces range queries to O(1)")
        print("- Common in: range sum, subarray problems")
    
    @staticmethod
    def show_all_patterns():
        """Display all array patterns"""
        print("\nCOMMON ARRAY PATTERNS:")
        print("=" * 40)
        ArrayPatterns.two_pointers_pattern()
        ArrayPatterns.sliding_window_pattern()
        ArrayPatterns.fast_slow_pointers()
        ArrayPatterns.prefix_sum_pattern()

def demonstrate_arrays():
    """
    Demonstrate various array operations and algorithms
    """
    print("\nARRAY DEMONSTRATIONS")
    print("=" * 50)
    
    array_ops = ArrayOperations()
    array_algos = ArrayAlgorithms()
    
    # Basic operations
    print("\nBASIC ARRAY OPERATIONS:")
    arr = [1, 3, 5, 7, 9]
    print(f"Original array: {arr}")
    print(f"Access element at index 2: {array_ops.access_element(arr, 2)}")
    
    arr_copy = arr.copy()
    array_ops.insert_element(arr_copy, 2, 4)
    print(f"After inserting 4 at index 2: {arr_copy}")
    
    arr_copy = arr.copy()
    array_ops.delete_element(arr_copy, 2)
    print(f"After deleting element at index 2: {arr_copy}")
    
    # Search operations
    print("\nSEARCH OPERATIONS:")
    print(f"Linear search for 5: index {array_ops.linear_search(arr, 5)}")
    print(f"Binary search for 5: index {array_ops.binary_search(arr, 5)}")
    
    min_val, max_val = array_ops.find_min_max(arr)
    print(f"Min: {min_val}, Max: {max_val}")
    
    # Array manipulation
    print("\nARRAY MANIPULATION:")
    arr_copy = [1, 2, 3, 4, 5]
    print(f"Original: {arr_copy}")
    
    reversed_arr = arr_copy.copy()
    array_ops.reverse_array(reversed_arr)
    print(f"Reversed: {reversed_arr}")
    
    rotated_left = arr_copy.copy()
    array_ops.rotate_left(rotated_left, 2)
    print(f"Rotated left by 2: {rotated_left}")
    
    rotated_right = arr_copy.copy()
    array_ops.rotate_right(rotated_right, 2)
    print(f"Rotated right by 2: {rotated_right}")
    
    # Sorting algorithms
    print("\nSORTING ALGORITHMS:")
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    print(f"Unsorted array: {unsorted}")
    
    bubble_sorted = array_algos.bubble_sort(unsorted.copy())
    print(f"Bubble sort: {bubble_sorted}")
    
    selection_sorted = array_algos.selection_sort(unsorted.copy())
    print(f"Selection sort: {selection_sorted}")
    
    insertion_sorted = array_algos.insertion_sort(unsorted.copy())
    print(f"Insertion sort: {insertion_sorted}")
    
    quick_sorted = array_algos.quick_sort(unsorted.copy())
    print(f"Quick sort: {quick_sorted}")
    
    merge_sorted = array_algos.merge_sort(unsorted.copy())
    print(f"Merge sort: {merge_sorted}")
    
    # Classic problems
    print("\nCLASSIC ARRAY PROBLEMS:")
    
    # Two Sum
    nums = [2, 7, 11, 15]
    target = 9
    two_sum_result = array_algos.two_sum(nums, target)
    print(f"Two Sum - Array: {nums}, Target: {target}, Indices: {two_sum_result}")
    
    # Maximum Subarray
    subarray_nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = array_algos.maximum_subarray_sum(subarray_nums)
    print(f"Maximum Subarray Sum of {subarray_nums}: {max_sum}")
    
    # Stock Problem
    stock_prices = [7, 1, 5, 3, 6, 4]
    max_profit = array_algos.buy_sell_stock(stock_prices)
    print(f"Stock prices {stock_prices}, Max profit: {max_profit}")
    
    # Container with most water
    container_heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    max_water = array_algos.container_with_most_water(container_heights)
    print(f"Container heights {container_heights}, Max water: {max_water}")
    
    # Show patterns
    ArrayPatterns.show_all_patterns()
    
    # Performance comparison
    print("\nPERFORMANCE COMPARISON:")
    test_array = list(range(1000, 0, -1))  # Reverse sorted array
    
    algorithms = [
        ("Bubble Sort", array_algos.bubble_sort),
        ("Selection Sort", array_algos.selection_sort),
        ("Insertion Sort", array_algos.insertion_sort),
        ("Quick Sort", array_algos.quick_sort),
        ("Merge Sort", array_algos.merge_sort)
    ]
    
    print(f"Sorting {len(test_array)} elements:")
    for name, algorithm in algorithms:
        test_copy = test_array.copy()
        start_time = time.time()
        algorithm(test_copy)
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.4f} seconds")

def array_tips_and_tricks():
    """
    Tips for working with arrays effectively
    """
    print("\nARRAY TIPS AND TRICKS")
    print("=" * 50)
    
    tips = [
        "1. Use two pointers for sorted array problems",
        "2. Consider hash maps for O(1) lookups",
        "3. Sliding window for subarray problems",
        "4. Prefix sums for range queries",
        "5. Sort first for many optimization problems",
        "6. In-place operations to save space",
        "7. Binary search on sorted arrays",
        "8. Use Python's built-in functions when appropriate",
        "9. Consider edge cases: empty arrays, single elements",
        "10. Visualize the problem with small examples first"
    ]
    
    for tip in tips:
        print(f"- {tip}")

if __name__ == "__main__":
    demonstrate_arrays()
    array_tips_and_tricks()
    
    print("\nARRAY STUDY COMPLETE!")
    print("Master these patterns to solve most array problems!")