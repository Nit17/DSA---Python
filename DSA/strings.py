"""
STRINGS - Complete Guide with Examples and Problems
==================================================

Strings are sequences of characters and one of the most fundamental data types.
In Python, strings are immutable, which affects how we manipulate them.

Key Properties:
- Immutable: Cannot be changed after creation
- Indexed: Access characters by position (0-based)
- Iterable: Can loop through characters
- Unicode support: Handle international characters
- Rich built-in methods: Many string operations available

Time Complexities:
- Access: O(1) by index
- Search: O(n) for substring
- Concatenation: O(n) creates new string
- Slicing: O(k) where k is slice length
"""


import time
import re
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict

class StringOperations:
    """
    Fundamental string operations and manipulations
    """
    
    def __init__(self):
        print("String Operations Initialized")
        print("=" * 50)
    
    # ==================== BASIC OPERATIONS ====================
    
    def string_length(self, s: str) -> int:
        """
        Get length of string
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(s)
    
    def access_character(self, s: str, index: int) -> Optional[str]:
        """
        Access character at given index with bounds checking
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if 0 <= index < len(s):
            return s[index]
        return None
    
    def substring(self, s: str, start: int, end: int = None) -> str:
        """
        Extract substring from start to end
        
        Time Complexity: O(k) where k is substring length
        Space Complexity: O(k)
        """
        if end is None:
            end = len(s)
        return s[start:end]
    
    def concatenate_strings(self, strings: List[str]) -> str:
        """
        Concatenate list of strings efficiently
        
        Time Complexity: O(n) where n is total characters
        Space Complexity: O(n)
        """
        return ''.join(strings)
    
    def repeat_string(self, s: str, times: int) -> str:
        """
        Repeat string multiple times
        
        Time Complexity: O(n * times)
        Space Complexity: O(n * times)
        """
        return s * times
    
    # ==================== SEARCH OPERATIONS ====================
    
    def linear_search(self, s: str, char: str) -> int:
        """
        Find first occurrence of character
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i, c in enumerate(s):
            if c == char:
                return i
        return -1
    
    def find_all_occurrences(self, s: str, char: str) -> List[int]:
        """
        Find all occurrences of character
        
        Time Complexity: O(n)
        Space Complexity: O(k) where k is number of matches
        """
        return [i for i, c in enumerate(s) if c == char]
    
    def substring_search_naive(self, text: str, pattern: str) -> int:
        """
        Naive substring search
        
        Time Complexity: O(n * m) where n = len(text), m = len(pattern)
        Space Complexity: O(1)
        """
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            if text[i:i + m] == pattern:
                return i
        return -1
    
    def substring_search_kmp(self, text: str, pattern: str) -> int:
        """
        KMP (Knuth-Morris-Pratt) substring search
        
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        """
        if not pattern:
            return 0
        
        # Build failure function
        failure = self._build_failure_function(pattern)
        
        i = j = 0
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == len(pattern):
                    return i - j
            elif j > 0:
                j = failure[j - 1]
            else:
                i += 1
        
        return -1
    
    def _build_failure_function(self, pattern: str) -> List[int]:
        """Build failure function for KMP algorithm"""
        failure = [0] * len(pattern)
        j = 0
        
        for i in range(1, len(pattern)):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            
            if pattern[i] == pattern[j]:
                j += 1
            
            failure[i] = j
        
        return failure
    
    # ==================== STRING MANIPULATION ====================
    
    def reverse_string(self, s: str) -> str:
        """
        Reverse string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s[::-1]
    
    def reverse_words(self, s: str) -> str:
        """
        Reverse words in string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return ' '.join(s.split()[::-1])
    
    def reverse_words_in_place(self, s: str) -> str:
        """
        Reverse each word individually but keep word order
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return ' '.join(word[::-1] for word in s.split())
    
    def capitalize_words(self, s: str) -> str:
        """
        Capitalize first letter of each word
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return ' '.join(word.capitalize() for word in s.split())
    
    def remove_spaces(self, s: str) -> str:
        """
        Remove all spaces from string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s.replace(' ', '')
    
    def compress_spaces(self, s: str) -> str:
        """
        Replace multiple spaces with single space
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return ' '.join(s.split())
    
    def to_lowercase(self, s: str) -> str:
        """
        Convert string to lowercase
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s.lower()
    
    def to_uppercase(self, s: str) -> str:
        """
        Convert string to uppercase
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s.upper()
    
    def swap_case(self, s: str) -> str:
        """
        Swap case of all characters
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s.swapcase()

class StringAlgorithms:
    """
    Advanced string algorithms and problem-solving techniques
    """
    
    def __init__(self):
        print("String Algorithms Initialized")
        print("=" * 50)
    
    # ==================== PALINDROME PROBLEMS ====================
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if string is palindrome (case-insensitive, alphanumeric only)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        # Clean string: remove non-alphanumeric and convert to lowercase
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        
        left, right = 0, len(cleaned) - 1
        
        while left < right:
            if cleaned[left] != cleaned[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    def longest_palindromic_substring(self, s: str) -> str:
        """
        Find longest palindromic substring
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        for i in range(len(s)):
            # Check for odd length palindromes
            len1 = self._expand_around_center(s, i, i)
            # Check for even length palindromes
            len2 = self._expand_around_center(s, i, i + 1)
            
            current_max = max(len1, len2)
            
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    def _expand_around_center(self, s: str, left: int, right: int) -> int:
        """Helper function for palindrome expansion"""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    def palindromic_substrings_count(self, s: str) -> int:
        """
        Count all palindromic substrings
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        count = 0
        
        for i in range(len(s)):
            # Count odd length palindromes
            count += self._count_palindromes_around_center(s, i, i)
            # Count even length palindromes
            count += self._count_palindromes_around_center(s, i, i + 1)
        
        return count
    
    def _count_palindromes_around_center(self, s: str, left: int, right: int) -> int:
        """Helper function for counting palindromes"""
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    # ==================== ANAGRAM PROBLEMS ====================
    
    def are_anagrams(self, s1: str, s2: str) -> bool:
        """
        Check if two strings are anagrams
        
        Time Complexity: O(n)
        Space Complexity: O(1) - limited character set
        """
        if len(s1) != len(s2):
            return False
        
        return Counter(s1.lower()) == Counter(s2.lower())
    
    def find_anagrams(self, s: str, p: str) -> List[int]:
        """
        Find all start indices of anagrams of p in s
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(p) > len(s):
            return []
        
        result = []
        p_count = Counter(p)
        window_count = Counter()
        
        # Initialize window
        for i in range(len(p)):
            window_count[s[i]] += 1
        
        if window_count == p_count:
            result.append(0)
        
        # Slide window
        for i in range(len(p), len(s)):
            # Add new character
            window_count[s[i]] += 1
            
            # Remove old character
            old_char = s[i - len(p)]
            window_count[old_char] -= 1
            if window_count[old_char] == 0:
                del window_count[old_char]
            
            # Check if current window is anagram
            if window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
    
    def group_anagrams(self, strs: List[str]) -> List[List[str]]:
        """
        Group strings that are anagrams of each other
        
        Time Complexity: O(n * k log k) where k is max string length
        Space Complexity: O(n * k)
        """
        anagram_groups = defaultdict(list)
        
        for s in strs:
            # Use sorted string as key
            key = ''.join(sorted(s))
            anagram_groups[key].append(s)
        
        return list(anagram_groups.values())
    
    # ==================== SUBSEQUENCE PROBLEMS ====================
    
    def is_subsequence(self, s: str, t: str) -> bool:
        """
        Check if s is subsequence of t
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        i = j = 0
        
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        
        return i == len(s)
    
    def longest_common_subsequence(self, s1: str, s2: str) -> int:
        """
        Find length of longest common subsequence
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def longest_increasing_subsequence_string(self, s: str) -> int:
        """
        Find length of longest increasing subsequence in string
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if s[j] < s[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    # ==================== PATTERN MATCHING ====================
    
    def word_pattern_match(self, pattern: str, s: str) -> bool:
        """
        Check if string follows a pattern
        
        Time Complexity: O(n)
        Space Complexity: O(k) where k is unique characters in pattern
        """
        words = s.split()
        
        if len(pattern) != len(words):
            return False
        
        char_to_word = {}
        word_to_char = {}
        
        for char, word in zip(pattern, words):
            if char in char_to_word:
                if char_to_word[char] != word:
                    return False
            else:
                char_to_word[char] = word
            
            if word in word_to_char:
                if word_to_char[word] != char:
                    return False
            else:
                word_to_char[word] = char
        
        return True
    
    def regex_match(self, s: str, pattern: str) -> bool:
        """
        Simple regex matching with . and *
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s), len(pattern)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns like a*b*c*
        for j in range(2, n + 1):
            if pattern[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pattern[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]  # Zero occurrences
                    if pattern[j - 2] == '.' or pattern[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # One or more
                elif pattern[j - 1] == '.' or pattern[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    
    # ==================== STRING TRANSFORMATION ====================
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate minimum edit distance (Levenshtein distance)
        
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        
        return dp[m][n]
    
    def min_window_substring(self, s: str, t: str) -> str:
        """
        Find minimum window substring containing all characters of t
        
        Time Complexity: O(|s| + |t|)
        Space Complexity: O(|s| + |t|)
        """
        if not s or not t:
            return ""
        
        dict_t = Counter(t)
        required = len(dict_t)
        
        left = right = 0
        formed = 0
        window_counts = {}
        
        ans = float("inf"), None, None
        
        while right < len(s):
            character = s[right]
            window_counts[character] = window_counts.get(character, 0) + 1
            
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            while left <= right and formed == required:
                character = s[left]
                
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

class StringPatterns:
    """
    Common string patterns and techniques
    """
    
    @staticmethod
    def two_pointers_pattern():
        """
        Two Pointers Pattern for Strings:
        - Use two pointers moving from ends or same direction
        - Useful for palindromes, reversing, comparing
        """
        print("\nTWO POINTERS PATTERN (STRINGS):")
        print("- Start from both ends and move towards center")
        print("- Or use slow and fast pointers")
        print("- Reduces space complexity for many problems")
        print("- Common in: palindrome check, string reversal")
    
    @staticmethod
    def sliding_window_pattern():
        """
        Sliding Window Pattern for Strings:
        - Maintain a window and slide through string
        - Useful for substring problems with constraints
        """
        print("\nSLIDING WINDOW PATTERN (STRINGS):")
        print("- Maintain window with start and end pointers")
        print("- Expand window until constraint violated")
        print("- Contract window while maintaining validity")
        print("- Common in: substring search, anagram finding")
    
    @staticmethod
    def frequency_map_pattern():
        """
        Frequency Map Pattern:
        - Count character frequencies using hash map
        - Useful for anagram and character counting problems
        """
        print("\nFREQUENCY MAP PATTERN:")
        print("- Use Counter or dictionary to count characters")
        print("- Compare frequency maps for anagram problems")
        print("- Track frequency changes in sliding window")
        print("- Common in: anagrams, character permutations")
    
    @staticmethod
    def dynamic_programming_pattern():
        """
        Dynamic Programming Pattern for Strings:
        - Build solutions for subproblems
        - Useful for subsequence and edit distance problems
        """
        print("\nDYNAMIC PROGRAMMING PATTERN:")
        print("- Build 2D table for string comparisons")
        print("- Define recurrence relations")
        print("- Optimize space using 1D arrays when possible")
        print("- Common in: LCS, edit distance, regex matching")
    
    @staticmethod
    def show_all_patterns():
        """Display all string patterns"""
        print("\nCOMMON STRING PATTERNS:")
        print("=" * 40)
        StringPatterns.two_pointers_pattern()
        StringPatterns.sliding_window_pattern()
        StringPatterns.frequency_map_pattern()
        StringPatterns.dynamic_programming_pattern()

def demonstrate_strings():
    """
    Demonstrate various string operations and algorithms
    """
    print("\nSTRING DEMONSTRATIONS")
    print("=" * 50)
    
    string_ops = StringOperations()
    string_algos = StringAlgorithms()
    
    # Basic operations
    print("\nBASIC STRING OPERATIONS:")
    text = "Hello World"
    print(f"Original string: '{text}'")
    print(f"Length: {string_ops.string_length(text)}")
    print(f"Character at index 6: '{string_ops.access_character(text, 6)}'")
    print(f"Substring (0, 5): '{string_ops.substring(text, 0, 5)}'")
    
    # String manipulation
    print("\nSTRING MANIPULATION:")
    print(f"Reversed: '{string_ops.reverse_string(text)}'")
    print(f"Reversed words: '{string_ops.reverse_words(text)}'")
    print(f"Uppercase: '{string_ops.to_uppercase(text)}'")
    print(f"Lowercase: '{string_ops.to_lowercase(text)}'")
    
    sentence = "hello world python programming"
    print(f"Capitalized words: '{string_ops.capitalize_words(sentence)}'")
    
    # Search operations
    print("\nSEARCH OPERATIONS:")
    text = "programming"
    print(f"Text: '{text}'")
    print(f"Linear search for 'g': {string_ops.linear_search(text, 'g')}")
    print(f"All occurrences of 'g': {string_ops.find_all_occurrences(text, 'g')}")
    
    text = "ababcababa"
    pattern = "ababa"
    print(f"Substring search '{pattern}' in '{text}': {string_ops.substring_search_kmp(text, pattern)}")
    
    # Palindrome problems
    print("\nPALINDROME PROBLEMS:")
    test_strings = ["racecar", "A man a plan a canal Panama", "race a car"]
    for s in test_strings:
        result = string_algos.is_palindrome(s)
        print(f"'{s}' is palindrome: {result}")
    
    text = "babad"
    longest_pal = string_algos.longest_palindromic_substring(text)
    print(f"Longest palindromic substring in '{text}': '{longest_pal}'")
    
    # Anagram problems
    print("\nANAGRAM PROBLEMS:")
    s1, s2 = "listen", "silent"
    print(f"'{s1}' and '{s2}' are anagrams: {string_algos.are_anagrams(s1, s2)}")
    
    s, p = "abab", "ab"
    anagram_indices = string_algos.find_anagrams(s, p)
    print(f"Anagram indices of '{p}' in '{s}': {anagram_indices}")
    
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    grouped = string_algos.group_anagrams(words)
    print(f"Grouped anagrams: {grouped}")
    
    # Advanced problems
    print("\nADVANCED STRING PROBLEMS:")
    
    # Edit distance
    s1, s2 = "kitten", "sitting"
    distance = string_algos.edit_distance(s1, s2)
    print(f"Edit distance between '{s1}' and '{s2}': {distance}")
    
    # Minimum window substring
    s, t = "ADOBECODEBANC", "ABC"
    min_window = string_algos.min_window_substring(s, t)
    print(f"Minimum window substring of '{t}' in '{s}': '{min_window}'")
    
    # Pattern matching
    pattern = "abba"
    s = "dog cat cat dog"
    matches = string_algos.word_pattern_match(pattern, s)
    print(f"Pattern '{pattern}' matches '{s}': {matches}")
    
    # Show patterns
    StringPatterns.show_all_patterns()

def string_tips_and_tricks():
    """
    Tips for working with strings effectively
    """
    print("\nSTRING TIPS AND TRICKS")
    print("=" * 50)
    
    tips = [
        "1. Strings are immutable in Python - operations create new strings",
        "2. Use join() for efficient string concatenation",
        "3. Two pointers technique for palindrome problems",
        "4. Sliding window for substring problems",
        "5. Counter/frequency maps for anagram problems",
        "6. KMP algorithm for efficient pattern matching",
        "7. Dynamic programming for subsequence problems",
        "8. Regular expressions for complex pattern matching",
        "9. Consider Unicode and encoding for international text",
        "10. Use built-in string methods when available"
    ]
    
    for tip in tips:
        print(f"- {tip}")

if __name__ == "__main__":
    demonstrate_strings()
    string_tips_and_tricks()
    
    print("\nSTRING STUDY COMPLETE!")
    print("Master these patterns to solve most string problems!")