"""
STACKS - Implementations, Algorithms, and Patterns
=================================================

A stack is a LIFO (Last-In-First-Out) data structure that supports two primary
operations: push (add to top) and pop (remove from top).

Common Use Cases:
- Expression parsing (infix -> postfix, expression evaluation)
- Parentheses/Bracket validation
- Backtracking (undo operations)
- Monotonic stack patterns for next-greater, temperatures, histograms
- Path simplification and state tracking

Time Complexities:
- Push: O(1)
- Pop: O(1)
- Peek/Top: O(1)
- Search: O(n) (need to traverse)
"""

from __future__ import annotations
from typing import Any, List, Optional

# ==================== ARRAY-BASED STACK ====================

class ArrayStack:
    """
    Array (list)-backed Stack implementation.

    Methods: push, pop, peek, is_empty, size, clear
    """

    def __init__(self) -> None:
        self._data: List[Any] = []

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return f"Stack(top -> bottom): {list(reversed(self._data))}"

    def is_empty(self) -> bool:
        """Check if the stack is empty. O(1)."""
        return len(self._data) == 0

    def push(self, item: Any) -> None:
        """Push item onto the stack. O(1)."""
        self._data.append(item)

    def pop(self) -> Optional[Any]:
        """Pop the top item from the stack. Returns None if empty. O(1)."""
        if self.is_empty():
            return None
        return self._data.pop()

    def peek(self) -> Optional[Any]:
        """Return the top item without removing it. Returns None if empty. O(1)."""
        if self.is_empty():
            return None
        return self._data[-1]

    def size(self) -> int:
        """Return the current size. O(1)."""
        return len(self._data)

    def clear(self) -> None:
        """Remove all items. O(n) due to underlying list clear, but amortized constant per item."""
        self._data.clear()

# ==================== LINKED-LIST-BASED STACK ====================

class _Node:
    def __init__(self, data: Any, next: Optional['_Node'] = None) -> None:
        self.data = data
        self.next = next

class LinkedListStack:
    """
    Linked-list-backed Stack implementation (singly linked).

    Methods: push, pop, peek, is_empty, size, clear

    Space: O(n) nodes; Push/Pop: O(1)
    """

    def __init__(self) -> None:
        self._head: Optional[_Node] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def __str__(self) -> str:
        items: List[str] = []
        cur = self._head
        while cur:
            items.append(str(cur.data))
            cur = cur.next
        return f"Stack(top -> bottom): {items}"

    def is_empty(self) -> bool:
        return self._head is None

    def push(self, item: Any) -> None:
        self._head = _Node(item, self._head)
        self._size += 1

    def pop(self) -> Optional[Any]:
        if self._head is None:
            return None
        val = self._head.data
        self._head = self._head.next
        self._size -= 1
        return val

    def peek(self) -> Optional[Any]:
        return self._head.data if self._head else None

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = None
        self._size = 0

# ==================== MIN STACK (O(1) get_min) ====================

class MinStack:
    """
    Stack supporting get_min in O(1) time using an auxiliary stack.

    push/pop/top/get_min are O(1).
    """

    def __init__(self) -> None:
        self._stack: List[Any] = []
        self._mins: List[Any] = []

    def is_empty(self) -> bool:
        return not self._stack

    def push(self, x: Any) -> None:
        self._stack.append(x)
        if not self._mins or x <= self._mins[-1]:
            self._mins.append(x)

    def pop(self) -> Optional[Any]:
        if not self._stack:
            return None
        x = self._stack.pop()
        if self._mins and x == self._mins[-1]:
            self._mins.pop()
        return x

    def top(self) -> Optional[Any]:
        return self._stack[-1] if self._stack else None

    def get_min(self) -> Optional[Any]:
        return self._mins[-1] if self._mins else None

# ==================== STACK ALGORITHMS ====================

class StackAlgorithms:
    """Common stack-based algorithms and monotonic stack patterns."""

    # ----- Parentheses / Brackets Validation -----
    def is_valid_parentheses(self, s: str) -> bool:
        """
        Validate balanced parentheses/brackets.
        Supports (), {}, [].
        Time: O(n), Space: O(n)
        """
        stack: List[str] = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in s:
            if ch in '([{':
                stack.append(ch)
            elif ch in ')]}':
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
        return not stack

    # ----- Infix to Postfix (Shunting Yard) -----
    def infix_to_postfix(self, expression: str) -> List[str]:
        """
        Convert arithmetic infix expression to postfix (RPN).
        Supports + - * / ^ and parentheses. Tokens are space-separated in output.
        Time: O(n), Space: O(n)
        """
        def precedence(op: str) -> int:
            if op == '^':
                return 3
            if op in ('*', '/'):
                return 2
            if op in ('+', '-'):
                return 1
            return 0

        def is_right_associative(op: str) -> bool:
            return op == '^'

        output: List[str] = []
        ops: List[str] = []
        tokens: List[str] = []

        # Tokenize: support multi-digit numbers and variables; split by space if provided
        i = 0
        expr = expression.replace(' ', '')
        while i < len(expr):
            ch = expr[i]
            if ch.isdigit():
                j = i
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                tokens.append(expr[i:j])
                i = j
            elif ch.isalpha():
                j = i
                while j < len(expr) and expr[j].isalnum():
                    j += 1
                tokens.append(expr[i:j])
                i = j
            else:
                tokens.append(ch)
                i += 1

        for t in tokens:
            if t.isalnum():
                output.append(t)
            elif t == '(':
                ops.append(t)
            elif t == ')':
                while ops and ops[-1] != '(':  # pop until left paren
                    output.append(ops.pop())
                if ops and ops[-1] == '(':
                    ops.pop()
            else:  # operator
                while (
                    ops and ops[-1] != '('
                    and (precedence(ops[-1]) > precedence(t)
                         or (precedence(ops[-1]) == precedence(t) and not is_right_associative(t)))
                ):
                    output.append(ops.pop())
                ops.append(t)

        while ops:
            output.append(ops.pop())
        return output

    # ----- Evaluate Postfix -----
    def evaluate_postfix(self, tokens: List[str]) -> int:
        """
        Evaluate a postfix (RPN) expression where tokens are numbers/operators.
        Time: O(n), Space: O(n)
        """
        stack: List[int] = []
        for t in tokens:
            if t.lstrip('-').isdigit():
                stack.append(int(t))
            else:
                b = stack.pop()
                a = stack.pop()
                if t == '+':
                    stack.append(a + b)
                elif t == '-':
                    stack.append(a - b)
                elif t == '*':
                    stack.append(a * b)
                elif t == '/':
                    # integer division like many problems; truncate toward zero
                    stack.append(int(a / b))
                elif t == '^':
                    stack.append(a ** b)
                else:
                    raise ValueError(f"Unsupported operator: {t}")
        return stack[-1] if stack else 0

    # ----- Next Greater Element (Monotonic Stack) -----
    def next_greater_elements(self, nums: List[int]) -> List[int]:
        """
        For each element, find the next greater element to its right; -1 if none.
        Time: O(n), Space: O(n)
        """
        res = [-1] * len(nums)
        stack: List[int] = []  # indices, maintain decreasing stack
        for i, val in enumerate(nums):
            while stack and nums[stack[-1]] < val:
                idx = stack.pop()
                res[idx] = val
            stack.append(i)
        return res

    # ----- Daily Temperatures (Monotonic Decreasing Stack) -----
    def daily_temperatures(self, temps: List[int]) -> List[int]:
        """
        For each day, return how many days to wait until a warmer temperature. 0 if none.
        Time: O(n), Space: O(n)
        """
        res = [0] * len(temps)
        stack: List[int] = []  # indices of decreasing temperatures
        for i, t in enumerate(temps):
            while stack and temps[stack[-1]] < t:
                j = stack.pop()
                res[j] = i - j
            stack.append(i)
        return res

    # ----- Largest Rectangle in Histogram -----
    def largest_rectangle_histogram(self, heights: List[int]) -> int:
        """
        Compute largest rectangle area in histogram using a monotonic stack.
        Time: O(n), Space: O(n)
        """
        stack: List[int] = []  # store indices
        max_area = 0
        heights.append(0)  # sentinel to flush stack
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        heights.pop()  # restore
        return max_area

    # ----- Trapping Rain Water -----
    def trapping_rain_water(self, height: List[int]) -> int:
        """
        Compute trapped water using a stack.
        Time: O(n), Space: O(n)
        """
        stack: List[int] = []
        water = 0
        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                bottom = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                width = i - left - 1
                bounded = min(height[left], h) - height[bottom]
                if bounded > 0:
                    water += bounded * width
            stack.append(i)
        return water

    # ----- Simplify Unix Path -----
    def simplify_path(self, path: str) -> str:
        """
        Simplify a Unix-style path using a stack.
        Time: O(n), Space: O(n)
        """
        parts = path.split('/')
        stack: List[str] = []
        for p in parts:
            if p == '' or p == '.':
                continue
            if p == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(p)
        return '/' + '/'.join(stack)

# ==================== DEMONSTRATION ====================

def demonstrate_stacks() -> None:
    print("\nSTACK DEMONSTRATIONS")
    print("=" * 50)

    # ArrayStack basics
    print("\nArrayStack:")
    a = ArrayStack()
    for x in [1, 2, 3]:
        a.push(x)
    print(f"Top: {a.peek()}, Size: {a.size()}")
    print(f"Pop: {a.pop()}, New Top: {a.peek()}")

    # LinkedListStack basics
    print("\nLinkedListStack:")
    lls = LinkedListStack()
    for ch in ['A', 'B', 'C']:
        lls.push(ch)
    print(f"Top: {lls.peek()}, Size: {lls.size()}")
    print(f"Pop: {lls.pop()}, New Top: {lls.peek()}")

    # MinStack
    print("\nMinStack:")
    ms = MinStack()
    for v in [3, 5, 2, 2, 4]:
        ms.push(v)
    print(f"Current Min: {ms.get_min()}")
    ms.pop()  # 4
    ms.pop()  # 2
    print(f"After pops, Min: {ms.get_min()}")  # should be 2
    ms.pop()  # 2
    print(f"After pop, Min: {ms.get_min()}")  # should be 3 or 5? Actually stack: [3,5] => 3

    # Algorithms
    algo = StackAlgorithms()

    print("\nParentheses Validation:")
    s = "{[()]}()"
    print(f"{s}: {algo.is_valid_parentheses(s)}")

    print("\nInfix to Postfix and Evaluation:")
    infix = "3+(4*5-6)/(1+2)"
    postfix = algo.infix_to_postfix(infix)
    print(f"Infix: {infix}")
    print(f"Postfix: {' '.join(postfix)}")
    print(f"Evaluated: {algo.evaluate_postfix(postfix)}")

    print("\nNext Greater Elements:")
    nums = [2, 1, 2, 4, 3]
    print(f"{nums} -> {algo.next_greater_elements(nums)}")

    print("\nDaily Temperatures:")
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    print(f"{temps} -> {algo.daily_temperatures(temps)}")

    print("\nLargest Rectangle in Histogram:")
    heights = [2, 1, 5, 6, 2, 3]
    print(f"{heights} -> {algo.largest_rectangle_histogram(heights)}")

    print("\nTrapping Rain Water:")
    elevation = [0,1,0,2,1,0,1,3,2,1,2,1]
    print(f"Water trapped: {algo.trapping_rain_water(elevation)}")

    print("\nSimplify Path:")
    path = "/a//b////c/d//././/.."
    print(f"{path} -> {algo.simplify_path(path)}")


def stack_tips_and_tricks() -> None:
    print("\nSTACK TIPS AND TRICKS")
    print("=" * 50)
    tips = [
        "Use a stack for matching parentheses and expression parsing",
        "Monotonic stacks solve next/previous greater/smaller problems in O(n)",
        "For getMin in O(1), keep an auxiliary min stack",
        "Shunting-yard converts infix to postfix cleanly",
        "Use stacks to backtrack or undo operations",
        "When evaluating expressions, ensure correct operator associativity",
        "Histogram and rainwater problems are classic stack use-cases",
        "Prefer list-based stacks for Python unless node-level control needed",
        "Keep tokens clean (multi-digit numbers) when parsing expressions",
        "Practice dry runs with paper to avoid pointer/index mistakes",
    ]
    for t in tips:
        print(f"- {t}")


if __name__ == "__main__":
    demonstrate_stacks()
    stack_tips_and_tricks()
