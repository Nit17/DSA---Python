"""
LINKED LISTS - Complete Guide with Examples and Problems
========================================================

Linked Lists are linear data structures where elements are stored in nodes,
and each node contains data and a reference to the next node.

Types of Linked Lists:
1. Singly Linked List: Each node points to the next node
2. Doubly Linked List: Each node has pointers to both next and previous nodes
3. Circular Linked List: Last node points back to the first node

Key Properties:
- Dynamic size: Can grow/shrink during runtime
- No random access: Must traverse from head to reach elements
- Efficient insertion/deletion at known positions
- Memory efficient: Only allocates memory as needed

Time Complexities:
- Access: O(n) - must traverse from head
- Search: O(n) - linear search required
- Insertion: O(1) at head, O(n) at arbitrary position
- Deletion: O(1) at head, O(n) at arbitrary position
"""

from typing import Optional, List, Any
import time

# ==================== SINGLY LINKED LIST ====================

class SinglyNode:
    """Node for singly linked list"""
    def __init__(self, data: Any):
        self.data = data
        self.next: Optional['SinglyNode'] = None
    
    def __repr__(self):
        return f"Node({self.data})"

class SinglyLinkedList:
    """
    Singly Linked List implementation with comprehensive operations
    """
    
    def __init__(self):
        self.head: Optional[SinglyNode] = None
        self.size: int = 0
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        if not self.head:
            return "[]"
        
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        return " -> ".join(result) + " -> None"
    
    # ==================== BASIC OPERATIONS ====================
    
    def is_empty(self) -> bool:
        """
        Check if list is empty
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.head is None
    
    def prepend(self, data: Any) -> None:
        """
        Add element at the beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        new_node = SinglyNode(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def append(self, data: Any) -> None:
        """
        Add element at the end
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        new_node = SinglyNode(data)
        
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def insert_at_position(self, position: int, data: Any) -> bool:
        """
        Insert element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position > self.size:
            return False
        
        if position == 0:
            self.prepend(data)
            return True
        
        new_node = SinglyNode(data)
        current = self.head
        
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def delete_from_beginning(self) -> Optional[Any]:
        """
        Delete element from beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.head:
            return None
        
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return data
    
    def delete_from_end(self) -> Optional[Any]:
        """
        Delete element from end
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.head:
            return None
        
        if not self.head.next:
            data = self.head.data
            self.head = None
            self.size -= 1
            return data
        
        current = self.head
        while current.next.next:
            current = current.next
        
        data = current.next.data
        current.next = None
        self.size -= 1
        return data
    
    def delete_at_position(self, position: int) -> Optional[Any]:
        """
        Delete element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position >= self.size:
            return None
        
        if position == 0:
            return self.delete_from_beginning()
        
        current = self.head
        for _ in range(position - 1):
            current = current.next
        
        data = current.next.data
        current.next = current.next.next
        self.size -= 1
        return data
    
    def delete_by_value(self, value: Any) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.head:
            return False
        
        if self.head.data == value:
            self.delete_from_beginning()
            return True
        
        current = self.head
        while current.next:
            if current.next.data == value:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    # ==================== SEARCH OPERATIONS ====================
    
    def search(self, value: Any) -> int:
        """
        Find position of value in list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = self.head
        position = 0
        
        while current:
            if current.data == value:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_at_position(self, position: int) -> Optional[Any]:
        """
        Get element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position >= self.size:
            return None
        
        current = self.head
        for _ in range(position):
            current = current.next
        
        return current.data
    
    def get_head(self) -> Optional[Any]:
        """Get first element"""
        return self.head.data if self.head else None
    
    def get_tail(self) -> Optional[Any]:
        """Get last element"""
        if not self.head:
            return None
        
        current = self.head
        while current.next:
            current = current.next
        
        return current.data
    
    # ==================== UTILITY OPERATIONS ====================
    
    def reverse(self) -> None:
        """
        Reverse the linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    def to_list(self) -> List[Any]:
        """Convert to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def clear(self) -> None:
        """Clear all elements"""
        self.head = None
        self.size = 0

# ==================== DOUBLY LINKED LIST ====================

class DoublyNode:
    """Node for doubly linked list"""
    def __init__(self, data: Any):
        self.data = data
        self.next: Optional['DoublyNode'] = None
        self.prev: Optional['DoublyNode'] = None
    
    def __repr__(self):
        return f"Node({self.data})"

class DoublyLinkedList:
    """
    Doubly Linked List implementation with comprehensive operations
    """
    
    def __init__(self):
        self.head: Optional[DoublyNode] = None
        self.tail: Optional[DoublyNode] = None
        self.size: int = 0
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        if not self.head:
            return "[]"
        
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        return " <-> ".join(result)
    
    # ==================== BASIC OPERATIONS ====================
    
    def is_empty(self) -> bool:
        """
        Check if list is empty
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.head is None
    
    def prepend(self, data: Any) -> None:
        """
        Add element at the beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        new_node = DoublyNode(data)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def append(self, data: Any) -> None:
        """
        Add element at the end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        new_node = DoublyNode(data)
        
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_at_position(self, position: int, data: Any) -> bool:
        """
        Insert element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position > self.size:
            return False
        
        if position == 0:
            self.prepend(data)
            return True
        
        if position == self.size:
            self.append(data)
            return True
        
        new_node = DoublyNode(data)
        
        # Choose direction based on position
        if position <= self.size // 2:
            # Start from head
            current = self.head
            for _ in range(position):
                current = current.next
        else:
            # Start from tail
            current = self.tail
            for _ in range(self.size - position - 1):
                current = current.prev
        
        # Insert before current
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node
        current.prev = new_node
        
        self.size += 1
        return True
    
    def delete_from_beginning(self) -> Optional[Any]:
        """
        Delete element from beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.head:
            return None
        
        data = self.head.data
        
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        
        self.size -= 1
        return data
    
    def delete_from_end(self) -> Optional[Any]:
        """
        Delete element from end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not self.tail:
            return None
        
        data = self.tail.data
        
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        
        self.size -= 1
        return data
    
    def delete_at_position(self, position: int) -> Optional[Any]:
        """
        Delete element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position >= self.size:
            return None
        
        if position == 0:
            return self.delete_from_beginning()
        
        if position == self.size - 1:
            return self.delete_from_end()
        
        # Choose direction based on position
        if position <= self.size // 2:
            # Start from head
            current = self.head
            for _ in range(position):
                current = current.next
        else:
            # Start from tail
            current = self.tail
            for _ in range(self.size - position - 1):
                current = current.prev
        
        data = current.data
        current.prev.next = current.next
        current.next.prev = current.prev
        
        self.size -= 1
        return data
    
    # ==================== SEARCH OPERATIONS ====================
    
    def search(self, value: Any) -> int:
        """
        Find position of value in list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = self.head
        position = 0
        
        while current:
            if current.data == value:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_at_position(self, position: int) -> Optional[Any]:
        """
        Get element at specific position
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if position < 0 or position >= self.size:
            return None
        
        # Choose direction based on position
        if position <= self.size // 2:
            # Start from head
            current = self.head
            for _ in range(position):
                current = current.next
        else:
            # Start from tail
            current = self.tail
            for _ in range(self.size - position - 1):
                current = current.prev
        
        return current.data
    
    # ==================== UTILITY OPERATIONS ====================
    
    def reverse(self) -> None:
        """
        Reverse the doubly linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = self.head
        
        while current:
            # Swap next and prev
            current.next, current.prev = current.prev, current.next
            current = current.prev  # Move to next (which is now prev)
        
        # Swap head and tail
        self.head, self.tail = self.tail, self.head
    
    def to_list(self) -> List[Any]:
        """Convert to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def to_list_reverse(self) -> List[Any]:
        """Convert to Python list in reverse order"""
        result = []
        current = self.tail
        while current:
            result.append(current.data)
            current = current.prev
        return result

# ==================== CIRCULAR LINKED LIST ====================

class CircularLinkedList:
    """
    Circular Linked List implementation using singly linked nodes
    """
    
    def __init__(self):
        self.head: Optional[SinglyNode] = None
        self.size: int = 0
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        if not self.head:
            return "[]"
        
        result = []
        current = self.head
        for _ in range(self.size):
            result.append(str(current.data))
            current = current.next
        return " -> ".join(result) + " -> (back to head)"
    
    # ==================== BASIC OPERATIONS ====================
    
    def is_empty(self) -> bool:
        """
        Check if list is empty
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.head is None
    
    def append(self, data: Any) -> None:
        """
        Add element at the end
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        new_node = SinglyNode(data)
        
        if not self.head:
            self.head = new_node
            new_node.next = new_node  # Point to itself
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = new_node
            new_node.next = self.head
        
        self.size += 1
    
    def prepend(self, data: Any) -> None:
        """
        Add element at the beginning
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        new_node = SinglyNode(data)
        
        if not self.head:
            self.head = new_node
            new_node.next = new_node
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def delete_by_value(self, value: Any) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.head:
            return False
        
        # If only one node
        if self.head.next == self.head:
            if self.head.data == value:
                self.head = None
                self.size -= 1
                return True
            return False
        
        # If head needs to be deleted
        if self.head.data == value:
            # Find last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = self.head.next
            self.head = self.head.next
            self.size -= 1
            return True
        
        # Search for the node to delete
        current = self.head
        while current.next != self.head:
            if current.next.data == value:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def search(self, value: Any) -> int:
        """
        Find position of value in list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not self.head:
            return -1
        
        current = self.head
        position = 0
        
        for _ in range(self.size):
            if current.data == value:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def to_list(self) -> List[Any]:
        """Convert to Python list"""
        if not self.head:
            return []
        
        result = []
        current = self.head
        for _ in range(self.size):
            result.append(current.data)
            current = current.next
        return result

# ==================== LINKED LIST ALGORITHMS ====================

class LinkedListAlgorithms:
    """
    Advanced linked list algorithms and problem-solving techniques
    """
    
    def __init__(self):
        print("Linked List Algorithms Initialized")
        print("=" * 50)
    
    # ==================== CLASSIC PROBLEMS ====================
    
    def detect_cycle(self, head: Optional[SinglyNode]) -> bool:
        """
        Detect cycle in linked list using Floyd's algorithm
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return False
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def find_cycle_start(self, head: Optional[SinglyNode]) -> Optional[SinglyNode]:
        """
        Find the start of cycle in linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        # Phase 1: Detect cycle
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return None  # No cycle
        
        # Phase 2: Find start of cycle
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def find_middle(self, head: Optional[SinglyNode]) -> Optional[SinglyNode]:
        """
        Find middle node of linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None
        
        slow = fast = head
        
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def find_nth_from_end(self, head: Optional[SinglyNode], n: int) -> Optional[SinglyNode]:
        """
        Find nth node from end
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or n <= 0:
            return None
        
        first = second = head
        
        # Move first pointer n steps ahead
        for _ in range(n):
            if not first:
                return None
            first = first.next
        
        # Move both pointers until first reaches end
        while first:
            first = first.next
            second = second.next
        
        return second
    
    def merge_sorted_lists(self, l1: Optional[SinglyNode], l2: Optional[SinglyNode]) -> Optional[SinglyNode]:
        """
        Merge two sorted linked lists
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        dummy = SinglyNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.data <= l2.data:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        # Attach remaining nodes
        current.next = l1 or l2
        
        return dummy.next
    
    def remove_duplicates(self, head: Optional[SinglyNode]) -> Optional[SinglyNode]:
        """
        Remove duplicates from sorted linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = head
        
        while current and current.next:
            if current.data == current.next.data:
                current.next = current.next.next
            else:
                current = current.next
        
        return head
    
    def reverse_in_groups(self, head: Optional[SinglyNode], k: int) -> Optional[SinglyNode]:
        """
        Reverse linked list in groups of k
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or k <= 1:
            return head
        
        # Check if we have k nodes
        current = head
        count = 0
        while current and count < k:
            current = current.next
            count += 1
        
        if count == k:  # We have k nodes
            current = self.reverse_in_groups(current, k)  # Recursively reverse rest
            
            # Reverse first k nodes
            prev = current
            curr = head
            
            for _ in range(k):
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node
            
            head = prev
        
        return head
    
    def add_two_numbers(self, l1: Optional[SinglyNode], l2: Optional[SinglyNode]) -> Optional[SinglyNode]:
        """
        Add two numbers represented as linked lists
        
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        dummy = SinglyNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.data if l1 else 0
            val2 = l2.data if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            current.next = SinglyNode(digit)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next

def demonstrate_linked_lists():
    """
    Demonstrate various linked list operations and algorithms
    """
    print("\nLINKED LIST DEMONSTRATIONS")
    print("=" * 50)
    
    # ==================== SINGLY LINKED LIST ====================
    print("\nSINGLY LINKED LIST:")
    sll = SinglyLinkedList()
    
    # Basic operations
    print("Adding elements: 1, 2, 3")
    sll.append(1)
    sll.append(2)
    sll.append(3)
    print(f"List: {sll}")
    print(f"Size: {len(sll)}")
    
    print("Prepending 0:")
    sll.prepend(0)
    print(f"List: {sll}")
    
    print("Inserting 1.5 at position 2:")
    sll.insert_at_position(2, 1.5)
    print(f"List: {sll}")
    
    print("Searching for 2:")
    pos = sll.search(2)
    print(f"Found at position: {pos}")
    
    print("Deleting from beginning:")
    deleted = sll.delete_from_beginning()
    print(f"Deleted: {deleted}, List: {sll}")
    
    print("Reversing list:")
    sll.reverse()
    print(f"Reversed: {sll}")
    
    # ==================== DOUBLY LINKED LIST ====================
    print("\nDOUBLY LINKED LIST:")
    dll = DoublyLinkedList()
    
    print("Adding elements: A, B, C")
    dll.append('A')
    dll.append('B')
    dll.append('C')
    print(f"List: {dll}")
    
    print("Prepending Z:")
    dll.prepend('Z')
    print(f"List: {dll}")
    
    print("Inserting X at position 2:")
    dll.insert_at_position(2, 'X')
    print(f"List: {dll}")
    
    print("Forward traversal:", dll.to_list())
    print("Backward traversal:", dll.to_list_reverse())
    
    print("Deleting from end:")
    deleted = dll.delete_from_end()
    print(f"Deleted: {deleted}, List: {dll}")
    
    # ==================== CIRCULAR LINKED LIST ====================
    print("\nCIRCULAR LINKED LIST:")
    cll = CircularLinkedList()
    
    print("Adding elements: Red, Green, Blue")
    cll.append('Red')
    cll.append('Green')
    cll.append('Blue')
    print(f"List: {cll}")
    print(f"As list: {cll.to_list()}")
    
    print("Prepending Yellow:")
    cll.prepend('Yellow')
    print(f"List: {cll}")
    
    print("Deleting Green:")
    deleted = cll.delete_by_value('Green')
    print(f"Deleted: {deleted}, List: {cll}")
    
    # ==================== ALGORITHMS ====================
    print("\nLINKED LIST ALGORITHMS:")
    algos = LinkedListAlgorithms()
    
    # Create a simple linked list for algorithms
    head = SinglyNode(1)
    head.next = SinglyNode(2)
    head.next.next = SinglyNode(3)
    head.next.next.next = SinglyNode(4)
    head.next.next.next.next = SinglyNode(5)
    
    print("Finding middle of list [1,2,3,4,5]:")
    middle = algos.find_middle(head)
    print(f"Middle node: {middle.data}")
    
    print("Finding 2nd node from end:")
    nth_node = algos.find_nth_from_end(head, 2)
    print(f"2nd from end: {nth_node.data}")
    
    # Create sorted lists for merging
    l1 = SinglyNode(1)
    l1.next = SinglyNode(3)
    l1.next.next = SinglyNode(5)
    
    l2 = SinglyNode(2)
    l2.next = SinglyNode(4)
    l2.next.next = SinglyNode(6)
    
    print("Merging sorted lists [1,3,5] and [2,4,6]:")
    merged = algos.merge_sorted_lists(l1, l2)
    result = []
    current = merged
    while current:
        result.append(current.data)
        current = current.next
    print(f"Merged: {result}")

def linked_list_tips_and_tricks():
    """
    Tips for working with linked lists effectively
    """
    print("\nLINKED LIST TIPS AND TRICKS")
    print("=" * 50)
    
    tips = [
        "1. Use dummy nodes to simplify edge cases in algorithms",
        "2. Two pointers (slow/fast) for cycle detection and finding middle",
        "3. Keep track of previous node when deletion is needed",
        "4. Doubly linked lists allow O(1) deletion when node reference is known",
        "5. Circular lists are useful for round-robin scheduling",
        "6. Always check for null pointers to avoid segmentation faults",
        "7. Consider tail pointer for O(1) append operations",
        "8. Recursive solutions often lead to cleaner code",
        "9. Draw diagrams when working with complex pointer manipulations",
        "10. Practice reversing operations - key skill for many problems"
    ]
    
    for tip in tips:
        print(f"- {tip}")

if __name__ == "__main__":
    demonstrate_linked_lists()
    linked_list_tips_and_tricks()
    
    print("\nLINKED LIST STUDY COMPLETE!")
    print("Master these concepts to handle any linked list problem!")