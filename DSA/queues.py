"""
QUEUES - Implementations, Algorithms, and Patterns
=================================================

A queue is a FIFO (First-In-First-Out) data structure that supports enqueue 
(add to rear) and dequeue (remove from front) operations.

Queue Types Covered:
1. ArrayQueue: Simple list-based implementation
2. LinkedListQueue: Node-based implementation
3. CircularQueue: Fixed-size circular buffer
4. Deque (Double-ended Queue): Add/remove from both ends
5. PriorityQueue: Elements ordered by priority

Common Use Cases:
- Breadth-First Search (BFS) traversal
- Process scheduling and task queues
- Sliding window maximum/minimum problems
- Level-order tree traversal
- Cache implementation (LRU with deque)
- Print job queues and buffering

Time Complexities:
- Enqueue/Dequeue: O(1) for most implementations
- Peek/Front: O(1)
- Search: O(n) (need to traverse)
- Priority operations depend on underlying heap
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Deque as DequeType
from collections import deque
import heapq

# ==================== ARRAY-BASED QUEUE ====================

class ArrayQueue:
    """
    Array (list)-backed Queue implementation using two pointers.
    
    Optimized to avoid shifting elements on dequeue by using front pointer.
    """
    
    def __init__(self, capacity: int = 100) -> None:
        self._data: List[Optional[Any]] = [None] * capacity
        self._front: int = 0
        self._size: int = 0
        self._capacity: int = capacity
    
    def __len__(self) -> int:
        return self._size
    
    def __str__(self) -> str:
        if self.is_empty():
            return "Queue([])"
        items = []
        for i in range(self._size):
            items.append(str(self._data[(self._front + i) % self._capacity]))
        return f"Queue([{', '.join(items)}])"
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full. O(1)"""
        return self._size == self._capacity
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear of queue. Returns False if full. O(1)"""
        if self.is_full():
            return False
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = item
        self._size += 1
        return True
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return front item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        item = self._data[self._front]
        self._data[self._front] = None  # Help GC
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item
    
    def front(self) -> Optional[Any]:
        """Return front item without removing. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._data[self._front]
    
    def rear(self) -> Optional[Any]:
        """Return rear item without removing. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        rear_index = (self._front + self._size - 1) % self._capacity
        return self._data[rear_index]
    
    def size(self) -> int:
        """Return current size. O(1)"""
        return self._size
    
    def clear(self) -> None:
        """Remove all items. O(1)"""
        self._front = 0
        self._size = 0
        for i in range(self._capacity):
            self._data[i] = None

# ==================== LINKED LIST QUEUE ====================

class _QueueNode:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.next: Optional['_QueueNode'] = None

class LinkedListQueue:
    """
    Linked list-backed Queue implementation.
    
    Uses head for dequeue and tail for enqueue operations.
    """
    
    def __init__(self) -> None:
        self._head: Optional[_QueueNode] = None
        self._tail: Optional[_QueueNode] = None
        self._size: int = 0
    
    def __len__(self) -> int:
        return self._size
    
    def __str__(self) -> str:
        if self.is_empty():
            return "Queue([])"
        items = []
        current = self._head
        while current:
            items.append(str(current.data))
            current = current.next
        return f"Queue([{', '.join(items)}])"
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return self._head is None
    
    def enqueue(self, item: Any) -> None:
        """Add item to rear of queue. O(1)"""
        new_node = _QueueNode(item)
        if self.is_empty():
            self._head = self._tail = new_node
        else:
            self._tail.next = new_node
            self._tail = new_node
        self._size += 1
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return front item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        
        item = self._head.data
        self._head = self._head.next
        if self._head is None:  # Queue became empty
            self._tail = None
        self._size -= 1
        return item
    
    def front(self) -> Optional[Any]:
        """Return front item without removing. Returns None if empty. O(1)"""
        return self._head.data if self._head else None
    
    def rear(self) -> Optional[Any]:
        """Return rear item without removing. Returns None if empty. O(1)"""
        return self._tail.data if self._tail else None
    
    def size(self) -> int:
        """Return current size. O(1)"""
        return self._size
    
    def clear(self) -> None:
        """Remove all items. O(1)"""
        self._head = self._tail = None
        self._size = 0

# ==================== CIRCULAR QUEUE ====================

class CircularQueue:
    """
    Circular Queue implementation with fixed capacity.
    
    Efficiently uses space with circular indexing.
    """
    
    def __init__(self, capacity: int) -> None:
        self._data: List[Optional[Any]] = [None] * (capacity + 1)  # +1 to distinguish full/empty
        self._front: int = 0
        self._rear: int = 0
        self._capacity: int = capacity + 1
    
    def __len__(self) -> int:
        return (self._rear - self._front + self._capacity) % self._capacity
    
    def __str__(self) -> str:
        if self.is_empty():
            return "CircularQueue([])"
        items = []
        current = self._front
        while current != self._rear:
            items.append(str(self._data[current]))
            current = (current + 1) % self._capacity
        return f"CircularQueue([{', '.join(items)}])"
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return self._front == self._rear
    
    def is_full(self) -> bool:
        """Check if queue is full. O(1)"""
        return (self._rear + 1) % self._capacity == self._front
    
    def enqueue(self, item: Any) -> bool:
        """Add item to rear. Returns False if full. O(1)"""
        if self.is_full():
            return False
        
        self._data[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        return True
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return front item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        
        item = self._data[self._front]
        self._data[self._front] = None  # Help GC
        self._front = (self._front + 1) % self._capacity
        return item
    
    def front(self) -> Optional[Any]:
        """Return front item without removing. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._data[self._front]
    
    def rear(self) -> Optional[Any]:
        """Return rear item without removing. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        rear_index = (self._rear - 1 + self._capacity) % self._capacity
        return self._data[rear_index]

# ==================== DEQUE (Double-ended Queue) ====================

class ArrayDeque:
    """
    Double-ended queue allowing insertion/deletion at both ends.
    
    Uses Python's collections.deque internally for efficiency.
    """
    
    def __init__(self) -> None:
        self._data: DequeType[Any] = deque()
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __str__(self) -> str:
        return f"Deque({list(self._data)})"
    
    def is_empty(self) -> bool:
        """Check if deque is empty. O(1)"""
        return len(self._data) == 0
    
    # Front operations
    def add_front(self, item: Any) -> None:
        """Add item to front. O(1)"""
        self._data.appendleft(item)
    
    def remove_front(self) -> Optional[Any]:
        """Remove and return front item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._data.popleft()
    
    def front(self) -> Optional[Any]:
        """Return front item without removing. Returns None if empty. O(1)"""
        return self._data[0] if self._data else None
    
    # Rear operations
    def add_rear(self, item: Any) -> None:
        """Add item to rear. O(1)"""
        self._data.append(item)
    
    def remove_rear(self) -> Optional[Any]:
        """Remove and return rear item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._data.pop()
    
    def rear(self) -> Optional[Any]:
        """Return rear item without removing. Returns None if empty. O(1)"""
        return self._data[-1] if self._data else None
    
    # Convenience methods (queue interface)
    def enqueue(self, item: Any) -> None:
        """Add to rear (queue behavior). O(1)"""
        self.add_rear(item)
    
    def dequeue(self) -> Optional[Any]:
        """Remove from front (queue behavior). O(1)"""
        return self.remove_front()
    
    def clear(self) -> None:
        """Remove all items. O(n)"""
        self._data.clear()

# ==================== PRIORITY QUEUE ====================

class PriorityQueue:
    """
    Priority Queue implementation using a min-heap.
    
    Elements with lower priority values are dequeued first.
    """
    
    def __init__(self) -> None:
        self._heap: List[Tuple[Any, Any]] = []  # (priority, item) pairs
        self._counter: int = 0  # For stable ordering of equal priorities
    
    def __len__(self) -> int:
        return len(self._heap)
    
    def __str__(self) -> str:
        items = [f"{item}(p:{priority})" for priority, _, item in self._heap]
        return f"PriorityQueue({items})"
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return len(self._heap) == 0
    
    def enqueue(self, item: Any, priority: Any = 0) -> None:
        """Add item with priority. Lower priority = higher precedence. O(log n)"""
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return highest priority item. Returns None if empty. O(log n)"""
        if self.is_empty():
            return None
        _, _, item = heapq.heappop(self._heap)
        return item
    
    def peek(self) -> Optional[Any]:
        """Return highest priority item without removing. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._heap[0][2]
    
    def peek_priority(self) -> Optional[Any]:
        """Return priority of highest priority item. Returns None if empty. O(1)"""
        if self.is_empty():
            return None
        return self._heap[0][0]

# ==================== QUEUE ALGORITHMS ====================

class QueueAlgorithms:
    """Queue-based algorithms and problem-solving patterns."""
    
    # ==================== BFS ALGORITHMS ====================
    
    def bfs_tree_level_order(self, root) -> List[List[Any]]:
        """
        Level-order traversal of binary tree using queue.
        Returns list of levels, each level is a list of node values.
        
        Time: O(n), Space: O(w) where w is max width
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val if hasattr(node, 'val') else node.data)
                
                if hasattr(node, 'left') and node.left:
                    queue.append(node.left)
                if hasattr(node, 'right') and node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    def bfs_graph(self, graph: dict, start: Any) -> List[Any]:
        """
        BFS traversal of graph represented as adjacency list.
        
        Time: O(V + E), Space: O(V)
        """
        if start not in graph:
            return []
        
        visited = set()
        queue = deque([start])
        result = []
        
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def shortest_path_unweighted(self, graph: dict, start: Any, target: Any) -> Optional[List[Any]]:
        """
        Find shortest path in unweighted graph using BFS.
        
        Time: O(V + E), Space: O(V)
        """
        if start == target:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            vertex, path = queue.popleft()
            
            for neighbor in graph.get(vertex, []):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    # ==================== SLIDING WINDOW WITH DEQUE ====================
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """
        Find maximum in each sliding window of size k using deque.
        Maintains indices in decreasing order of their values.
        
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        result = []
        window = deque()  # Store indices
        
        for i, num in enumerate(nums):
            # Remove indices outside current window
            while window and window[0] <= i - k:
                window.popleft()
            
            # Remove indices whose values are smaller than current
            while window and nums[window[-1]] < num:
                window.pop()
            
            window.append(i)
            
            # Add to result when window is full
            if i >= k - 1:
                result.append(nums[window[0]])
        
        return result
    
    def sliding_window_minimum(self, nums: List[int], k: int) -> List[int]:
        """
        Find minimum in each sliding window of size k using deque.
        
        Time: O(n), Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        result = []
        window = deque()  # Store indices
        
        for i, num in enumerate(nums):
            # Remove indices outside current window
            while window and window[0] <= i - k:
                window.popleft()
            
            # Remove indices whose values are larger than current
            while window and nums[window[-1]] > num:
                window.pop()
            
            window.append(i)
            
            # Add to result when window is full
            if i >= k - 1:
                result.append(nums[window[0]])
        
        return result
    
    # ==================== TASK SCHEDULING ====================
    
    def task_scheduler(self, tasks: List[str], n: int) -> int:
        """
        CPU task scheduling with cooldown period using priority queue.
        
        Time: O(m log k), Space: O(k) where k is unique tasks, m is total time
        """
        from collections import Counter
        
        task_counts = Counter(tasks)
        heap = [-count for count in task_counts.values()]
        heapq.heapify(heap)
        
        time = 0
        
        while heap:
            cycle_tasks = []
            
            # Execute tasks for one cooldown cycle
            for _ in range(n + 1):
                if heap:
                    count = heapq.heappop(heap)
                    cycle_tasks.append(count + 1)  # Decrease count (was negative)
                time += 1
                
                # If no more tasks, we're done
                if not heap and all(c == 0 for c in cycle_tasks):
                    break
            
            # Add back tasks that still need execution
            for count in cycle_tasks:
                if count < 0:  # Still has remaining executions
                    heapq.heappush(heap, count)
        
        return time
    
    # ==================== CACHE ALGORITHMS ====================
    
    def lru_cache_simulation(self, capacity: int, operations: List[Tuple[str, Any]]) -> List[Any]:
        """
        Simulate LRU cache using deque for ordering and dict for fast access.
        
        Operations: [('get', key), ('put', key, value), ...]
        Returns: List of results for get operations, None for put operations
        """
        cache = {}  # key -> value
        order = deque()  # Most recent at right
        results = []
        
        def move_to_end(key):
            order.remove(key)
            order.append(key)
        
        for op, *args in operations:
            if op == 'get':
                key = args[0]
                if key in cache:
                    move_to_end(key)
                    results.append(cache[key])
                else:
                    results.append(None)
            
            elif op == 'put':
                key, value = args[0], args[1]
                if key in cache:
                    cache[key] = value
                    move_to_end(key)
                else:
                    if len(cache) >= capacity:
                        # Evict least recently used
                        lru_key = order.popleft()
                        del cache[lru_key]
                    
                    cache[key] = value
                    order.append(key)
                
                results.append(None)
        
        return results

# ==================== DEMONSTRATION ====================

def demonstrate_queues() -> None:
    print("\nQUEUE DEMONSTRATIONS")
    print("=" * 50)
    
    # ArrayQueue
    print("\nArrayQueue:")
    aq = ArrayQueue(5)
    for i in [1, 2, 3]:
        aq.enqueue(i)
    print(f"Front: {aq.front()}, Rear: {aq.rear()}, Size: {aq.size()}")
    print(f"Dequeue: {aq.dequeue()}, New Front: {aq.front()}")
    print(aq)
    
    # LinkedListQueue
    print("\nLinkedListQueue:")
    llq = LinkedListQueue()
    for ch in ['A', 'B', 'C']:
        llq.enqueue(ch)
    print(f"Front: {llq.front()}, Rear: {llq.rear()}, Size: {llq.size()}")
    print(f"Dequeue: {llq.dequeue()}, New Front: {llq.front()}")
    print(llq)
    
    # CircularQueue
    print("\nCircularQueue:")
    cq = CircularQueue(4)
    for i in [10, 20, 30, 40]:
        success = cq.enqueue(i)
        print(f"Enqueue {i}: {'Success' if success else 'Failed (Full)'}")
    print(f"Is Full: {cq.is_full()}")
    print(f"Dequeue: {cq.dequeue()}")
    cq.enqueue(50)
    print(cq)
    
    # ArrayDeque
    print("\nArrayDeque:")
    dq = ArrayDeque()
    dq.add_front(2)
    dq.add_rear(3)
    dq.add_front(1)
    dq.add_rear(4)
    print(f"Deque: {dq}")
    print(f"Remove front: {dq.remove_front()}, Remove rear: {dq.remove_rear()}")
    print(f"After removals: {dq}")
    
    # PriorityQueue
    print("\nPriorityQueue:")
    pq = PriorityQueue()
    tasks = [('Task C', 3), ('Task A', 1), ('Task B', 2), ('Task D', 1)]
    for task, priority in tasks:
        pq.enqueue(task, priority)
    print(f"Priority Queue: {pq}")
    print("Dequeue order:", end=" ")
    while not pq.is_empty():
        print(pq.dequeue(), end=" ")
    print()
    
    # Algorithms
    print("\nQueue Algorithms:")
    algo = QueueAlgorithms()
    
    # BFS Graph
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    print(f"BFS from A: {algo.bfs_graph(graph, 'A')}")
    print(f"Shortest path A->F: {algo.shortest_path_unweighted(graph, 'A', 'F')}")
    
    # Sliding Window Maximum
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Sliding window max ({nums}, k={k}): {algo.sliding_window_maximum(nums, k)}")
    
    # LRU Cache Simulation
    operations = [
        ('put', 1, 1), ('put', 2, 2), ('get', 1), ('put', 3, 3),
        ('get', 2), ('put', 4, 4), ('get', 1), ('get', 3), ('get', 4)
    ]
    lru_results = algo.lru_cache_simulation(2, operations)
    print(f"LRU Cache results: {[r for r in lru_results if r is not None]}")

def queue_tips_and_tricks() -> None:
    print("\nQUEUE TIPS AND TRICKS")
    print("=" * 50)
    tips = [
        "Use queues for BFS traversal and level-order processing",
        "Deques are perfect for sliding window maximum/minimum problems",
        "Priority queues excel at task scheduling and pathfinding algorithms",
        "Circular queues efficiently use fixed memory for buffering",
        "Use two pointers in array queues to avoid shifting elements",
        "Linked list queues have no capacity limit but use more memory",
        "Monotonic deques solve window extrema problems in O(n) time",
        "BFS guarantees shortest path in unweighted graphs",
        "Priority queue + greedy often solves optimization problems",
        "Deque-based LRU cache provides O(1) get and put operations"
    ]
    for tip in tips:
        print(f"- {tip}")

if __name__ == "__main__":
    demonstrate_queues()
    queue_tips_and_tricks()