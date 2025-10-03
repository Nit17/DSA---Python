"""
Test script to verify all DSA modules are working correctly
"""

import sys
import os

# Add the DSA directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'DSA'))

def test_time_space_complexity():
    """Test time and space complexity module"""
    print("Testing Time/Space Complexity module...")
    try:
        from time_space_complexity import binary_search
        # Quick test using available function
        result = binary_search([1, 2, 3, 4, 5], 3)
        assert result == 2, "Binary search failed"
        print("✓ Time/Space Complexity module working")
        return True
    except Exception as e:
        print(f"✗ Time/Space Complexity module failed: {e}")
        return False

def test_recursion():
    """Test recursion module"""
    print("Testing Recursion module...")
    try:
        from recursion import RecursionExamples
        recursion = RecursionExamples()
        # Quick test
        result = recursion.factorial(5)
        assert result == 120, "Factorial failed"
        print("✓ Recursion module working")
        return True
    except Exception as e:
        print(f"✗ Recursion module failed: {e}")
        return False

def test_arrays_lists():
    """Test arrays and lists module"""
    print("Testing Arrays/Lists module...")
    try:
        from arrays_lists import ArrayOperations
        operations = ArrayOperations()
        # Quick test
        result = operations.linear_search([1, 2, 3, 4, 5], 3)
        assert result == 2, "Array linear search failed"
        print("✓ Arrays/Lists module working")
        return True
    except Exception as e:
        print(f"✗ Arrays/Lists module failed: {e}")
        return False

def test_strings():
    """Test strings module"""
    print("Testing Strings module...")
    try:
        from strings import StringOperations
        operations = StringOperations()
        # Quick test
        result = operations.reverse_string("hello")
        assert result == "olleh", "String reverse failed"
        print("✓ Strings module working")
        return True
    except Exception as e:
        print(f"✗ Strings module failed: {e}")
        return False

def test_linked_lists():
    """Test linked lists module"""
    print("Testing Linked Lists module...")
    try:
        from linked_lists import SinglyLinkedList
        sll = SinglyLinkedList()
        sll.append(1)
        sll.append(2)
        sll.append(3)
        # Quick test
        assert len(sll) == 3, "Linked list size failed"
        assert sll.get_head() == 1, "Linked list head failed"
        print("✓ Linked Lists module working")
        return True
    except Exception as e:
        print(f"✗ Linked Lists module failed: {e}")
        return False

def test_stacks():
    """Test stacks module"""
    print("Testing Stacks module...")
    try:
        from stacks import ArrayStack, MinStack, StackAlgorithms
        s = ArrayStack()
        s.push(10); s.push(20)
        assert s.peek() == 20, "ArrayStack peek failed"
        assert s.pop() == 20 and s.peek() == 10, "ArrayStack pop failed"

        ms = MinStack()
        for v in [3, 5, 2, 2, 4]:
            ms.push(v)
        assert ms.get_min() == 2, "MinStack get_min failed"
        ms.pop(); ms.pop();
        assert ms.get_min() == 2, "MinStack get_min after pops failed"
        ms.pop();
        assert ms.get_min() == 3, "MinStack get_min after more pops failed"

        algo = StackAlgorithms()
        assert algo.is_valid_parentheses("{[()]}()"), "Parentheses validation failed"
        print("✓ Stacks module working")
        return True
    except Exception as e:
        print(f"✗ Stacks module failed: {e}")
        return False

def test_queues():
    """Test queues module"""
    print("Testing Queues module...")
    try:
        from queues import ArrayQueue, LinkedListQueue, CircularQueue, ArrayDeque, PriorityQueue, QueueAlgorithms
        
        # Test ArrayQueue
        aq = ArrayQueue(5)
        aq.enqueue(1); aq.enqueue(2); aq.enqueue(3)
        assert aq.front() == 1 and aq.rear() == 3, "ArrayQueue front/rear failed"
        assert aq.dequeue() == 1 and aq.front() == 2, "ArrayQueue dequeue failed"
        
        # Test LinkedListQueue
        llq = LinkedListQueue()
        llq.enqueue('A'); llq.enqueue('B')
        assert llq.front() == 'A' and llq.rear() == 'B', "LinkedListQueue front/rear failed"
        
        # Test CircularQueue
        cq = CircularQueue(3)
        cq.enqueue(10); cq.enqueue(20); cq.enqueue(30)
        assert cq.is_full(), "CircularQueue is_full failed"
        assert cq.dequeue() == 10, "CircularQueue dequeue failed"
        
        # Test ArrayDeque
        dq = ArrayDeque()
        dq.add_front(2); dq.add_rear(3); dq.add_front(1)
        assert dq.front() == 1 and dq.rear() == 3, "ArrayDeque front/rear failed"
        
        # Test PriorityQueue
        pq = PriorityQueue()
        pq.enqueue('Task C', 3); pq.enqueue('Task A', 1)
        assert pq.dequeue() == 'Task A', "PriorityQueue dequeue failed"
        
        # Test algorithms
        algo = QueueAlgorithms()
        graph = {'A': ['B'], 'B': []}
        assert algo.bfs_graph(graph, 'A') == ['A', 'B'], "BFS algorithm failed"
        
        nums = [1, 3, -1, -3, 5]
        result = algo.sliding_window_maximum(nums, 3)
        assert result == [3, 3, 5], "Sliding window maximum failed"
        
        print("✓ Queues module working")
        return True
    except Exception as e:
        print(f"✗ Queues module failed: {e}")
        return False

def test_hashing():
    """Test hashing module"""
    print("Testing Hashing module...")
    try:
        from hashing import ChainingHashTable, LinearProbingHashTable, HashingAlgorithms
        # Test chaining table
        cht = ChainingHashTable()
        cht.put('apple', 1)
        cht.put('banana', 2)
        assert cht.get('apple') == 1, "Chaining get failed"
        cht.put('apple', 5)
        assert cht.get('apple') == 5, "Chaining update failed"
        # Test linear probing
        lp = LinearProbingHashTable()
        lp.put(10, 'X'); lp.put(18, 'Y')  # likely collision
        assert lp.get(10) == 'X' and lp.get(18) == 'Y', "Linear probing get failed"
        lp.remove(10)
        assert lp.get(10) is None, "Linear probing remove failed"
        # Algorithms
        alg = HashingAlgorithms()
        assert alg.two_sum([2,7,11,15], 9) == (0,1), "Two Sum failed"
        assert alg.longest_unique_substring('abcabcbb') == 3, "Longest unique substring failed"
        assert alg.longest_consecutive([100,4,200,1,3,2]) == 4, "Longest consecutive sequence failed"
        print("✓ Hashing module working")
        return True
    except Exception as e:
        print(f"✗ Hashing module failed: {e}")
        return False

def test_trees():
    """Test tree-related data structures (BST, Heaps, Trie, SegmentTree, FenwickTree)."""
    print("Testing Trees/Heaps/Trie/Segment/Fenwick module group...")
    try:
        from trees.bst import BinarySearchTree
        from trees.heaps import MinHeap, MaxHeap
        from trees.trie import Trie
        from trees.segment_tree import SegmentTree
        from trees.fenwick_tree import FenwickTree
        from trees.binary_tree import BinaryTreeNode, inorder as bt_inorder

        # BST
        bst = BinarySearchTree()
        for v in [8,3,10,1,6,14,4,7,13]:
            bst.insert(v)
        assert bst.search(6) and not bst.search(99), "BST search failed"
        before_delete = bst.inorder()
        bst.delete(3)
        after_delete = bst.inorder()
        assert sorted(before_delete) == before_delete, "BST inorder not sorted"
        assert 3 not in after_delete, "BST delete failed"
        assert bst.is_valid(), "BST validation failed"

        # MinHeap / MaxHeap
        mh = MinHeap([5,3,8,1])
        mh.push(0)
        popped = [mh.pop() for _ in range(len(mh))]
        assert popped == sorted(popped), "MinHeap order incorrect"
        xh = MaxHeap([5,3,8,1])
        xh.push(10)
        max_first = xh.pop()
        assert max_first == 10, "MaxHeap max extraction failed"

        # Trie
        tr = Trie()
        for w in ["apple","app","apex","bat"]:
            tr.insert(w)
        assert tr.search("app") and not tr.search("apply"), "Trie search failed"
        assert tr.starts_with("ap"), "Trie prefix failed"
        tr.delete("app")
        assert not tr.search("app") and tr.search("apple"), "Trie delete failed"

        # Segment Tree
        st = SegmentTree([2,1,5,3,4])
        assert st.range_sum(0,4) == 15 and st.range_sum(1,3) == 9, "SegmentTree range sum failed"
        st.update(2,10)
        assert st.range_sum(0,4) == 20 and st.range_sum(2,2) == 10, "SegmentTree update failed"

        # Fenwick Tree
        ft = FenwickTree.build([2,1,5,3,4])
        assert ft.prefix_sum(2) == 8 and ft.range_sum(1,3) == 9, "Fenwick prefix/range failed"
        ft.update(2,5)
        assert ft.range_sum(0,4) == 20, "Fenwick update failed"

        # Binary Tree basic traversal
        root = BinaryTreeNode(1, BinaryTreeNode(2), BinaryTreeNode(3))
        assert bt_inorder(root) == [2,1,3], "Binary tree inorder failed"

        print("✓ Trees/Heaps/Trie/Segment/Fenwick structures working")
        return True
    except Exception as e:
        print(f"✗ Trees/Heaps/Trie/Segment/Fenwick group failed: {e}")
        return False

def test_graphs():
    """Test graphs module"""
    print("Testing Graphs module...")
    try:
        # Import from graphs package (graphs/__init__.py exposes symbols)
        from graphs import (
            AdjacencyListGraph, AdjacencyMatrixGraph, GraphAlgorithms,
            WeightedAdjacencyListGraph, dijkstra, bellman_ford,
            floyd_warshall, dijkstra_with_path, bellman_ford_with_path,
            reconstruct_fw_path
        )

        # Test AdjacencyListGraph
        g_list = AdjacencyListGraph()
        g_list.add_edge(0, 1)
        g_list.add_edge(0, 2)
        g_list.add_edge(1, 2)
        g_list.add_edge(1, 3)
        assert 0 in g_list.adj_list and 1 in g_list.adj_list[0], "AdjacencyList add_edge failed"
        
        # Test AdjacencyMatrixGraph
        g_matrix = AdjacencyMatrixGraph(4)
        g_matrix.add_edge(0, 1)
        g_matrix.add_edge(0, 2)
        g_matrix.add_edge(1, 2)
        g_matrix.add_edge(1, 3)
        assert g_matrix.matrix[0][1] == 1 and g_matrix.matrix[1][0] == 1, "AdjacencyMatrix add_edge failed"
        
        # Test GraphAlgorithms
        alg = GraphAlgorithms()
        
        # BFS
        bfs_result = alg.bfs(g_list, 0)
        assert set(bfs_result) == {0, 1, 2, 3}, "BFS traversal failed"
        
        # DFS
        dfs_result = alg.dfs(g_list, 0)
        assert set(dfs_result) == {0, 1, 2, 3}, "DFS traversal failed"
        
        # Shortest path (unweighted)
        path = alg.shortest_path_unweighted(g_list, 0, 3)
        assert path == [0, 1, 3], "Shortest path failed"

        # Weighted graph tests
        wg = WeightedAdjacencyListGraph(directed=True)
        for u,v,w in [
            ('A','B',4), ('A','C',2), ('C','B',1), ('B','D',5),
            ('C','D',8), ('C','E',10), ('D','E',2)
        ]:
            wg.add_edge(u,v,w)
        dist_dij = dijkstra(wg,'A')
        # Shortest paths: A->C (2), C->B (1) => B=3; A->C (2) + C->B (1) + B->D (5) => D=8
        assert dist_dij['D'] == 8, "Dijkstra distance incorrect"
        d_cost, d_path = dijkstra_with_path(wg,'A','E')
        assert d_path and d_path[0] == 'A' and d_path[-1] == 'E', "Dijkstra path endpoints wrong"

        dist_bf, neg_cycle = bellman_ford(wg,'A')
        assert not neg_cycle and dist_bf['D'] == 8, "Bellman-Ford basic failed"
        bf_cost, bf_path, neg2 = bellman_ford_with_path(wg,'A','E')
        assert not neg2 and bf_path and bf_path[0] == 'A' and bf_path[-1] == 'E', "Bellman-Ford path failed"

        all_dist, nxt = floyd_warshall(wg)
        fw_path = reconstruct_fw_path(nxt,'A','E')
        assert all_dist['A']['D'] == 8 and fw_path and fw_path[0] == 'A' and fw_path[-1] == 'E', "Floyd-Warshall failed"
        
        print("✓ Graphs module working")
        return True
    except Exception as e:
        print(f"✗ Graphs module failed: {e}")
        return False

def main():
    """Run all tests"""
    print("DSA-Python Module Verification")
    print("=" * 40)
    
    tests = [
        test_time_space_complexity,
        test_recursion,
        test_arrays_lists,
        test_strings,
        test_linked_lists,
        test_stacks,
        test_queues,
        test_hashing,
        test_trees,
        test_graphs,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} modules working correctly")
    
    if passed == total:
        print("🎉 All modules are working perfectly!")
        print("Your DSA-Python repository is ready for use!")
    else:
        print("⚠️  Some modules need attention.")
    
    return passed == total

if __name__ == "__main__":
    main()