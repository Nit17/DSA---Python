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
        from time_space_complexity import ComplexityAnalysis
        analyzer = ComplexityAnalysis()
        # Quick test
        result = analyzer.linear_search([1, 2, 3, 4, 5], 3)
        assert result == 2, "Linear search failed"
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

def main():
    """Run all tests"""
    print("DSA-Python Module Verification")
    print("=" * 40)
    
    tests = [
        test_time_space_complexity,
        test_recursion,
        test_arrays_lists,
        test_strings,
        test_linked_lists
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