"""
21. Merge Two Sorted Lists

You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.

Example 1:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:
Input: list1 = [], list2 = []
Output: []

Example 3:
Input: list1 = [], list2 = [0]
Output: [0]

Constraints:
- The number of nodes in both lists is in the range [0, 50]
- -100 <= Node.val <= 100
- Both list1 and list2 are sorted in non-decreasing order
"""

from typing import Optional, List


class ListNode:
    """Definition for singly-linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        """String representation for debugging."""
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)


def merge_two_lists_iterative(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative approach with dummy node.
    
    Time Complexity: O(n + m) where n and m are lengths of the lists
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Create a dummy node to simplify edge cases
    2. Use two pointers to traverse both lists
    3. Always choose the smaller value and advance that pointer
    4. Append remaining nodes from the non-empty list
    """
    dummy = ListNode(0)
    current = dummy
    
    # Merge nodes while both lists have elements
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # Append remaining nodes
    current.next = list1 or list2
    
    return dummy.next


def merge_two_lists_recursive(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n + m) where n and m are lengths of the lists
    Space Complexity: O(n + m) - recursion stack
    
    Algorithm:
    1. Base cases: if one list is empty, return the other
    2. Compare heads and recursively merge the smaller head with the rest
    3. Return the head of the merged list
    """
    # Base cases
    if not list1:
        return list2
    if not list2:
        return list1
    
    # Choose smaller head and recursively merge
    if list1.val <= list2.val:
        list1.next = merge_two_lists_recursive(list1.next, list2)
        return list1
    else:
        list2.next = merge_two_lists_recursive(list1, list2.next)
        return list2


def merge_two_lists_in_place(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    In-place merge without dummy node.
    
    Time Complexity: O(n + m) where n and m are lengths of the lists
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Handle edge cases first
    2. Determine which list should be the head
    3. Merge remaining nodes in place
    """
    # Handle edge cases
    if not list1:
        return list2
    if not list2:
        return list1
    
    # Ensure list1 starts with smaller value
    if list1.val > list2.val:
        list1, list2 = list2, list1
    
    head = list1
    
    # Merge remaining nodes
    while list1.next and list2:
        if list1.next.val <= list2.val:
            list1 = list1.next
        else:
            # Insert list2 node between list1 and list1.next
            temp = list1.next
            list1.next = list2
            list2 = list2.next
            list1.next.next = temp
            list1 = list1.next
    
    # Append remaining nodes from list2
    if list2:
        list1.next = list2
    
    return head


def merge_two_lists_stack(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Stack-based approach (alternative implementation).
    
    Time Complexity: O(n + m) where n and m are lengths of the lists
    Space Complexity: O(n + m) - stack space
    
    Algorithm:
    1. Use stack to collect nodes in reverse order
    2. Pop nodes to build the merged list
    """
    if not list1:
        return list2
    if not list2:
        return list1
    
    stack = []
    
    # Push all nodes to stack in reverse order
    while list1 or list2:
        if not list1:
            stack.append(list2)
            list2 = list2.next
        elif not list2:
            stack.append(list1)
            list1 = list1.next
        elif list1.val <= list2.val:
            stack.append(list1)
            list1 = list1.next
        else:
            stack.append(list2)
            list2 = list2.next
    
    # Build result by popping from stack
    if not stack:
        return None
    
    # Reverse the connections
    head = stack.pop()
    current = head
    
    while stack:
        current.next = stack.pop()
        current = current.next
    
    current.next = None
    return head


def merge_two_lists_priority_queue(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Priority queue approach (simulated without heapq).
    
    Time Complexity: O(n + m) where n and m are lengths of the lists
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Simulate priority queue behavior by always choosing minimum
    2. Use two pointers to track current positions
    """
    if not list1:
        return list2
    if not list2:
        return list1
    
    # Always choose the node with smaller value
    if list1.val <= list2.val:
        head = list1
        list1 = list1.next
    else:
        head = list2
        list2 = list2.next
    
    current = head
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # Append remaining nodes
    current.next = list1 or list2
    
    return head


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create a linked list from a list of values."""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert linked list to list for testing."""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result


def test_merge_two_lists():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,4], [1,3,4], [1,1,2,3,4,4]),
        ([], [], []),
        ([], [0], [0]),
        ([0], [], [0]),
        
        # Single elements
        ([1], [2], [1,2]),
        ([2], [1], [1,2]),
        ([1], [1], [1,1]),
        
        # Different lengths
        ([1,2,3], [4,5,6,7,8], [1,2,3,4,5,6,7,8]),
        ([4,5,6,7,8], [1,2,3], [1,2,3,4,5,6,7,8]),
        
        # All same values
        ([1,1,1], [1,1,1], [1,1,1,1,1,1]),
        
        # Interleaved
        ([1,3,5,7], [2,4,6,8], [1,2,3,4,5,6,7,8]),
        
        # One list much smaller
        ([5], [1,2,3,4,6,7,8], [1,2,3,4,5,6,7,8]),
        
        # Negative numbers
        ([-10,-5,0], [-7,-2,3], [-10,-7,-5,-2,0,3]),
        
        # Large ranges
        ([1,10,100], [2,20,200], [1,2,10,20,100,200]),
    ]
    
    # Test all implementations
    implementations = [
        ("Iterative", merge_two_lists_iterative),
        ("Recursive", merge_two_lists_recursive),
        ("In-place", merge_two_lists_in_place),
        ("Stack", merge_two_lists_stack),
        ("Priority Queue", merge_two_lists_priority_queue),
    ]
    
    print("Testing Merge Two Sorted Lists implementations:")
    print("=" * 60)
    
    for i, (list1_vals, list2_vals, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {list1_vals} + {list2_vals}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked lists for each implementation
                list1 = create_linked_list(list1_vals)
                list2 = create_linked_list(list2_vals)
                
                result_head = func(list1, list2)
                result = linked_list_to_list(result_head)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test case
    large_list1 = list(range(0, 1000, 2))  # [0, 2, 4, ..., 998]
    large_list2 = list(range(1, 1000, 2))  # [1, 3, 5, ..., 999]
    
    for name, func in implementations:
        try:
            list1 = create_linked_list(large_list1)
            list2 = create_linked_list(large_list2)
            
            start_time = time.time()
            result_head = func(list1, list2)
            end_time = time.time()
            
            result_length = len(linked_list_to_list(result_head))
            print(f"{name}: Merged {result_length} nodes in {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_merge_two_lists() 