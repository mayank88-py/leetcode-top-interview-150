"""
83. Remove Duplicates from Sorted List

Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

Example 1:
Input: head = [1,1,2]
Output: [1,2]

Example 2:
Input: head = [1,1,2,3,3]
Output: [1,2,3]

Constraints:
- The number of nodes in the list is in the range [0, 300]
- -100 <= Node.val <= 100
- The list is guaranteed to be sorted in ascending order
"""

from typing import Optional, List, Set


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


def delete_duplicates_iterative(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative approach using single pointer.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Traverse the list with a single pointer
    2. Compare current node with next node
    3. If values are equal, skip the next node
    4. Continue until end of list
    """
    current = head
    
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    
    return head


def delete_duplicates_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Base case: if head is None or head.next is None, return head
    2. If current value equals next value, skip next and recurse
    3. Otherwise, recursively process the rest
    """
    if not head or not head.next:
        return head
    
    if head.val == head.next.val:
        # Skip the next node and recurse
        head.next = head.next.next
        return delete_duplicates_recursive(head)
    else:
        # Process the rest recursively
        head.next = delete_duplicates_recursive(head.next)
        return head


def delete_duplicates_two_pointers(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Two pointers approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers: current and next
    2. Move next pointer to find different value
    3. Connect current to next when different value found
    """
    if not head:
        return None
    
    current = head
    
    while current:
        next_different = current.next
        
        # Find next different value
        while next_different and next_different.val == current.val:
            next_different = next_different.next
        
        # Connect to next different node
        current.next = next_different
        current = next_different
    
    return head


def delete_duplicates_hash_set(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Hash set approach (works for unsorted lists too).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hash set to store seen values
    
    Algorithm:
    1. Use hash set to track seen values
    2. Traverse list and keep only first occurrence
    3. Remove subsequent duplicates
    """
    if not head:
        return None
    
    seen = set()
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    current = head
    
    while current:
        if current.val not in seen:
            seen.add(current.val)
            prev = current
            current = current.next
        else:
            # Remove duplicate
            prev.next = current.next
            current = current.next
    
    return dummy.next


def delete_duplicates_while_loop(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    While loop approach with explicit duplicate removal.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. For each node, use while loop to remove all duplicates
    2. Move to next unique value
    """
    current = head
    
    while current:
        # Remove all duplicates of current value
        while current.next and current.val == current.next.val:
            current.next = current.next.next
        
        current = current.next
    
    return head


def delete_duplicates_array(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Array-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - array to store unique values
    
    Algorithm:
    1. Extract all values to array
    2. Remove duplicates from array
    3. Rebuild linked list
    """
    if not head:
        return None
    
    values = []
    current = head
    
    # Extract values
    while current:
        values.append(current.val)
        current = current.next
    
    # Remove duplicates while preserving order
    unique_values = []
    for val in values:
        if not unique_values or unique_values[-1] != val:
            unique_values.append(val)
    
    # Rebuild list
    dummy = ListNode(0)
    current = dummy
    
    for val in unique_values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next


def delete_duplicates_dummy_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Dummy node approach for consistent handling.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use dummy node to handle edge cases
    2. Traverse and remove duplicates
    3. Return dummy.next
    """
    if not head:
        return None
    
    dummy = ListNode(0)
    dummy.next = head
    current = head
    
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    
    return dummy.next


def delete_duplicates_count_based(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Count-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Count consecutive duplicate nodes
    2. Skip all but the first occurrence
    """
    if not head:
        return None
    
    current = head
    
    while current:
        count = 1
        temp = current.next
        
        # Count consecutive duplicates
        while temp and temp.val == current.val:
            count += 1
            temp = temp.next
        
        # Skip all duplicates except the first
        current.next = temp
        current = temp
    
    return head


def delete_duplicates_functional(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Use functional programming style
    2. Process list recursively
    3. Build result list
    """
    def remove_duplicates_helper(node, prev_val):
        if not node:
            return None
        
        if node.val == prev_val:
            # Skip this duplicate
            return remove_duplicates_helper(node.next, prev_val)
        else:
            # Keep this node
            new_node = ListNode(node.val)
            new_node.next = remove_duplicates_helper(node.next, node.val)
            return new_node
    
    if not head:
        return None
    
    # Start with first node
    result = ListNode(head.val)
    result.next = remove_duplicates_helper(head.next, head.val)
    return result


def delete_duplicates_tail_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Tail recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Use tail recursion for efficiency
    2. Process list from end to beginning
    """
    def tail_helper(current, result):
        if not current:
            return result
        
        if not result or current.val != result.val:
            new_node = ListNode(current.val)
            new_node.next = result
            return tail_helper(current.next, new_node)
        else:
            return tail_helper(current.next, result)
    
    # Reverse the result since we're building backwards
    def reverse_list(node):
        prev = None
        current = node
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    if not head:
        return None
    
    result = tail_helper(head, None)
    return reverse_list(result)


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


def test_delete_duplicates():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,1,2], [1,2]),
        ([1,1,2,3,3], [1,2,3]),
        ([], []),
        ([1], [1]),
        
        # All duplicates
        ([1,1,1,1], [1]),
        ([2,2,2], [2]),
        
        # No duplicates
        ([1,2,3,4,5], [1,2,3,4,5]),
        ([1,2,3], [1,2,3]),
        
        # Mixed patterns
        ([1,1,2,2,3,3], [1,2,3]),
        ([1,2,2,3,3,3,4], [1,2,3,4]),
        
        # Large gaps
        ([1,1,10,10,20,20], [1,10,20]),
        ([5,5,100,100,200], [5,100,200]),
        
        # Negative values
        ([-3,-3,-1,-1,0,0,1,1], [-3,-1,0,1]),
        ([-1,-1,-1], [-1]),
        
        # Zero values
        ([0,0,0,1,1,2], [0,1,2]),
        ([0,0,1], [0,1]),
        
        # Single duplicates
        ([1,2,2,3], [1,2,3]),
        ([1,1,2,3], [1,2,3]),
        
        # Multiple duplicate groups
        ([1,1,1,2,2,3,3,3,3], [1,2,3]),
        ([1,2,2,2,3,4,4,5,5,5], [1,2,3,4,5]),
        
        # Large values
        ([100,100,200,200,300], [100,200,300]),
        ([-100,-100,-50,-50,0,0,50,50,100,100], [-100,-50,0,50,100]),
        
        # Long sequences
        ([1,1,1,1,1,1,1,2,2,2,2,3,3,3], [1,2,3]),
        ([5,5,5,5,5,5,5,5,5,5], [5]),
        
        # Edge cases
        ([1,2], [1,2]),
        ([1,1], [1]),
        ([1,2,3,4,5,5], [1,2,3,4,5]),
    ]
    
    # Test all implementations
    implementations = [
        ("Iterative", delete_duplicates_iterative),
        ("Recursive", delete_duplicates_recursive),
        ("Two Pointers", delete_duplicates_two_pointers),
        ("Hash Set", delete_duplicates_hash_set),
        ("While Loop", delete_duplicates_while_loop),
        ("Array", delete_duplicates_array),
        ("Dummy Node", delete_duplicates_dummy_node),
        ("Count Based", delete_duplicates_count_based),
        ("Functional", delete_duplicates_functional),
        ("Tail Recursive", delete_duplicates_tail_recursive),
    ]
    
    print("Testing Remove Duplicates from Sorted List implementations:")
    print("=" * 60)
    
    for i, (values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head)
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
    
    # Generate large test cases
    test_scenarios = [
        ("No duplicates", list(range(100))),
        ("All duplicates", [1] * 100),
        ("Mixed duplicates", [i//5 for i in range(100)]),
        ("Few duplicates", [i//2 for i in range(50)]),
        ("Large values", [i*100 for i in range(20)] * 5),
    ]
    
    for scenario_name, values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} unique nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_delete_duplicates() 