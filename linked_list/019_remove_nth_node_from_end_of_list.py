"""
19. Remove Nth Node From End of List

Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:
Input: head = [1], n = 1
Output: []

Example 3:
Input: head = [1,2], n = 1
Output: [1]

Constraints:
- The number of nodes in the list is sz
- 1 <= sz <= 30
- 0 <= Node.val <= 100
- 1 <= n <= sz

Follow up: Could you do this in one pass?
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


def remove_nth_from_end_two_pass(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Two-pass approach: First pass to count nodes, second pass to remove.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. First pass: Count the total number of nodes
    2. Second pass: Remove the (L-n+1)th node from the beginning
    3. Handle edge case where we need to remove the head
    """
    # First pass: count nodes
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Calculate position from the beginning
    position_from_start = length - n
    
    # Handle edge case: removing the head
    if position_from_start == 0:
        return head.next
    
    # Second pass: find and remove the node
    current = head
    for _ in range(position_from_start - 1):
        current = current.next
    
    # Remove the node
    current.next = current.next.next
    
    return head


def remove_nth_from_end_one_pass(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    One-pass approach using two pointers with n+1 gap.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use two pointers: fast and slow
    2. Move fast pointer n+1 steps ahead
    3. Move both pointers until fast reaches the end
    4. Remove the node after slow pointer
    """
    dummy = ListNode(0)
    dummy.next = head
    fast = dummy
    slow = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        if fast:
            fast = fast.next
    
    # Move both pointers until fast reaches the end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove the node
    slow.next = slow.next.next
    
    return dummy.next


def remove_nth_from_end_stack(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Stack-based approach to track nodes.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(L) - stack to store all nodes
    
    Algorithm:
    1. Push all nodes to a stack
    2. Pop n nodes to find the target node
    3. Update the previous node's next pointer
    """
    if not head:
        return None
    
    stack = []
    current = head
    
    # Push all nodes to stack
    while current:
        stack.append(current)
        current = current.next
    
    # Pop n nodes to find the target
    for _ in range(n):
        target = stack.pop()
    
    # Handle edge case: removing the head
    if not stack:
        return head.next
    
    # Update the previous node's next pointer
    prev = stack[-1]
    prev.next = target.next
    
    return head


def remove_nth_from_end_recursive(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Recursive approach with counter.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(L) - recursion stack
    
    Algorithm:
    1. Use recursion to reach the end of the list
    2. Count nodes on the way back
    3. Remove the node when counter reaches n
    """
    def helper(node):
        if not node:
            return 0
        
        # Count from the end
        count = helper(node.next) + 1
        
        # If this is the nth node from the end, remove it
        if count == n + 1:
            node.next = node.next.next
        
        return count
    
    dummy = ListNode(0)
    dummy.next = head
    helper(dummy)
    
    return dummy.next


def remove_nth_from_end_array(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Array-based approach for comparison.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(L) - array to store all nodes
    
    Algorithm:
    1. Store all nodes in an array
    2. Calculate the index of the node to remove
    3. Update the previous node's next pointer
    """
    if not head:
        return None
    
    nodes = []
    current = head
    
    # Store all nodes
    while current:
        nodes.append(current)
        current = current.next
    
    # Calculate index to remove (0-based from start)
    index_to_remove = len(nodes) - n
    
    # Handle edge case: removing the head
    if index_to_remove == 0:
        return head.next
    
    # Update the previous node's next pointer
    nodes[index_to_remove - 1].next = nodes[index_to_remove].next
    
    return head


def remove_nth_from_end_slow_fast(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Alternative two-pointer approach with different initialization.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Initialize both pointers at head
    2. Move fast pointer n steps ahead
    3. Move both pointers until fast reaches the end
    4. Remove the node after slow pointer
    """
    if not head:
        return None
    
    fast = head
    slow = head
    
    # Move fast pointer n steps ahead
    for _ in range(n):
        if fast:
            fast = fast.next
    
    # If fast is None, we need to remove the head
    if not fast:
        return head.next
    
    # Move both pointers until fast reaches the end
    while fast.next:
        fast = fast.next
        slow = slow.next
    
    # Remove the node
    slow.next = slow.next.next
    
    return head


def remove_nth_from_end_iterative_reverse(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Approach using list reversal (for educational purposes).
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Reverse the linked list
    2. Remove the nth node from the beginning
    3. Reverse the list back
    """
    def reverse_list(node):
        prev = None
        current = node
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        return prev
    
    # Reverse the list
    reversed_head = reverse_list(head)
    
    # Remove nth node from the beginning
    if n == 1:
        reversed_head = reversed_head.next
    else:
        current = reversed_head
        for _ in range(n - 2):
            current = current.next
        current.next = current.next.next
    
    # Reverse back
    return reverse_list(reversed_head)


def remove_nth_from_end_length_based(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Length-based approach with early termination.
    
    Time Complexity: O(L) where L is the length of the list
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Calculate the length of the list
    2. Use the length to find the position to remove
    3. Traverse to that position and remove the node
    """
    if not head:
        return None
    
    # Calculate length
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Position from the beginning (0-based)
    pos = length - n
    
    # Handle edge case: removing the head
    if pos == 0:
        return head.next
    
    # Find the node before the one to remove
    current = head
    for _ in range(pos - 1):
        current = current.next
    
    # Remove the node
    current.next = current.next.next
    
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


def test_remove_nth_from_end():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3,4,5], 2, [1,2,3,5]),
        ([1], 1, []),
        ([1,2], 1, [1]),
        ([1,2], 2, [2]),
        
        # Edge cases
        ([1,2,3], 3, [2,3]),  # Remove first
        ([1,2,3], 1, [1,2]),  # Remove last
        ([1,2,3], 2, [1,3]),  # Remove middle
        
        # Single element
        ([42], 1, []),
        
        # Two elements
        ([1,2], 1, [1]),
        ([1,2], 2, [2]),
        
        # Longer lists
        ([1,2,3,4,5,6,7,8,9,10], 5, [1,2,3,4,5,7,8,9,10]),
        ([1,2,3,4,5,6,7,8,9,10], 1, [1,2,3,4,5,6,7,8,9]),
        ([1,2,3,4,5,6,7,8,9,10], 10, [2,3,4,5,6,7,8,9,10]),
        
        # Same values
        ([1,1,1,1,1], 3, [1,1,1,1]),
        ([0,0,0], 2, [0,0]),
        
        # Large values
        ([100,99,98,97,96], 3, [100,99,97,96]),
        
        # Sequential
        ([1,2,3,4,5,6], 4, [1,2,4,5,6]),
        ([10,20,30,40,50], 2, [10,20,30,50]),
    ]
    
    # Test all implementations
    implementations = [
        ("Two Pass", remove_nth_from_end_two_pass),
        ("One Pass", remove_nth_from_end_one_pass),
        ("Stack", remove_nth_from_end_stack),
        ("Recursive", remove_nth_from_end_recursive),
        ("Array", remove_nth_from_end_array),
        ("Slow Fast", remove_nth_from_end_slow_fast),
        ("Iterative Reverse", remove_nth_from_end_iterative_reverse),
        ("Length Based", remove_nth_from_end_length_based),
    ]
    
    print("Testing Remove Nth Node From End implementations:")
    print("=" * 60)
    
    for i, (values, n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}, n={n}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head, n)
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
        ("Small list", list(range(1, 6)), 2),
        ("Medium list", list(range(1, 21)), 10),
        ("Large list", list(range(1, 101)), 50),
        ("Remove first", list(range(1, 51)), 50),
        ("Remove last", list(range(1, 51)), 1),
    ]
    
    for scenario_name, values, n in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head, n)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} nodes remaining in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_remove_nth_from_end() 