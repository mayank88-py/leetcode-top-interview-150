"""
92. Reverse Linked List II

Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

Example 1:
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Example 2:
Input: head = [5], left = 1, right = 1
Output: [5]

Constraints:
- The number of nodes in the list is n
- 1 <= n <= 500
- -500 <= Node.val <= 500
- 1 <= left <= right <= n

Follow up: Could you do it in one pass?
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


def reverse_between_iterative(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Iterative approach with one pass.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use dummy node to handle edge cases
    2. Find the node before the reversal start
    3. Reverse the sublist from left to right
    4. Connect the reversed sublist back
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move to the node before the reversal start
    for _ in range(left - 1):
        prev = prev.next
    
    # Start of the reversal
    current = prev.next
    
    # Reverse the sublist
    for _ in range(right - left):
        next_node = current.next
        current.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
    
    return dummy.next


def reverse_between_stack(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Stack-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(right - left + 1) - stack for the reversed part
    
    Algorithm:
    1. Traverse to the left position
    2. Push nodes to stack from left to right
    3. Pop from stack to reverse and reconnect
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move to the node before the reversal start
    for _ in range(left - 1):
        prev = prev.next
    
    # Start of the reversal
    current = prev.next
    stack = []
    
    # Push nodes to stack
    for _ in range(right - left + 1):
        stack.append(current)
        current = current.next
    
    # Pop from stack and reconnect
    while stack:
        node = stack.pop()
        prev.next = node
        prev = node
    
    prev.next = current
    
    return dummy.next


def reverse_between_recursive(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Use recursion to reach the left position
    2. Reverse the sublist recursively
    3. Reconnect the parts
    """
    if not head or left == right:
        return head
    
    def reverse_n(node, n):
        """Reverse first n nodes of the list starting from node."""
        if n == 1:
            return node
        
        last = reverse_n(node.next, n - 1)
        next_node = node.next.next
        node.next.next = node
        node.next = next_node
        
        return last
    
    if left == 1:
        return reverse_n(head, right)
    
    head.next = reverse_between_recursive(head.next, left - 1, right - 1)
    return head


def reverse_between_array(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Array-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - array to store all nodes
    
    Algorithm:
    1. Store all nodes in an array
    2. Reverse the subarray from left to right
    3. Reconnect all nodes
    """
    if not head or left == right:
        return head
    
    nodes = []
    current = head
    
    # Store all nodes
    while current:
        nodes.append(current)
        current = current.next
    
    # Reverse the subarray (convert to 0-based indexing)
    nodes[left-1:right] = nodes[left-1:right][::-1]
    
    # Reconnect nodes
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    nodes[-1].next = None
    
    return nodes[0]


def reverse_between_three_pointers(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Three pointers approach for clear logic.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Find the three key positions: before left, at left, after right
    2. Reverse the middle portion
    3. Reconnect all parts
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    
    # Find the node before the reversal start
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next
    
    # Store the start of reversal
    start = prev.next
    
    # Find the end of reversal
    end = start
    for _ in range(right - left):
        end = end.next
    
    # Store the node after the reversal
    next_node = end.next
    
    # Break the connection
    end.next = None
    
    # Reverse the sublist
    def reverse_list(node):
        prev_node = None
        current = node
        
        while current:
            next_temp = current.next
            current.next = prev_node
            prev_node = current
            current = next_temp
        
        return prev_node
    
    # Reverse and reconnect
    new_start = reverse_list(start)
    prev.next = new_start
    start.next = next_node
    
    return dummy.next


def reverse_between_two_pass(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Two-pass approach: first pass to find positions, second to reverse.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. First pass: find the nodes at key positions
    2. Second pass: reverse the sublist
    3. Reconnect the parts
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    
    # First pass: find positions
    nodes = []
    current = dummy
    for i in range(right + 1):
        nodes.append(current)
        if current.next:
            current = current.next
    
    # Get key nodes
    before_left = nodes[left - 1]
    left_node = nodes[left]
    right_node = nodes[right]
    after_right = right_node.next
    
    # Break connections
    before_left.next = None
    right_node.next = None
    
    # Reverse the sublist
    def reverse_list(node):
        prev_node = None
        current = node
        
        while current:
            next_temp = current.next
            current.next = prev_node
            prev_node = current
            current = next_temp
        
        return prev_node
    
    # Reverse and reconnect
    new_start = reverse_list(left_node)
    before_left.next = new_start
    left_node.next = after_right
    
    return dummy.next


def reverse_between_iterative_clean(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Clean iterative approach with clear variable names.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use dummy node for edge cases
    2. Find the connection point
    3. Reverse nodes one by one
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    connection_point = dummy
    
    # Move to the node before the reversal
    for _ in range(left - 1):
        connection_point = connection_point.next
    
    # Start of the section to reverse
    tail_of_reversed = connection_point.next
    
    # Reverse the section
    for _ in range(right - left):
        node_to_move = tail_of_reversed.next
        tail_of_reversed.next = node_to_move.next
        node_to_move.next = connection_point.next
        connection_point.next = node_to_move
    
    return dummy.next


def reverse_between_values(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Value-based approach (extracts values and rebuilds).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - to store values
    
    Algorithm:
    1. Extract all values from the list
    2. Reverse the values in the specified range
    3. Rebuild the list with reversed values
    """
    if not head or left == right:
        return head
    
    # Extract values
    values = []
    current = head
    while current:
        values.append(current.val)
        current = current.next
    
    # Reverse the sublist (convert to 0-based indexing)
    values[left-1:right] = values[left-1:right][::-1]
    
    # Rebuild the list
    dummy = ListNode(0)
    current = dummy
    for val in values:
        current.next = ListNode(val)
        current = current.next
    
    return dummy.next


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


def test_reverse_between():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3,4,5], 2, 4, [1,4,3,2,5]),
        ([5], 1, 1, [5]),
        
        # Edge cases
        ([1,2,3,4,5], 1, 5, [5,4,3,2,1]),  # Reverse entire list
        ([1,2,3,4,5], 1, 1, [1,2,3,4,5]),  # Reverse single element
        ([1,2,3,4,5], 5, 5, [1,2,3,4,5]),  # Reverse last element
        
        # Two elements
        ([1,2], 1, 2, [2,1]),
        ([1,2], 1, 1, [1,2]),
        ([1,2], 2, 2, [1,2]),
        
        # Three elements
        ([1,2,3], 1, 2, [2,1,3]),
        ([1,2,3], 2, 3, [1,3,2]),
        ([1,2,3], 1, 3, [3,2,1]),
        
        # Longer lists
        ([1,2,3,4,5,6,7,8,9,10], 3, 7, [1,2,7,6,5,4,3,8,9,10]),
        ([1,2,3,4,5,6,7,8,9,10], 1, 10, [10,9,8,7,6,5,4,3,2,1]),
        ([1,2,3,4,5,6,7,8,9,10], 5, 8, [1,2,3,4,8,7,6,5,9,10]),
        
        # Same values
        ([1,1,1,1,1], 2, 4, [1,1,1,1,1]),
        ([2,2,2,2,2], 1, 5, [2,2,2,2,2]),
        
        # Negative values
        ([-1,-2,-3,-4,-5], 2, 4, [-1,-4,-3,-2,-5]),
        ([1,-2,3,-4,5], 1, 5, [5,-4,3,-2,1]),
        
        # Large values
        ([100,200,300,400,500], 2, 4, [100,400,300,200,500]),
        
        # Adjacent reversals
        ([1,2,3,4,5,6], 2, 3, [1,3,2,4,5,6]),
        ([1,2,3,4,5,6], 4, 5, [1,2,3,5,4,6]),
    ]
    
    # Test all implementations
    implementations = [
        ("Iterative", reverse_between_iterative),
        ("Stack", reverse_between_stack),
        ("Recursive", reverse_between_recursive),
        ("Array", reverse_between_array),
        ("Three Pointers", reverse_between_three_pointers),
        ("Two Pass", reverse_between_two_pass),
        ("Iterative Clean", reverse_between_iterative_clean),
        ("Values", reverse_between_values),
    ]
    
    print("Testing Reverse Linked List II implementations:")
    print("=" * 60)
    
    for i, (values, left, right, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}, left={left}, right={right}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head, left, right)
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
        ("Small reversal", list(range(1, 21)), 5, 15),
        ("Large reversal", list(range(1, 101)), 20, 80),
        ("Reverse entire", list(range(1, 51)), 1, 50),
        ("Reverse beginning", list(range(1, 51)), 1, 25),
        ("Reverse end", list(range(1, 51)), 26, 50),
    ]
    
    for scenario_name, values, left, right in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head, left, right)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_reverse_between() 